"""Expert training module for MoE system."""

import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from peft import (
    LoraConfig,
    get_peft_model,
    TaskType,
    PeftModel
)
import wandb
import os
import json
import logging
from typing import Dict, Optional, Tuple
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from .config import MoETrainingConfig
from .data_loader import MoEDataLoader

logger = logging.getLogger(__name__)


class ExpertTrainer:
    """Trainer for individual MoE experts."""
    
    def __init__(self, config: MoETrainingConfig, expert_name: str):
        self.config = config
        self.expert_name = expert_name
        self.expert_config = config.get_expert_config(expert_name)
        self.training_config = config.get_training_config_for_expert(expert_name)
        
        # Initialize components
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.data_loader = None
        
        # Training state
        self.training_history = []
        self.best_metrics = {}
        
    def setup(self):
        """Set up tokenizer, model, and data."""
        logger.info(f"Setting up {self.expert_name} expert trainer")
        
        # Initialize tokenizer
        self._setup_tokenizer()
        
        # Initialize model
        self._setup_model()
        
        # Initialize data loader
        self._setup_data()
        
        # Setup logging
        self._setup_logging()
        
        logger.info(f"Setup complete for {self.expert_name} expert")
    
    def _setup_tokenizer(self):
        """Initialize and configure tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.base_model_name,
            cache_dir=self.config.system.cache_dir
        )
        
        # Set padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        
        logger.info(f"Tokenizer loaded: {len(self.tokenizer)} tokens")
    
    def _setup_model(self):
        """Initialize and configure model with LoRA."""
        # Load base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.config.model.base_model_name,
            cache_dir=self.config.system.cache_dir,
            torch_dtype=torch.bfloat16 if self.config.training.bf16 else torch.float16,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Resize token embeddings if needed
        if len(self.tokenizer) != base_model.config.vocab_size:
            base_model.resize_token_embeddings(len(self.tokenizer))
        
        # Configure LoRA
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.model.lora_r,
            lora_alpha=self.config.model.lora_alpha,
            lora_dropout=self.config.model.lora_dropout,
            target_modules=self.config.model.target_modules,
            bias="none"
        )
        
        # Apply LoRA
        self.model = get_peft_model(base_model, lora_config)
        
        # Print trainable parameters
        self.model.print_trainable_parameters()
        
        logger.info(f"Model loaded with LoRA: {self.config.model.base_model_name}")
    
    def _setup_data(self):
        """Set up data loaders."""
        self.data_loader = MoEDataLoader(self.config, self.tokenizer)
        self.train_loader, self.val_loader = self.data_loader.prepare_expert_data(self.expert_name)
        
        # Print data statistics
        stats = self.data_loader.get_dataset_stats(self.expert_name)
        logger.info(f"Data loaded: {stats['train_size']} train, {stats['val_size']} val examples")
    
    def _setup_logging(self):
        """Set up Weights & Biases logging."""
        if wandb.run is None:
            wandb.init(
                project=self.config.system.wandb_project,
                entity=self.config.system.wandb_entity,
                name=f"{self.expert_name}_expert",
                config={
                    "expert": self.expert_name,
                    "model": self.config.model.base_model_name,
                    "lora_r": self.config.model.lora_r,
                    "learning_rate": self.training_config.learning_rate,
                    "batch_size": self.training_config.batch_size,
                    "epochs": self.training_config.num_epochs
                }
            )
    
    def train(self) -> Dict:
        """Train the expert model."""
        logger.info(f"Starting training for {self.expert_name} expert")
        
        # Create output directory
        os.makedirs(self.expert_config.output_dir, exist_ok=True)
        
        # Configure training arguments
        training_args = TrainingArguments(
            output_dir=self.expert_config.output_dir,
            num_train_epochs=self.training_config.num_epochs,
            per_device_train_batch_size=self.training_config.batch_size,
            per_device_eval_batch_size=self.training_config.batch_size,
            gradient_accumulation_steps=self.training_config.gradient_accumulation_steps,
            learning_rate=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
            warmup_ratio=self.training_config.warmup_ratio,
            max_grad_norm=self.training_config.max_grad_norm,
            
            # Optimization
            optim=self.training_config.optimizer,
            lr_scheduler_type=self.training_config.lr_scheduler_type,
            fp16=self.training_config.fp16,
            bf16=self.training_config.bf16,
            gradient_checkpointing=self.training_config.gradient_checkpointing,
            dataloader_num_workers=self.training_config.dataloader_num_workers,
            
            # Evaluation and logging
            evaluation_strategy="steps",
            eval_steps=self.training_config.eval_steps,
            save_steps=self.training_config.save_steps,
            logging_steps=self.training_config.logging_steps,
            
            # Checkpointing
            save_total_limit=self.config.system.save_total_limit,
            load_best_model_at_end=self.config.system.load_best_model_at_end,
            metric_for_best_model=self.config.system.metric_for_best_model,
            greater_is_better=self.config.system.greater_is_better,
            
            # Logging
            report_to="wandb",
            run_name=f"{self.expert_name}_expert",
            
            # Memory optimization
            remove_unused_columns=self.training_config.remove_unused_columns,
            dataloader_pin_memory=True,
        )
        
        # Create trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=self.data_loader.datasets[self.expert_name]['train'],
            eval_dataset=self.data_loader.datasets[self.expert_name]['val'],
            tokenizer=self.tokenizer,
            compute_metrics=self._compute_metrics,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.training_config.early_stopping_patience,
                    early_stopping_threshold=self.training_config.early_stopping_threshold
                )
            ]
        )
        
        # Train model
        train_result = self.trainer.train()
        
        # Save final model
        self.trainer.save_model()
        self.tokenizer.save_pretrained(self.expert_config.output_dir)
        
        # Save training metrics
        self._save_training_results(train_result)
        
        logger.info(f"Training completed for {self.expert_name} expert")
        
        return train_result.metrics
    
    def _compute_metrics(self, eval_pred) -> Dict:
        """Compute evaluation metrics."""
        predictions, labels = eval_pred
        
        # Calculate perplexity
        predictions = predictions.reshape(-1, predictions.shape[-1])
        labels = labels.reshape(-1)
        
        # Filter out ignored tokens
        mask = labels != -100
        predictions = predictions[mask]
        labels = labels[mask]
        
        # Calculate loss manually for perplexity
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(predictions, labels)
        perplexity = torch.exp(loss).item()
        
        # Calculate accuracy
        pred_ids = torch.argmax(predictions, dim=-1)
        accuracy = accuracy_score(labels.cpu(), pred_ids.cpu())
        
        return {
            "perplexity": perplexity,
            "accuracy": accuracy,
            "eval_loss": loss.item()
        }
    
    def _save_training_results(self, train_result):
        """Save training results and metadata."""
        results = {
            "expert_name": self.expert_name,
            "model_name": self.config.model.base_model_name,
            "training_metrics": train_result.metrics,
            "config": {
                "lora_r": self.config.model.lora_r,
                "lora_alpha": self.config.model.lora_alpha,
                "learning_rate": self.training_config.learning_rate,
                "batch_size": self.training_config.batch_size,
                "epochs": self.training_config.num_epochs
            },
            "data_stats": self.data_loader.get_dataset_stats(self.expert_name)
        }
        
        # Save to file
        results_file = Path(self.expert_config.output_dir) / "training_results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Training results saved to {results_file}")
    
    def evaluate(self) -> Dict:
        """Evaluate the trained model."""
        if self.trainer is None:
            raise ValueError("Model not trained yet. Call train() first.")
        
        eval_results = self.trainer.evaluate()
        
        logger.info(f"Evaluation results for {self.expert_name}:")
        for key, value in eval_results.items():
            logger.info(f"  {key}: {value:.4f}")
        
        return eval_results
    
    def generate_sample(self, prompt: str, max_length: int = 200) -> str:
        """Generate a sample response from the trained expert."""
        if self.model is None:
            raise ValueError("Model not loaded. Call setup() first.")
        
        # Tokenize prompt
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove prompt from output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
    
    def save_for_inference(self, output_path: str):
        """Save model in format optimized for inference."""
        if self.model is None:
            raise ValueError("Model not loaded.")
        
        # Save LoRA adapter
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        
        # Save metadata
        metadata = {
            "expert_name": self.expert_name,
            "base_model": self.config.model.base_model_name,
            "lora_config": {
                "r": self.config.model.lora_r,
                "alpha": self.config.model.lora_alpha,
                "target_modules": self.config.model.target_modules
            },
            "training_config": {
                "learning_rate": self.training_config.learning_rate,
                "epochs": self.training_config.num_epochs
            }
        }
        
        with open(Path(output_path) / "expert_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Expert saved for inference: {output_path}")


class MoETrainingOrchestrator:
    """Orchestrates training of all MoE experts."""
    
    def __init__(self, config: MoETrainingConfig):
        self.config = config
        self.expert_trainers = {}
        self.training_results = {}
    
    def setup_all_experts(self):
        """Set up all expert trainers."""
        logger.info("Setting up all expert trainers")
        
        for expert_name in self.config.experts.keys():
            trainer = ExpertTrainer(self.config, expert_name)
            trainer.setup()
            self.expert_trainers[expert_name] = trainer
        
        logger.info(f"Set up {len(self.expert_trainers)} expert trainers")
    
    def train_all_experts(self, sequential: bool = True) -> Dict:
        """Train all experts."""
        logger.info("Starting training for all experts")
        
        if sequential:
            # Train experts one by one (recommended for single H100)
            for expert_name, trainer in self.expert_trainers.items():
                logger.info(f"Training {expert_name} expert")
                metrics = trainer.train()
                self.training_results[expert_name] = metrics
                
                # Clear GPU memory between experts
                torch.cuda.empty_cache()
        else:
            # Parallel training (requires multiple GPUs)
            raise NotImplementedError("Parallel training not implemented yet")
        
        logger.info("All experts trained successfully")
        return self.training_results
    
    def evaluate_all_experts(self) -> Dict:
        """Evaluate all trained experts."""
        evaluation_results = {}
        
        for expert_name, trainer in self.expert_trainers.items():
            eval_metrics = trainer.evaluate()
            evaluation_results[expert_name] = eval_metrics
        
        return evaluation_results
    
    def save_all_experts(self, base_output_dir: str):
        """Save all experts for inference."""
        for expert_name, trainer in self.expert_trainers.items():
            output_path = Path(base_output_dir) / f"{expert_name}_expert"
            trainer.save_for_inference(str(output_path))
        
        # Save overall training summary
        summary = {
            "training_results": self.training_results,
            "config": self.config.__dict__,
            "experts": list(self.config.experts.keys())
        }
        
        with open(Path(base_output_dir) / "moe_training_summary.json", 'w') as f:
            json.dump(summary, f, indent=2, default=str)
        
        logger.info(f"All experts saved to {base_output_dir}")
    
    def generate_samples(self, prompts: Dict[str, str]) -> Dict[str, str]:
        """Generate sample responses from all experts."""
        samples = {}
        
        for expert_name, trainer in self.expert_trainers.items():
            if expert_name in prompts:
                sample = trainer.generate_sample(prompts[expert_name])
                samples[expert_name] = sample
        
        return samples
