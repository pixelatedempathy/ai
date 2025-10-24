#!/usr/bin/env python3
"""
Aggressive training for lowest possible loss
"""

import json
import torch
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import BitsAndBytesConfig, DataCollatorForLanguageModeling
from transformers.trainer_callback import TrainerCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from tqdm import tqdm

class WandBCallback(TrainerCallback):
    def __init__(self):
        self.step_count = 0
        
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if logs:
            self.step_count += 1
            current_loss = logs.get('train_loss', 0)
            eval_loss = logs.get('eval_loss', 0)
            
            if 'epoch' in logs:
                progress = (logs['epoch'] / args.num_train_epochs) * 100
                print(f"📊 Progress: {progress:.1f}% | Train Loss: {current_loss:.4f} | Eval Loss: {eval_loss:.4f} | Step: {self.step_count}")
            
            enhanced_logs = logs.copy()
            enhanced_logs.update({
                'training/steps_completed': self.step_count,
                'training/progress_percent': (logs.get('epoch', 0) / args.num_train_epochs) * 100 if args.num_train_epochs else 0
            })
            
            wandb.log(enhanced_logs, step=state.global_step)

def setup_wandb():
    with open('wandb_config.json', 'r') as f:
        config = json.load(f)
    
    return wandb.init(
        project=config['project'],
        name=f"{config['name']}-aggressive",
        tags=config['tags'] + ['aggressive', 'low-loss'],
        notes="Aggressive training for minimum loss",
        config=config['config']
    )

def main():
    print("🔥 Aggressive Low-Loss Training")
    
    try:
        wandb_run = setup_wandb()
        
        print("📊 Loading dataset...")
        with open('training_dataset.json', 'r') as f:
            data = json.load(f)
        
        # Use full dataset
        texts = [conv['text'] for conv in tqdm(data['conversations'], desc="Loading")]
        dataset = Dataset.from_dict({"text": texts})
        
        print(f"📊 Dataset: {len(dataset)} samples")
        
        wandb.log({
            'dataset/total_conversations': len(texts),
            'dataset/avg_length': sum(len(text.split()) for text in texts) / len(texts)
        })
        
        # 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        MODEL_NAME = "LatitudeGames/Wayfarer-2-12B"
        
        print("🚀 Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        model = prepare_model_for_kbit_training(model)
        
        # High-rank LoRA for maximum adaptation
        lora_config = LoraConfig(
            r=96,
            lora_alpha=192,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)
        
        def tokenize_function(examples):
            result = tokenizer(
                examples["text"], 
                truncation=True, 
                padding=False,  # Dynamic padding
                max_length=896
            )
            result["labels"] = result["input_ids"].copy()
            return result
        
        print("🔤 Tokenizing...")
        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        
        # Data collator for dynamic padding
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # Aggressive training configuration
        training_args = TrainingArguments(
            output_dir="./wayfarer-aggressive",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=32,  # Large effective batch size
            num_train_epochs=5,
            learning_rate=5e-5,  # Lower LR for stability
            weight_decay=0.01,
            warmup_steps=500,
            lr_scheduler_type="cosine_with_restarts",
            bf16=True,
            logging_steps=50,
            save_steps=1000,
            eval_steps=1000,
            eval_strategy="steps",
            save_total_limit=2,
            report_to="wandb",
            gradient_checkpointing=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            adam_beta1=0.9,
            adam_beta2=0.999,
            max_grad_norm=0.3,  # Tighter gradient clipping
            dataloader_num_workers=0,
            group_by_length=True,  # Group similar lengths for efficiency
            length_column_name="length"
        )
        
        # Add length column for grouping
        def add_length(examples):
            examples["length"] = [len(ids) for ids in examples["input_ids"]]
            return examples
        
        tokenized_dataset = tokenized_dataset.map(add_length, batched=True)
        
        # Split for validation
        split_dataset = tokenized_dataset.train_test_split(test_size=0.1, seed=42)
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=split_dataset["train"],
            eval_dataset=split_dataset["test"],
            tokenizer=tokenizer,
            data_collator=data_collator,
            callbacks=[WandBCallback()]
        )
        
        wandb.log({
            'training/status': 'started',
            'training/total_parameters': sum(p.numel() for p in model.parameters()),
            'training/trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        })
        
        print("🎯 Training aggressively...")
        trainer.train()
        
        wandb.log({'training/status': 'completed'})
        
        print("💾 Saving...")
        trainer.save_model()
        
        print("✅ Aggressive training completed!")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        raise
    finally:
        wandb.finish()
        print("🧹 Done")

if __name__ == "__main__":
    main()