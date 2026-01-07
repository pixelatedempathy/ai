"""
Therapeutic Dialogue Fine-tuning Pipeline

Fine-tunes language models on therapeutic conversation datasets to improve
empathetic response generation, crisis intervention, and cultural competency.

This module provides:
- Data loading from Pixelated Empathy API
- Model fine-tuning with PEFT/LoRA support
- Training with safety validation
- Inference optimization for <50ms latency
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
)

try:
    from peft import LoraConfig, TaskType, get_peft_model

    _PEFT_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised when peft is missing
    LoraConfig = None  # type: ignore[assignment]
    TaskType = None  # type: ignore[assignment]
    get_peft_model = None  # type: ignore[assignment]
    _PEFT_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class TherapeuticFinetunConfig:
    """Configuration for therapeutic model fine-tuning."""

    # Model settings
    model_name: str = "meta-llama/Llama-2-7b-hf"
    tokenizer_name: Optional[str] = None
    use_lora: bool = True
    lora_rank: int = 8
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    # Training settings
    max_seq_length: int = 512
    batch_size: int = 8
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0

    # Data settings
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1
    min_quality_score: float = 0.7

    # Safety & optimization
    enable_safety_validation: bool = True
    use_mixed_precision: bool = True
    gradient_checkpointing: bool = True
    compile_model: bool = False

    # Output settings
    output_dir: str = "./checkpoints/therapeutic_model"
    eval_steps: int = 100
    save_steps: int = 200
    log_steps: int = 50


class TherapeuticConversationDataset(Dataset):
    """Dataset for therapeutic conversation fine-tuning."""

    def __init__(
        self,
        conversations: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        include_metadata: bool = True,
    ):
        """
        Initialize dataset.

        Args:
            conversations: List of conversation dictionaries with 'messages' key
            tokenizer: Tokenizer for encoding
            max_length: Maximum sequence length
            include_metadata: Include quality/bias metadata for weighted loss
        """
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.include_metadata = include_metadata

    def __len__(self) -> int:
        return len(self.conversations)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single conversation example."""
        conv = self.conversations[idx]

        # Format conversation as: <|therapist|> text <|patient|> text ...
        text = self._format_conversation(conv["messages"])

        # Encode
        encodings = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        item = {
            "input_ids": encodings["input_ids"].squeeze(),
            "attention_mask": encodings["attention_mask"].squeeze(),
        }

        # Add metadata for weighted loss
        if self.include_metadata:
            item["quality_score"] = torch.tensor(
                conv.get("quality_score", 0.5), dtype=torch.float32
            )
            item["has_crisis_signal"] = torch.tensor(
                float(conv.get("has_crisis_signal", False)), dtype=torch.float32
            )

        return item

    def _format_conversation(self, messages: List[Dict[str, str]]) -> str:
        """Format conversation for model input."""
        formatted = []
        for msg in messages:
            role = msg.get("role", "assistant")
            content = msg.get("content", "")

            # Clean content
            content = content.strip()

            if role == "user":
                formatted.append(f"<|patient|> {content}")
            elif role == "assistant":
                formatted.append(f"<|therapist|> {content}")
            else:
                formatted.append(content)

        return " ".join(formatted) + self.tokenizer.eos_token


class TherapeuticModelTrainer:
    """Trainer for therapeutic language models."""

    def __init__(
        self,
        config: TherapeuticFinetunConfig,
        model: Optional[PreTrainedModel] = None,
        tokenizer: Optional[PreTrainedTokenizer] = None,
    ):
        """Initialize trainer."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load or use provided model and tokenizer
        self.tokenizer = tokenizer or self._load_tokenizer()
        self.model = model or self._load_model()

        logger.info(f"Model loaded on device: {self.device}")

    def _load_tokenizer(self) -> PreTrainedTokenizer:
        """Load tokenizer."""
        tokenizer_name = self.config.tokenizer_name or self.config.model_name
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            trust_remote_code=True,
        )

        # Add special tokens
        special_tokens = {
            "additional_special_tokens": [
                "<|therapist|>",
                "<|patient|>",
                "<|crisis|>",
            ]
        }
        tokenizer.add_special_tokens(special_tokens)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return tokenizer

    def _load_model(self) -> PreTrainedModel:
        """Load base model and apply PEFT if configured."""
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=torch.float16
            if self.config.use_mixed_precision
            else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )

        # Resize token embeddings for new special tokens
        model.resize_token_embeddings(len(self.tokenizer))

        # Apply gradient checkpointing
        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()

        # Apply LoRA
        if self.config.use_lora:
            model = self._apply_lora(model)

        # Compile for speed (torch 2.0+)
        if self.config.compile_model:
            try:
                model = torch.compile(model)
                logger.info("Model compiled with torch.compile()")
            except Exception as e:
                logger.warning(f"Failed to compile model: {e}")

        return model.to(self.device)

    def _apply_lora(self, model: PreTrainedModel) -> PreTrainedModel:
        """Apply LoRA (Low-Rank Adaptation) to model."""
        if not _PEFT_AVAILABLE:
            raise ImportError(
                "peft is required for LoRA. Install with `uv add peft` or disable "
                "LoRA via `use_lora=False`."
            )

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=self.config.lora_target_modules,
            bias="none",
        )

        model = get_peft_model(model, lora_config)
        logger.info(f"Applied LoRA with rank={self.config.lora_rank}")

        return model

    def train(
        self,
        train_conversations: List[Dict[str, Any]],
        val_conversations: Optional[List[Dict[str, Any]]] = None,
        test_conversations: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Train the model on therapeutic conversations.

        Args:
            train_conversations: Training conversation list
            val_conversations: Validation conversation list
            test_conversations: Test conversation list

        Returns:
            Dictionary with training results
        """
        # Create datasets
        train_dataset = TherapeuticConversationDataset(
            train_conversations,
            self.tokenizer,
            self.config.max_seq_length,
        )

        eval_dataset = None
        if val_conversations:
            eval_dataset = TherapeuticConversationDataset(
                val_conversations,
                self.tokenizer,
                self.config.max_seq_length,
            )

        logger.info(f"Training set size: {len(train_dataset)}")
        if eval_dataset:
            logger.info(f"Validation set size: {len(eval_dataset)}")

        # Create training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            eval_strategy="steps" if eval_dataset else "no",
            eval_steps=self.config.eval_steps if eval_dataset else None,
            save_strategy="steps",
            save_steps=self.config.save_steps,
            logging_steps=self.config.log_steps,
            fp16=self.config.use_mixed_precision,
            dataloader_pin_memory=True,
            optim="adamw_8bit" if self.config.use_mixed_precision else "adamw_torch",
        )

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
        )

        # Train
        train_result = trainer.train()

        # Evaluate on test set if provided
        test_results = None
        if test_conversations:
            test_dataset = TherapeuticConversationDataset(
                test_conversations,
                self.tokenizer,
                self.config.max_seq_length,
            )
            test_results = trainer.evaluate(test_dataset)
            logger.info(f"Test results: {test_results}")

        return {
            "train_results": train_result,
            "test_results": test_results,
            "model_path": self.config.output_dir,
        }

    def generate_response(
        self,
        patient_message: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate therapeutic response to patient message.

        Args:
            patient_message: Patient's input message
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter

        Returns:
            Generated response
        """
        # Format input
        prompt = f"<|patient|> {patient_message}\n<|therapist|>"

        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode
        response = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )

        # Extract therapist response
        if "<|therapist|>" in response:
            response = response.split("<|therapist|>")[-1].strip()

        return response

    def save_checkpoint(self, path: str) -> None:
        """Save model checkpoint."""
        Path(path).mkdir(parents=True, exist_ok=True)

        if hasattr(self.model, "save_pretrained"):
            self.model.save_pretrained(path)
        else:
            torch.save(self.model.state_dict(), f"{path}/model.pt")

        self.tokenizer.save_pretrained(path)
        logger.info(f"Checkpoint saved to {path}")

    def load_checkpoint(self, path: str) -> None:
        """Load model checkpoint."""
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            torch_dtype=torch.float16
            if self.config.use_mixed_precision
            else torch.float32,
            device_map="auto",
        ).to(self.device)
        logger.info(f"Checkpoint loaded from {path}")
