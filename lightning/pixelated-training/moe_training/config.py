"""Training configuration for MoE system on Lightning.ai H100."""

from dataclasses import dataclass
from typing import Dict, List, Optional
import torch


@dataclass
class ModelConfig:
    """Base model configuration."""
    base_model_name: str = "microsoft/DialoGPT-medium"  # 355M params, good for H100
    max_length: int = 512
    vocab_size: int = 50257
    pad_token_id: int = 50256
    eos_token_id: int = 50256
    
    # LoRA configuration
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: List[str] = None
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = [
                "c_attn",  # Attention projection
                "c_proj",  # Output projection
                "c_fc",    # Feed-forward
            ]


@dataclass
class TrainingConfig:
    """Training hyperparameters optimized for H100."""
    # H100 optimization
    batch_size: int = 8  # Per device, H100 can handle larger
    gradient_accumulation_steps: int = 4  # Effective batch size = 32
    max_grad_norm: float = 1.0
    
    # Learning rates
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    
    # Training schedule
    num_epochs: int = 3
    save_steps: int = 500
    eval_steps: int = 250
    logging_steps: int = 50
    
    # Optimization
    optimizer: str = "adamw_torch"
    lr_scheduler_type: str = "cosine"
    fp16: bool = False  # H100 supports bf16 better
    bf16: bool = True
    dataloader_num_workers: int = 8
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.01
    
    # Memory optimization
    gradient_checkpointing: bool = True
    remove_unused_columns: bool = False


@dataclass
class ExpertConfig:
    """Configuration for individual expert training."""
    style: str
    data_file: str
    output_dir: str
    
    # Expert-specific hyperparameters
    learning_rate_multiplier: float = 1.0
    weight_decay_multiplier: float = 1.0
    
    # Data configuration
    train_split: float = 0.85
    val_split: float = 0.15
    min_length: int = 50
    max_length: int = 512


@dataclass
class SystemConfig:
    """System and environment configuration."""
    # Lightning.ai specific
    accelerator: str = "gpu"
    devices: int = 1  # Single H100
    precision: str = "bf16-mixed"
    
    # Paths
    base_output_dir: str = "/teamspace/studios/this_studio/models"
    data_dir: str = "/teamspace/studios/this_studio/moe_training_data"
    cache_dir: str = "/teamspace/studios/this_studio/.cache"
    
    # Logging
    wandb_project: str = "pixelated-empathy-moe"
    wandb_entity: str = "pixelated-empathy"
    log_level: str = "INFO"
    
    # Checkpointing
    save_total_limit: int = 3
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False


class MoETrainingConfig:
    """Complete MoE training configuration."""
    
    def __init__(self):
        self.model = ModelConfig()
        self.training = TrainingConfig()
        self.system = SystemConfig()
        
        # Expert configurations
        self.experts = {
            "therapeutic": ExpertConfig(
                style="therapeutic",
                data_file="therapeutic_high_quality.json",
                output_dir=f"{self.system.base_output_dir}/therapeutic_expert",
                learning_rate_multiplier=1.0,  # Standard rate for therapeutic
            ),
            "educational": ExpertConfig(
                style="educational", 
                data_file="educational_high_quality.json",
                output_dir=f"{self.system.base_output_dir}/educational_expert",
                learning_rate_multiplier=0.8,  # Slightly lower for educational
            ),
            "empathetic": ExpertConfig(
                style="empathetic",
                data_file="empathetic_high_quality.json", 
                output_dir=f"{self.system.base_output_dir}/empathetic_expert",
                learning_rate_multiplier=1.2,  # Higher for empathetic (less data)
            ),
            "practical": ExpertConfig(
                style="practical",
                data_file="practical_high_quality.json",
                output_dir=f"{self.system.base_output_dir}/practical_expert", 
                learning_rate_multiplier=0.9,  # Slightly lower for practical
            )
        }
    
    def get_expert_config(self, expert_name: str) -> ExpertConfig:
        """Get configuration for specific expert."""
        if expert_name not in self.experts:
            raise ValueError(f"Unknown expert: {expert_name}")
        return self.experts[expert_name]
    
    def get_training_config_for_expert(self, expert_name: str) -> TrainingConfig:
        """Get training config adjusted for specific expert."""
        expert_config = self.get_expert_config(expert_name)
        training_config = TrainingConfig()
        
        # Adjust learning rate based on expert
        training_config.learning_rate *= expert_config.learning_rate_multiplier
        training_config.weight_decay *= expert_config.weight_decay_multiplier
        
        return training_config
    
    def validate_config(self) -> bool:
        """Validate configuration for H100 training."""
        # Check memory requirements
        estimated_memory = self._estimate_memory_usage()
        if estimated_memory > 75:  # Leave 5GB buffer on H100
            print(f"Warning: Estimated memory usage {estimated_memory}GB exceeds H100 capacity")
            return False
        
        # Check data files exist
        import os
        for expert_name, expert_config in self.experts.items():
            data_path = os.path.join(self.system.data_dir, expert_config.data_file)
            if not os.path.exists(data_path):
                print(f"Error: Data file not found: {data_path}")
                return False
        
        return True
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage for H100."""
        # Base model memory (DialoGPT-medium in bf16)
        base_model_memory = 0.7  # ~700MB
        
        # LoRA adapters (much smaller)
        lora_memory = 0.1  # ~100MB per expert
        
        # Training overhead (gradients, optimizer states, etc.)
        training_overhead = base_model_memory * 3  # 3x for AdamW
        
        # Batch memory
        batch_memory = (
            self.training.batch_size * 
            self.training.gradient_accumulation_steps * 
            self.model.max_length * 
            4 * 2  # bf16 = 2 bytes, input + labels
        ) / (1024**3)  # Convert to GB
        
        total_memory = base_model_memory + lora_memory + training_overhead + batch_memory
        return total_memory
    
    def print_config_summary(self):
        """Print configuration summary."""
        print("=== MoE Training Configuration ===")
        print(f"Base Model: {self.model.base_model_name}")
        print(f"Max Length: {self.model.max_length}")
        print(f"LoRA r={self.model.lora_r}, alpha={self.model.lora_alpha}")
        print(f"Batch Size: {self.training.batch_size} x {self.training.gradient_accumulation_steps} = {self.training.batch_size * self.training.gradient_accumulation_steps}")
        print(f"Learning Rate: {self.training.learning_rate}")
        print(f"Epochs: {self.training.num_epochs}")
        print(f"Precision: {self.system.precision}")
        print(f"Estimated Memory: {self._estimate_memory_usage():.1f}GB")
        print("\nExperts:")
        for name, expert in self.experts.items():
            print(f"  {name}: {expert.data_file} (LR x{expert.learning_rate_multiplier})")
        print("=" * 40)


# Global configuration instance
config = MoETrainingConfig()
