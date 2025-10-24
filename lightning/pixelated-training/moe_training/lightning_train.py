"""Main training script for Lightning.ai H100 environment."""

import os
import sys
import logging
import torch
import wandb
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from moe_training.config import MoETrainingConfig
from moe_training.expert_trainer import MoETrainingOrchestrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/teamspace/studios/this_studio/training.log')
    ]
)

logger = logging.getLogger(__name__)


def check_environment():
    """Check Lightning.ai environment and H100 availability."""
    logger.info("Checking environment...")
    
    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    
    # Check GPU type
    gpu_name = torch.cuda.get_device_name(0)
    logger.info(f"GPU: {gpu_name}")
    
    if "H100" not in gpu_name:
        logger.warning(f"Expected H100, found {gpu_name}")
    
    # Check memory
    memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    logger.info(f"GPU Memory: {memory_gb:.1f}GB")
    
    if memory_gb < 70:
        logger.warning(f"Expected ~80GB for H100, found {memory_gb:.1f}GB")
    
    # Check disk space
    disk_usage = os.statvfs('/teamspace/studios/this_studio')
    free_gb = (disk_usage.f_bavail * disk_usage.f_frsize) / (1024**3)
    logger.info(f"Free disk space: {free_gb:.1f}GB")
    
    if free_gb < 50:
        logger.warning(f"Low disk space: {free_gb:.1f}GB")
    
    logger.info("Environment check complete")


def setup_directories(config: MoETrainingConfig):
    """Set up required directories."""
    directories = [
        config.system.base_output_dir,
        config.system.cache_dir,
        "/teamspace/studios/this_studio/logs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")


def validate_data_files(config: MoETrainingConfig):
    """Validate that all required data files exist."""
    logger.info("Validating data files...")
    
    missing_files = []
    
    for expert_name, expert_config in config.experts.items():
        data_file = Path(config.system.data_dir) / expert_config.data_file
        
        if not data_file.exists():
            missing_files.append(str(data_file))
        else:
            # Check file size
            size_mb = data_file.stat().st_size / (1024**2)
            logger.info(f"{expert_name}: {data_file.name} ({size_mb:.1f}MB)")
    
    if missing_files:
        raise FileNotFoundError(f"Missing data files: {missing_files}")
    
    logger.info("All data files validated")


def main():
    """Main training function."""
    logger.info("Starting MoE training on Lightning.ai H100")
    
    try:
        # Check environment
        check_environment()
        
        # Load configuration
        config = MoETrainingConfig()
        
        # Validate configuration
        if not config.validate_config():
            raise ValueError("Configuration validation failed")
        
        # Print configuration summary
        config.print_config_summary()
        
        # Set up directories
        setup_directories(config)
        
        # Validate data files
        validate_data_files(config)
        
        # Initialize training orchestrator
        orchestrator = MoETrainingOrchestrator(config)
        
        # Set up all expert trainers
        orchestrator.setup_all_experts()
        
        # Train all experts
        logger.info("Starting expert training...")
        training_results = orchestrator.train_all_experts(sequential=True)
        
        # Evaluate all experts
        logger.info("Evaluating trained experts...")
        evaluation_results = orchestrator.evaluate_all_experts()
        
        # Save all experts
        logger.info("Saving trained experts...")
        orchestrator.save_all_experts(config.system.base_output_dir)
        
        # Generate sample outputs
        logger.info("Generating sample outputs...")
        sample_prompts = {
            "therapeutic": "I'm struggling with trauma from my childhood and don't know how to heal.",
            "educational": "What is complex PTSD and how does it develop?",
            "empathetic": "I feel so alone and nobody understands what I'm going through.",
            "practical": "What are some concrete steps I can take to start my recovery?"
        }
        
        samples = orchestrator.generate_samples(sample_prompts)
        
        # Print results summary
        print("\n" + "="*60)
        print("TRAINING COMPLETE - RESULTS SUMMARY")
        print("="*60)
        
        for expert_name in config.experts.keys():
            train_metrics = training_results.get(expert_name, {})
            eval_metrics = evaluation_results.get(expert_name, {})
            
            print(f"\n{expert_name.upper()} EXPERT:")
            print(f"  Training Loss: {train_metrics.get('train_loss', 'N/A'):.4f}")
            print(f"  Eval Loss: {eval_metrics.get('eval_loss', 'N/A'):.4f}")
            print(f"  Perplexity: {eval_metrics.get('perplexity', 'N/A'):.2f}")
            print(f"  Accuracy: {eval_metrics.get('accuracy', 'N/A'):.4f}")
            
            if expert_name in samples:
                print(f"  Sample: {samples[expert_name][:100]}...")
        
        print(f"\nModels saved to: {config.system.base_output_dir}")
        print("="*60)
        
        logger.info("MoE training completed successfully!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    finally:
        # Clean up
        if wandb.run:
            wandb.finish()
        
        # Clear GPU memory
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
