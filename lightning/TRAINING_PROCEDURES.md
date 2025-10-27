# Therapeutic AI Training Procedures

## 📋 Table of Contents

1. [Overview](#overview)
2. [Pre-Training Setup](#pre-training-setup)
3. [Training Execution](#training-execution)
4. [Checkpoint Management](#checkpoint-management)
5. [Monitoring and Validation](#monitoring-and-validation)
6. [Post-Training Procedures](#post-training-procedures)
7. [Troubleshooting](#troubleshooting)
8. [Best Practices](#best-practices)

---

## Overview

This document provides comprehensive procedures for training the therapeutic AI model using the MoE (Mixture of Experts) architecture with LoRA fine-tuning on Lightning.ai H100 infrastructure.

### Training System Components

- **Base Model**: LatitudeGames/Wayfarer-2-12B
- **Architecture**: 4-expert MoE with domain specialization
- **Fine-tuning**: LoRA (Low-Rank Adaptation)
- **Hardware**: NVIDIA H100 GPU (80GB)
- **Time Constraint**: 12-hour maximum training window
- **Context Length**: 8192 tokens (training at 2048)

### Training Scripts

- `train_optimized.py` - Automatic optimization and training (recommended)
- `train_moe_h100.py` - Direct MoE training with manual configuration
- `training_optimizer.py` - Optimization utilities and profile selection

---

## Pre-Training Setup

### 1. Environment Preparation

#### Install Dependencies

```bash
# Navigate to Lightning.ai directory
cd ai/lightning/

# Install with uv (recommended)
uv pip install -r requirements_moe.txt

# Or with pip
pip install -r requirements_moe.txt
```

#### Verify Installation

```bash
# Check Python version (3.11+ required)
python --version

# Verify PyTorch and CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

# Verify transformers
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
```

### 2. Dataset Preparation

#### Dataset Format

Training data must be in JSON format:

```json
{
  "conversations": [
    {
      "text": "Therapist: How are you feeling today?\nClient: I've been struggling with anxiety lately.\nTherapist: I understand. Can you tell me more about when you feel most anxious?"
    },
    {
      "text": "Therapist: What brings you here today?\nClient: I'm having trouble sleeping..."
    }
  ]
}
```

#### Dataset Validation

```bash
# Validate dataset format
python -c "
import json
with open('training_dataset.json', 'r') as f:
    data = json.load(f)
    print(f'Total conversations: {len(data[\"conversations\"])}')
    print(f'Sample text length: {len(data[\"conversations\"][0][\"text\"])} chars')
"
```

#### Dataset Statistics

```bash
# Analyze dataset
python dataset_preprocessing.py --analyze training_dataset.json
```

Expected output:
```
📊 Dataset Analysis:
   Total conversations: 8,000
   Avg tokens per conversation: 500
   Min tokens: 50
   Max tokens: 2048
   Total tokens: 4,000,000
```

### 3. Configuration Files

#### training_config.json

```json
{
  "base_model": "LatitudeGames/Wayfarer-2-12B",
  "num_train_epochs": 3,
  "per_device_train_batch_size": 4,
  "gradient_accumulation_steps": 8,
  "learning_rate": 3e-4,
  "warmup_steps": 1000,
  "optimization_priority": "balanced",
  "max_training_hours": 12.0
}
```

**Configuration Parameters**:
- `num_train_epochs`: Number of complete passes through dataset (2-4 recommended)
- `per_device_train_batch_size`: Samples per GPU per step (2-8 for H100)
- `gradient_accumulation_steps`: Steps before optimizer update (4-16)
- `learning_rate`: Initial learning rate (3e-4 recommended for LoRA)
- `warmup_steps`: Linear warmup steps (1000 recommended)
- `optimization_priority`: Profile selection ('fast', 'balanced', 'quality', 'memory_efficient')

#### wandb_config.json

```json
{
  "project": "therapeutic-ai-training",
  "entity": "your-wandb-username",
  "name": "moe-training-run-1",
  "tags": ["moe", "h100", "lora", "therapeutic"],
  "notes": "Training run with balanced profile",
  "config": {
    "architecture": "moe",
    "num_experts": 4,
    "lora_rank": 16
  }
}
```

#### safety_config.json

```json
{
  "max_training_hours": 12.0,
  "checkpoint_interval_minutes": 30,
  "early_stopping_patience": 3,
  "max_grad_norm": 1.0,
  "enable_time_constraints": true,
  "enable_early_stopping": true
}
```

### 4. Pre-Training Checklist

- [ ] Python 3.11+ installed
- [ ] All dependencies installed (`requirements_moe.txt`)
- [ ] CUDA available and working
- [ ] Training dataset prepared and validated
- [ ] Configuration files created and reviewed
- [ ] WandB account configured and logged in
- [ ] Sufficient disk space (>100GB recommended)
- [ ] H100 GPU allocated on Lightning.ai

---

## Training Execution

### Method 1: Automatic Optimization (Recommended)

This method automatically analyzes your dataset and selects optimal training parameters.

```bash
# Start optimized training
python train_optimized.py
```

**What it does**:
1. Analyzes dataset (size, token distribution)
2. Selects optimal profile (fast/balanced/quality/memory_efficient)
3. Estimates training time
4. Adjusts parameters if needed to fit 12-hour window
5. Trains model with optimal configuration
6. Saves optimization report

**Expected Output**:
```
🚀 Optimized Therapeutic AI Training
============================================================

📋 Loading configurations...
📊 Analyzing dataset...
   Samples: 8,000
   Avg tokens: 500
   Min tokens: 50
   Max tokens: 2048

🎯 Optimizing training parameters for 12-hour window...

🎯 Selected Profile: balanced
   Batch Size: 4
   Gradient Accumulation: 8
   Effective Batch Size: 32
   Max Length: 2048
   Estimated Throughput: 800 tokens/sec
   Memory Usage: 60.0 GB

⏰ Time Estimate:
   Total Steps: 750
   Estimated Duration: 4.17 hours
   Completion: 2024-10-27 14:10:00
   Fits in 12h window: ✅ Yes

📊 Optimization report saved to: training_optimization_report.json

🚀 Creating MoE model from LatitudeGames/Wayfarer-2-12B...
✅ MoE model created
   Total params: 12,345,678,901
   Trainable: 123,456,789 (1.00%)

🔤 Tokenizing with max_length=2048...
   Train: 7,200 | Eval: 800

🎯 Starting optimized training...
⏰ Maximum duration: 12.0 hours
📊 Effective batch size: 32
🎓 Epochs: 3
📏 Max length: 2048
============================================================
```

### Method 2: Manual Configuration

For advanced users who want full control over training parameters.

```bash
# Start MoE training with manual configuration
python train_moe_h100.py
```

**Configuration**: Edit `moe_training_config.json` before running.

### Training Progress

During training, you'll see real-time progress:

```
📊 Progress: 15.3% | Loss: 2.145 | Step: 115
📊 Progress: 30.7% | Loss: 1.823 | Step: 230
💾 Checkpoint at 0.5 hours
📊 Progress: 45.2% | Loss: 1.567 | Step: 339
📊 Progress: 60.8% | Loss: 1.412 | Step: 456
💾 Checkpoint at 1.0 hours
📊 Progress: 75.1% | Loss: 1.298 | Step: 563
📊 Progress: 90.4% | Loss: 1.234 | Step: 678
📊 Progress: 100.0% | Loss: 1.198 | Step: 750

💾 Saving model...
✅ Training completed in 4.23 hours!
📁 Model saved to: ./therapeutic_moe_model
```

### Training Phases

#### Phase 1: Initialization (5-10 minutes)
- Load base model
- Apply LoRA adapters
- Create MoE layers
- Initialize optimizer
- Setup WandB logging

#### Phase 2: Warmup (First 1000 steps)
- Learning rate increases linearly from 0 to target
- Model adapts to therapeutic domain
- Expert routing stabilizes

#### Phase 3: Main Training
- Cosine learning rate schedule
- Expert specialization develops
- Loss decreases steadily
- Checkpoints saved every 30 minutes

#### Phase 4: Finalization
- Save final model
- Export LoRA adapters
- Save MoE layers
- Generate training report

---

## Checkpoint Management

### Automatic Checkpointing

Checkpoints are saved automatically:
- **Interval**: Every 30 minutes
- **Location**: `./therapeutic_moe_model/checkpoint-{step}/`
- **Contents**: Model weights, optimizer state, training state

### Checkpoint Structure

```
therapeutic_moe_model/
├── checkpoint-500/
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   ├── optimizer.pt
│   ├── scheduler.pt
│   ├── trainer_state.json
│   └── training_args.bin
├── checkpoint-1000/
├── checkpoint-1500/
└── moe_layers.pt
```

### Resume from Checkpoint

If training is interrupted, resume from the last checkpoint:

```bash
# Automatic resume (detects last checkpoint)
python train_optimized.py --resume_from_checkpoint auto

# Resume from specific checkpoint
python train_optimized.py --resume_from_checkpoint ./therapeutic_moe_model/checkpoint-1500
```

### Checkpoint Validation

Verify checkpoint integrity:

```bash
# Check checkpoint
python -c "
import torch
checkpoint = torch.load('./therapeutic_moe_model/checkpoint-1500/optimizer.pt')
print(f'Checkpoint step: {checkpoint[\"step\"]}')
print(f'Loss: {checkpoint[\"loss\"]:.4f}')
"
```

### Checkpoint Cleanup

Remove old checkpoints to save space:

```bash
# Keep only last 3 checkpoints (automatic)
# Configured in training_args: save_total_limit=5

# Manual cleanup
rm -rf ./therapeutic_moe_model/checkpoint-500
rm -rf ./therapeutic_moe_model/checkpoint-1000
```

### Best Model Selection

The best model (lowest validation loss) is automatically saved:

```
therapeutic_moe_model/
├── adapter_config.json      # Best model
├── adapter_model.bin         # Best model
├── moe_layers.pt            # Best model
└── checkpoint-{best_step}/  # Best checkpoint
```

---

## Monitoring and Validation

### Real-Time Monitoring

#### WandB Dashboard

Access your training dashboard:
```
https://wandb.ai/your-username/therapeutic-ai-training
```

**Key Metrics**:
- `train_loss`: Training loss (should decrease)
- `eval_loss`: Validation loss (should decrease)
- `learning_rate`: Current learning rate
- `training/elapsed_hours`: Time elapsed
- `training/remaining_hours`: Estimated remaining time
- `expert_usage`: Distribution across experts
- `routing_entropy`: Expert selection diversity

#### Terminal Output

Monitor in terminal:
```bash
# Follow training log
tail -f training.log

# Watch GPU usage
watch -n 1 nvidia-smi
```

### Validation Checks

#### During Training

**Every 500 steps**:
- Validation loss computed
- Model checkpoint saved
- Metrics logged to WandB

**Validation Metrics**:
- Loss: Should decrease over time
- Perplexity: Should decrease (target < 2.5)
- Expert balance: 20-30% per expert
- Routing entropy: > 1.0

#### Early Stopping

Training stops automatically if:
- Validation loss increases for 3 consecutive evaluations
- Time limit approaching (30 min before 12 hours)
- Manual interrupt (Ctrl+C)

### Quality Validation

After training completes, validate model quality:

```bash
# Run validation script
python validate_model.py --model_path ./therapeutic_moe_model

# Test on sample conversations
python test_inference.py --model_path ./therapeutic_moe_model --test_file test_conversations.json
```

**Quality Metrics**:
- Therapeutic appropriateness
- Response coherence
- Bias detection accuracy
- Expert routing correctness

---

## Post-Training Procedures

### 1. Model Export

Export the trained model for deployment:

```bash
# Model is already saved in ./therapeutic_moe_model/
# Contains:
# - LoRA adapters (adapter_model.bin)
# - MoE layers (moe_layers.pt)
# - Tokenizer files
# - Configuration files
```

### 2. Model Evaluation

Comprehensive evaluation:

```bash
# Evaluate on test set
python evaluate_model.py \
  --model_path ./therapeutic_moe_model \
  --test_data test_conversations.json \
  --output_dir ./evaluation_results/
```

**Evaluation Metrics**:
- Perplexity
- BLEU score
- Therapeutic appropriateness score
- Bias detection accuracy
- Expert routing analysis

### 3. Training Report

Generate comprehensive training report:

```bash
# Generate report
python generate_training_report.py \
  --model_path ./therapeutic_moe_model \
  --wandb_run_id your-run-id \
  --output training_report.pdf
```

**Report Contents**:
- Training configuration
- Dataset statistics
- Training curves (loss, learning rate)
- Expert usage distribution
- Time analysis (estimated vs actual)
- Model quality metrics
- Recommendations for next training

### 4. Model Versioning

Tag and version your trained model:

```bash
# Create version tag
git tag -a v1.0-moe-therapeutic "$(date +%Y%m%d-%H%M%S)"

# Save model metadata
cat > ./therapeutic_moe_model/model_info.json << EOF
{
  "version": "1.0",
  "training_date": "$(date -I)",
  "dataset_size": 8000,
  "training_hours": 4.23,
  "final_loss": 1.198,
  "base_model": "LatitudeGames/Wayfarer-2-12B",
  "architecture": "4-expert MoE with LoRA",
  "lora_rank": 16
}
EOF
```

### 5. Backup

Backup trained model:

```bash
# Create backup
tar -czf therapeutic_moe_model_$(date +%Y%m%d).tar.gz ./therapeutic_moe_model/

# Upload to cloud storage (example)
aws s3 cp therapeutic_moe_model_$(date +%Y%m%d).tar.gz s3://your-bucket/models/
```

---

## Troubleshooting

### Common Issues

#### Issue 1: Out of Memory (OOM)

**Symptoms**:
```
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
```

**Solutions**:
1. Reduce batch size:
   ```json
   {
     "per_device_train_batch_size": 2,
     "gradient_accumulation_steps": 16
   }
   ```

2. Use memory_efficient profile:
   ```json
   {
     "optimization_priority": "memory_efficient"
   }
   ```

3. Enable gradient checkpointing (already enabled by default)

4. Reduce max_length:
   ```json
   {
     "max_length": 1024
   }
   ```

#### Issue 2: Training Too Slow

**Symptoms**:
- Estimated time > 12 hours
- Low tokens/sec throughput

**Solutions**:
1. Use fast profile:
   ```json
   {
     "optimization_priority": "fast"
   }
   ```

2. Reduce epochs:
   ```json
   {
     "num_train_epochs": 2
   }
   ```

3. Increase batch size (if memory allows):
   ```json
   {
     "per_device_train_batch_size": 8,
     "gradient_accumulation_steps": 4
   }
   ```

#### Issue 3: Loss Not Decreasing

**Symptoms**:
- Loss plateaus or increases
- Validation loss higher than training loss

**Solutions**:
1. Check learning rate:
   ```json
   {
     "learning_rate": 1e-4  // Try lower
   }
   ```

2. Increase warmup steps:
   ```json
   {
     "warmup_steps": 2000
   }
   ```

3. Check dataset quality:
   ```bash
   python validate_dataset.py --dataset training_dataset.json
   ```

4. Reduce regularization:
   ```json
   {
     "weight_decay": 0.001  // Lower from 0.01
   }
   ```

#### Issue 4: Poor Expert Balance

**Symptoms**:
- One expert handles >50% of tokens
- Low routing entropy (<0.5)

**Solutions**:
1. Increase load balancing weight:
   ```python
   moe_config = MoEConfig(
       load_balancing_weight=0.05  # Increase from 0.01
   )
   ```

2. Check dataset diversity:
   - Ensure balanced representation of domains
   - Add more diverse therapeutic conversations

#### Issue 5: Training Timeout

**Symptoms**:
- Training stops at 11.5 hours
- Not all epochs completed

**Solutions**:
1. Reduce epochs:
   ```json
   {
     "num_train_epochs": 2
   }
   ```

2. Use faster profile:
   ```json
   {
     "optimization_priority": "fast"
   }
   ```

3. Reduce dataset size (if acceptable):
   ```bash
   python sample_dataset.py --input training_dataset.json --output sampled_dataset.json --size 6000
   ```

### Error Messages

#### "Model not found"

```bash
# Verify model name
huggingface-cli repo info LatitudeGames/Wayfarer-2-12B

# Login to HuggingFace if needed
huggingface-cli login
```

#### "WandB not initialized"

```bash
# Login to WandB
wandb login

# Or use offline mode
export WANDB_MODE=offline
```

#### "Checkpoint corrupted"

```bash
# Remove corrupted checkpoint
rm -rf ./therapeutic_moe_model/checkpoint-{step}

# Resume from previous checkpoint
python train_optimized.py --resume_from_checkpoint ./therapeutic_moe_model/checkpoint-{previous_step}
```

---

## Best Practices

### 1. Dataset Preparation

- **Quality over quantity**: 5,000 high-quality conversations > 20,000 low-quality
- **Diversity**: Include various therapeutic scenarios and client presentations
- **Balance**: Ensure even distribution across expert domains
- **Validation**: Always validate dataset format before training

### 2. Configuration

- **Start with defaults**: Use `train_optimized.py` with balanced profile
- **Iterate**: Adjust based on first training run results
- **Document changes**: Keep notes on configuration changes and results
- **Version control**: Track configuration files in git

### 3. Monitoring

- **Watch WandB**: Monitor training curves in real-time
- **Check checkpoints**: Validate checkpoints periodically
- **GPU utilization**: Ensure GPU is fully utilized (>80%)
- **Time tracking**: Monitor estimated vs actual time

### 4. Checkpointing

- **Frequent saves**: 30-minute intervals (default)
- **Keep multiple**: Maintain last 5 checkpoints
- **Backup best**: Always backup the best model
- **Test checkpoints**: Validate checkpoint quality before deleting old ones

### 5. Post-Training

- **Evaluate thoroughly**: Test on diverse scenarios
- **Document results**: Create comprehensive training report
- **Version models**: Tag and version all trained models
- **Backup**: Always backup trained models to cloud storage

### 6. Iteration

- **Analyze results**: Review training metrics and model quality
- **Identify issues**: Note any problems or areas for improvement
- **Adjust configuration**: Modify parameters based on analysis
- **Retrain**: Run new training with improved configuration

### 7. Safety

- **Time limits**: Always respect 12-hour window
- **Early stopping**: Enable to prevent overfitting
- **Validation**: Monitor validation loss closely
- **Bias detection**: Validate model for biases post-training

---

## Training Workflow Summary

```
1. Pre-Training Setup
   ├── Install dependencies
   ├── Prepare dataset
   ├── Configure training
   └── Verify environment

2. Training Execution
   ├── Start training (train_optimized.py)
   ├── Monitor progress (WandB + terminal)
   └── Wait for completion

3. Checkpoint Management
   ├── Automatic saves every 30 min
   ├── Resume if interrupted
   └── Keep best model

4. Monitoring & Validation
   ├── Watch training metrics
   ├── Validate checkpoints
   └── Check expert balance

5. Post-Training
   ├── Export model
   ├── Evaluate quality
   ├── Generate report
   ├── Version and backup
   └── Deploy (if quality acceptable)

6. Iteration (if needed)
   ├── Analyze results
   ├── Adjust configuration
   └── Retrain
```

---

## Quick Reference

### Essential Commands

```bash
# Start training (recommended)
python train_optimized.py

# Resume from checkpoint
python train_optimized.py --resume_from_checkpoint auto

# Validate dataset
python dataset_preprocessing.py --analyze training_dataset.json

# Check GPU
nvidia-smi

# Monitor training
tail -f training.log

# Evaluate model
python evaluate_model.py --model_path ./therapeutic_moe_model
```

### Configuration Files

- `training_config.json` - Training parameters
- `wandb_config.json` - Logging configuration
- `safety_config.json` - Safety constraints
- `moe_training_config.json` - MoE-specific settings

### Key Directories

- `./therapeutic_moe_model/` - Trained model output
- `./therapeutic_moe_model/checkpoint-*/` - Training checkpoints
- `./training_optimization_report.json` - Optimization report

### Important Metrics

- **Loss**: Should decrease (target < 1.5)
- **Perplexity**: Should decrease (target < 2.5)
- **Expert balance**: 20-30% per expert
- **Routing entropy**: > 1.0
- **Training time**: < 12 hours

---

## Additional Resources

- **MoE Architecture**: `moe_architecture.py`
- **Training Optimizer**: `training_optimizer.py`
- **MoE Training Guide**: `MOE_TRAINING_GUIDE.md`
- **Optimization Guide**: `TRAINING_OPTIMIZATION_GUIDE.md`
- **Inference Guide**: `INFERENCE_OPTIMIZATION_GUIDE.md`
- **Deployment Guide**: `LIGHTNING_H100_QUICK_DEPLOY.md`

---

**Document Version**: 1.0  
**Last Updated**: October 2025  
**Status**: Complete

For questions or issues, refer to the troubleshooting section or consult the additional resources.
