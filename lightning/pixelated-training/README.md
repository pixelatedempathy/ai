# Wayfarer2-Pixelated Training Package

**Complete training pipeline for fine-tuning Wayfarer-2-12B on 40,525 mental health + CoT conversations**

## 📋 Quick Start Checklist

### ✅ Pre-Training Setup
```bash
# 1. Extract package
tar -xzf pixelated-training.tar.gz
cd pixelated-training/

# 2. One-command setup (recommended)
./setup.sh

# OR manual setup:
# uv add -r requirements.txt
# uv run python setup_wandb.py
# uv run python comprehensive_test.py
```

### ✅ Training Execution
```bash
# Start training with enhanced monitoring
uv run python train.py

# Monitor system resources (optional, separate terminal)
uv run python system_monitor.py

# Monitor progress at: https://wandb.ai/your-username/wayfarer-2-12b-finetuning
```

### ✅ Post-Training (Automatic)
After training completes, the system automatically:
- ✅ Exports to GGUF formats (Q8_0, Q6_K, Q4_K_M, Q2_K)
- ✅ Uploads to HuggingFace as "Wayfarer2-Pixelated"
- ✅ Includes comprehensive model card

## 📊 What You Get

**Dataset**: 40,525 real conversations (275MB)
- Mental Health: 11,699 counseling conversations
- CoT Reasoning: 28,826 reasoning examples
- Quality: 99.97% clean (only 12 issues total)
- Format: ChatML optimized for Wayfarer-2-12B

**Training**: Optimized for Lightning AI A100 80GB
- Time: ~7 hours (40 Lightning credits)
- Memory: ~78GB GPU memory
- vCPUs: 12 (170GB RAM) - faster data loading
- Checkpoints: Every 1013 steps
- Safety: Real-time monitoring

**Output**: Multiple model formats
- PyTorch model (~22.4GB)
- GGUF Q8_0 (~12.9GB) - Recommended
- GGUF Q6_K (~10.3GB) - Good balance
- GGUF Q4_K_M (~7.2GB) - Smaller size
- GGUF Q2_K (~4.8GB) - Minimal size

## 🛠️ Hardware Requirements

**Training**:
- Lightning AI A100 80GB (recommended)
- 12 vCPUs, 170GB RAM
- Cost: ~62.5 Lightning credits (~$62.50)
- Wait time: ~5 minutes

**Alternative Cloud Options**:
- AWS p4d.xlarge (A100 80GB)
- GCP A2-highgpu-1g (A100 40GB + sharding)
- Azure NC24ads_A100_v4

## 🚀 Training Commands

### Standard Training
```bash
uv run python train.py
```

### Resume from Checkpoint
```bash
uv run python train.py --resume_from_checkpoint ./checkpoints/checkpoint-1000
```

### Hyperparameter Sweep
```bash
wandb sweep sweep_config.json
wandb agent <sweep-id>
```

### Manual Post-Training
```bash
# If automatic post-training fails
uv run python post_training_pipeline.py ./wayfarer-finetuned
```

## 📈 Monitoring & Progress

### WandB Dashboard
- Real-time loss curves
- Learning rate schedules
- GPU memory usage
- Safety metrics
- Dataset statistics

### Training Logs
```bash
# View training progress
tail -f logs/training.log

# Check GPU usage
nvidia-smi -l 1
```

### Checkpoints
- Saved every 1013 steps
- Located in `./checkpoints/`
- Automatic resume on restart

## 🔍 Testing & Validation

### Pre-Training Tests
```bash
# Run all validation tests
uv run python comprehensive_test.py

# Quick dry run
uv run python dry_run_test.py

# Test training components
uv run python mock_training_test.py
```

### Post-Training Evaluation
```bash
# Evaluate trained model
uv run python evaluate_model.py ./wayfarer-finetuned

# Compare with baseline
uv run python compare_models.py
```

## 🚀 Deployment Options

### Interactive Inference
```bash
uv run python inference.py --chat
```

### API Server
```bash
uv run python api_server.py
# Access at: http://localhost:8000/docs
```

### Docker Deployment
```bash
docker-compose up --build
```

## 🛡️ Enhanced Features

- **Gradient Clipping**: Prevents exploding gradients (max_grad_norm: 1.0)
- **System Monitoring**: Real-time GPU/memory/disk tracking
- **Progress Bars**: Visual feedback for all long operations
- **Graceful Shutdown**: Handle SIGTERM/SIGINT without corruption
- **One-Command Setup**: `./setup.sh` for complete automation
- **Enhanced Logging**: Loss trends, alerts, and health monitoring

## 📁 Package Contents

**Core Training**:
- `train.py` - Main training script with WandB integration
- `training_config.json` - Optimized hyperparameters
- `training_dataset.json` - 40,525 conversations (275MB)
- `safety_config.json` - Calibrated safety settings

**Post-Training**:
- `export_gguf.py` - GGUF export with quantizations
- `upload_hf.py` - HuggingFace upload automation
- `post_training_pipeline.py` - Complete automation
- `model_card.md` - HuggingFace model documentation

**Evaluation**:
- `evaluate_model.py` - Model quality assessment
- `compare_models.py` - Baseline comparison
- `evaluation_sets.json` - Test datasets

**Deployment**:
- `inference.py` - Optimized inference script
- `api_server.py` - FastAPI production server
- `Dockerfile` + `docker-compose.yml` - Container deployment

**Testing**:
- `comprehensive_test.py` - Complete validation suite
- `dry_run_test.py` - Quick compatibility check
- `mock_training_test.py` - Training pipeline test

## ⚠️ Troubleshooting

### Common Issues

**CUDA Out of Memory**:
```bash
# Reduce batch size in training_config.json
"per_device_train_batch_size": 1,
"gradient_accumulation_steps": 4
```

**WandB Login Issues**:
```bash
wandb login
# Or set: export WANDB_API_KEY=your_key
```

**HuggingFace Upload Fails**:
```bash
huggingface-cli login
# Or set: export HF_TOKEN=your_token
```

**Training Stalls**:
```bash
# Check GPU utilization
nvidia-smi

# Monitor training logs
tail -f logs/training.log
```

### Getting Help

1. Check `logs/training.log` for detailed error messages
2. Run `uv run python comprehensive_test.py` to validate setup
3. Verify hardware requirements are met
4. Ensure all dependencies are installed

## 💰 Lightning AI Cost Estimate

**Training Cost**: ~40 Lightning credits (~$40)
- Training: 7 hours × 5.68 credits/hour = 40 credits
- GGUF Export: ~20 minutes (included)
- HuggingFace Upload: ~10 minutes (included)

**Cost Optimization**:
- Use interruptible instances: 7 hours × 2.94 credits/hour = ~20.6 credits (~$20.60)
- Risk: May be interrupted, but training resumes from checkpoints
## 🎯 Expected Timeline

**Setup**: 10-15 minutes
**Training**: 7 hours on Lightning AI A100 80GB
**GGUF Export**: 30-45 minutes
**HuggingFace Upload**: 15-30 minutes
**Total**: ~8 hours end-to-end

## 📞 Support

- **Training Issues**: Check logs and run validation tests
- **Hardware Problems**: Verify GPU memory and CUDA installation
- **Upload Issues**: Confirm WandB and HuggingFace authentication

---

**Ready to train? Run `uv run python train.py` and monitor at your WandB dashboard!**
