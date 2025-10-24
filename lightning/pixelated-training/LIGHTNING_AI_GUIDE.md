# Lightning AI Training Guide

## 🚀 Quick Start on Lightning AI

### 1. Launch Lightning AI Studio
```bash
# Create new studio with A100 80GB
# Machine: A100 (80GB VRAM, 12 vCPUs, 170GB RAM)
# Cost: 5.68 credits/hour (~$5.68/hour)
```

### 2. Upload Training Package
```bash
# Upload pixelated-training.tar.gz to Lightning AI
# Or clone from repository
```

### 3. Setup & Train
```bash
# Extract and setup
tar -xzf pixelated-training.tar.gz
cd pixelated-training/

# One-command setup
./setup.sh

# Start training
uv run python train.py
```

## 💰 Cost Optimization

### Standard Instance
- **Cost**: 5.68 credits/hour
- **Training**: ~7 hours = 40 credits (~$40)
- **Reliability**: Guaranteed completion

### Interruptible Instance (Recommended)
- **Cost**: 2.94 credits/hour (48% savings)
- **Training**: ~7 hours = 20.6 credits (~$20.60)
- **Risk**: May be interrupted, resumes from checkpoints
- **Wait Time**: ~2 minutes vs 5 minutes

## 📊 Training Monitoring

### Lightning AI Dashboard
- GPU utilization and memory usage
- Training logs and metrics
- Cost tracking in real-time

### WandB Integration
- Detailed training metrics
- Loss curves and learning rates
- Model performance tracking
- Access at: https://wandb.ai/your-username/wayfarer-2-12b-finetuning

## 🔧 Lightning AI Optimizations

### Automatic Features
- **CUDA 12.1** pre-installed
- **PyTorch 2.0+** optimized for A100
- **High-speed NVMe storage** for fast data loading
- **Optimized networking** for model downloads

### Manual Optimizations
```bash
# Enable persistent storage (optional)
# Saves checkpoints across sessions

# Monitor GPU usage
watch -n 1 nvidia-smi

# Monitor system resources
uv run python system_monitor.py &
```

## 🛡️ Checkpoint Management

### Automatic Checkpointing
- Saves every 1013 steps (~1 hour intervals)
- 10 total checkpoints during training
- Automatic resume on restart/interruption

### Manual Checkpoint Control
```bash
# Resume from specific checkpoint
uv run python train.py --resume_from_checkpoint ./checkpoints/checkpoint-5065

# List available checkpoints
ls -la checkpoints/
```

## 📤 Post-Training Pipeline

### Automatic Export & Upload
After training completes:
1. **GGUF Export**: 4 quantization levels (Q8_0, Q6_K, Q4_K_M, Q2_K)
2. **HuggingFace Upload**: Model + variants as "Wayfarer2-Pixelated"
3. **Model Card**: Comprehensive documentation

### Manual Pipeline
```bash
# If automatic pipeline fails
uv run python post_training_pipeline.py ./wayfarer-finetuned
```

## 🚨 Troubleshooting

### Common Lightning AI Issues

**Out of Memory**:
```bash
# Reduce batch size if needed
# Edit training_config.json:
"per_device_train_batch_size": 1,
"gradient_accumulation_steps": 8
```

**Session Timeout**:
```bash
# Training automatically resumes from last checkpoint
# No manual intervention needed
```

**Slow Data Loading**:
```bash
# Dataset is optimized for fast loading
# 275MB loads in ~30 seconds on Lightning AI
```

**Upload Failures**:
```bash
# Verify HuggingFace authentication
huggingface-cli login

# Check network connectivity
ping huggingface.co
```

## 📈 Expected Performance

### Lightning AI A100 80GB Specs
- **VRAM**: 80GB (75GB used for training)
- **FP32 Performance**: 19.5 TFLOPs
- **BF16 Performance**: 312 TFLOPs (used for training)
- **Memory Bandwidth**: 2TB/s

### Training Metrics
- **Tokens/Second**: ~4,000-5,000 tokens/sec (12 vCPUs + 170GB RAM)
- **Steps/Hour**: ~1,450 steps/hour
- **Total Steps**: 10,131 steps
- **Estimated Time**: 7 hours

## 🎯 Success Checklist

- [ ] Lightning AI A100 instance launched
- [ ] Training package uploaded and extracted
- [ ] `./setup.sh` completed successfully
- [ ] WandB authentication configured
- [ ] HuggingFace authentication verified
- [ ] Training started with `uv run python train.py`
- [ ] Monitoring active (WandB + system monitor)
- [ ] Checkpoints saving every ~1 hour
- [ ] Post-training pipeline completes
- [ ] Model uploaded to HuggingFace as "Wayfarer2-Pixelated"

## 💡 Pro Tips

1. **Use Interruptible Instances**: Save 48% on costs with minimal risk
2. **Monitor Early**: Watch first few steps to catch issues quickly
3. **Checkpoint Strategy**: Training resumes automatically from interruptions
4. **Cost Tracking**: Monitor Lightning credits in real-time
5. **Parallel Monitoring**: Use both Lightning AI dashboard and WandB

---

**Total Cost**: ~$21-40 depending on instance type
**Total Time**: ~8 hours including setup and post-processing
**Output**: Production-ready model with multiple quantizations on HuggingFace
