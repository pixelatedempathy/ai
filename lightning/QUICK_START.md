# Quick Start Guide - Therapeutic AI Training

## 🚀 One-Command Training

```bash
# Automatic optimization for 12-hour window
python train_optimized.py
```

That's it! The system will:
- ✅ Analyze your dataset
- ✅ Select optimal configuration
- ✅ Ensure completion within 12 hours
- ✅ Train MoE model with LoRA
- ✅ Save checkpoints every 30 minutes

## 📋 Prerequisites

```bash
# Install dependencies
pip install -r requirements_moe.txt

# Ensure you have:
# - training_dataset.json (your data)
# - wandb_config.json (logging)
# - safety_config.json (safety rules)
# - training_config.json (base config)
```

## 🎯 What You Get

### Model Architecture
- **4 Domain Experts**: Psychology, Mental Health, Bias Detection, General Therapeutic
- **LoRA Fine-tuning**: ~1-2% trainable parameters
- **Extended Context**: 8192 tokens (4x training length)
- **H100 Optimized**: BFloat16, fused optimizer, gradient checkpointing

### Training Features
- **Automatic Optimization**: Fits in 12-hour window
- **Smart Checkpointing**: Every 30 minutes
- **Early Stopping**: 3-epoch patience
- **Real-time Monitoring**: WandB integration
- **Graceful Shutdown**: Saves before time limit

## 📊 Expected Results

### For 8,000 Samples
- **Training Time**: 4-5 hours
- **Model Size**: ~1.5GB (LoRA adapters)
- **Memory Usage**: 60GB (H100 has 80GB)
- **Target Loss**: < 1.5
- **Perplexity**: < 2.5

### For 16,000 Samples
- **Training Time**: 8-9 hours
- **Model Size**: ~1.5GB
- **Memory Usage**: 60GB
- **Completes**: Within 12-hour window ✅

## 🔧 Configuration Options

### In training_config.json

```json
{
  "num_train_epochs": 3,
  "optimization_priority": "balanced",
  "max_training_hours": 12.0
}
```

### Priority Options
- **`fast`**: Fastest training, good quality
- **`balanced`**: Best tradeoff (default)
- **`quality`**: Maximum quality, slower
- **`memory_efficient`**: Lowest memory usage

## 📈 Monitoring

### Console Output
```
📊 Progress: 45.2% | Loss: 1.234 | Step: 1500
⏰ Elapsed: 3.5h | Remaining: 8.5h | On track: ✅
💾 Checkpoint at 3.5 hours
```

### WandB Dashboard
- Training loss and validation accuracy
- Expert usage distribution
- Time progress and estimates
- Model parameters and memory

## 🎓 Training Profiles

| Profile | Speed | Quality | Memory | Best For |
|---------|-------|---------|--------|----------|
| Fast | ⚡⚡⚡ | ⭐⭐ | 75GB | Large datasets |
| Balanced | ⚡⚡ | ⭐⭐⭐ | 60GB | Most cases |
| Quality | ⚡ | ⭐⭐⭐⭐ | 70GB | Small datasets |
| Memory Efficient | ⚡ | ⭐⭐ | 45GB | Memory limits |

## 🐛 Troubleshooting

### Out of Memory
```bash
# Use memory-efficient profile
# Edit training_config.json:
{
  "optimization_priority": "memory_efficient"
}
```

### Training Too Slow
```bash
# Use fast profile
{
  "optimization_priority": "fast"
}
```

### Won't Fit in 12 Hours
The optimizer will automatically:
1. Try faster profile
2. Reduce epochs if needed
3. Adjust batch size
4. Warn you if still won't fit

## 📁 Output Files

After training:
```
therapeutic_moe_model/
├── adapter_config.json          # LoRA config
├── adapter_model.bin            # LoRA weights (~1.5GB)
├── moe_layers.pt                # MoE expert weights
├── tokenizer files              # Tokenizer
└── checkpoints/                 # Training checkpoints
    ├── checkpoint-500/
    ├── checkpoint-1000/
    └── checkpoint-1500/
```

## 🎯 Next Steps

After training:
1. **Evaluate**: Test on held-out data
2. **Deploy**: Use deployment scripts
3. **Monitor**: Set up production monitoring
4. **Iterate**: Fine-tune based on results

## 📚 More Information

- **Full Guide**: `MOE_TRAINING_GUIDE.md`
- **Optimization**: `TRAINING_OPTIMIZATION_GUIDE.md`
- **Architecture**: `moe_architecture.py`
- **Training Script**: `train_optimized.py`

## ⚡ Advanced Usage

### Manual Optimization
```python
from training_optimizer import optimize_for_dataset

profile, estimate, args = optimize_for_dataset(
    num_samples=8000,
    avg_tokens_per_sample=500,
    num_epochs=3,
    priority='balanced'
)
```

### Custom Configuration
```python
from moe_architecture import MoEConfig

config = MoEConfig(
    num_experts=4,
    lora_r=16,
    lora_alpha=32,
    max_position_embeddings=8192
)
```

## ✅ Checklist

Before training:
- [ ] Dataset prepared (`training_dataset.json`)
- [ ] WandB configured
- [ ] Dependencies installed
- [ ] GPU available (H100 recommended)

During training:
- [ ] Monitor progress in console
- [ ] Check WandB dashboard
- [ ] Verify checkpoints saving

After training:
- [ ] Model saved successfully
- [ ] Evaluate on test data
- [ ] Review training metrics
- [ ] Plan deployment

---

**Ready to train?** Run `python train_optimized.py` and you're good to go! 🚀
