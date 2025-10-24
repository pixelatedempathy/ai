# MoE Training System for Lightning.ai H100

This is a production-ready training system for the Pixelated Empathy MoE (Mixture of Experts) model, optimized for Lightning.ai H100 GPUs.

## 🎯 Overview

The system trains 4 specialized LoRA experts on a shared base model:
- **Therapeutic Expert**: Trauma-informed therapeutic responses
- **Educational Expert**: Clinical explanations and research-based content  
- **Empathetic Expert**: Emotional validation and support
- **Practical Expert**: Actionable advice and coping strategies

## 🏗️ Architecture

```
Base Model (DialoGPT-medium 355M)
├── Therapeutic LoRA Expert (16r/32α)
├── Educational LoRA Expert (16r/32α) 
├── Empathetic LoRA Expert (16r/32α)
└── Practical LoRA Expert (16r/32α)
```

**Key Features:**
- **H100 Optimized**: bf16 precision, gradient checkpointing, optimal batch sizes
- **Memory Efficient**: LoRA adapters (~100MB each vs 700MB base model)
- **High Quality Data**: 5,016 high-quality examples from Tim Fletcher transcripts
- **Production Ready**: Comprehensive logging, monitoring, and error handling

## 📊 Training Data

| Expert | High Quality | Medium Quality | Total |
|--------|-------------|----------------|-------|
| Therapeutic | 1,430 | 12 | 1,442 |
| Educational | 1,667 | 0 | 1,667 |
| Empathetic | 321 | 57 | 378 |
| Practical | 1,598 | 333 | 1,931 |
| **Total** | **5,016** | **402** | **5,418** |

## 🚀 Quick Start

### 1. Setup Lightning.ai Environment

```bash
# In your Lightning.ai studio terminal
git clone <your-repo>
cd pixelated-training/moe_training
./setup_lightning.sh
```

### 2. Upload Training Data

Upload these files to `/teamspace/studios/this_studio/moe_training_data/`:
- `therapeutic_high_quality.json`
- `educational_high_quality.json`
- `empathetic_high_quality.json`
- `practical_high_quality.json`

### 3. Configure Weights & Biases (Optional)

```bash
wandb login
# Or for offline mode:
export WANDB_MODE=offline
```

### 4. Start Training

```bash
./launch_training.sh
```

### 5. Monitor Progress

```bash
# In another terminal
python monitor_training.py
```

## ⚙️ Configuration

### Model Configuration
- **Base Model**: microsoft/DialoGPT-medium (355M parameters)
- **LoRA**: r=16, α=32, dropout=0.05
- **Target Modules**: c_attn, c_proj, c_fc
- **Max Length**: 512 tokens

### Training Configuration
- **Batch Size**: 8 per device × 4 accumulation = 32 effective
- **Learning Rate**: 2e-4 (adjusted per expert)
- **Epochs**: 3
- **Precision**: bf16-mixed
- **Optimizer**: AdamW with cosine scheduling

### H100 Optimizations
- **Memory Usage**: ~15GB (well under 80GB limit)
- **Gradient Checkpointing**: Enabled
- **Mixed Precision**: bf16 for H100 efficiency
- **DataLoader**: 8 workers with pin_memory

## 📈 Expected Results

### Training Time (H100)
- **Per Expert**: ~2-3 hours
- **Total Training**: ~8-12 hours (sequential)
- **Evaluation**: ~30 minutes per expert

### Performance Metrics
- **Target Perplexity**: <15 (lower is better)
- **Target Accuracy**: >85%
- **Memory Usage**: <20GB during training

## 📁 Output Structure

```
/teamspace/studios/this_studio/models/
├── therapeutic_expert/
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   ├── expert_metadata.json
│   └── training_results.json
├── educational_expert/
├── empathetic_expert/
├── practical_expert/
└── moe_training_summary.json
```

## 🔧 Advanced Configuration

### Custom Learning Rates
```python
# In config.py, adjust expert-specific multipliers:
"therapeutic": learning_rate_multiplier=1.0,
"educational": learning_rate_multiplier=0.8,
"empathetic": learning_rate_multiplier=1.2,  # Higher for less data
"practical": learning_rate_multiplier=0.9,
```

### Memory Optimization
```python
# Reduce batch size if OOM
batch_size: int = 4  # Instead of 8
gradient_accumulation_steps: int = 8  # Instead of 4
```

### Data Augmentation
```python
# Enable for smaller datasets
augment_data: bool = True  # In TherapeuticDataset
```

## 🐛 Troubleshooting

### Out of Memory (OOM)
```bash
# Reduce batch size
# Edit config.py: batch_size = 4, gradient_accumulation_steps = 8
```

### Slow Training
```bash
# Check GPU utilization
nvidia-smi
# Should show ~90%+ GPU utilization
```

### Data Loading Issues
```bash
# Verify data files
ls -la /teamspace/studios/this_studio/moe_training_data/
# Check file formats
python -c "import json; print(json.load(open('therapeutic_high_quality.json'))[:1])"
```

### W&B Connection Issues
```bash
# Use offline mode
export WANDB_MODE=offline
```

## 📊 Monitoring

### Real-time Monitoring
```bash
# GPU usage
watch -n 1 nvidia-smi

# Training logs
tail -f /teamspace/studios/this_studio/training.log

# W&B dashboard
# Visit wandb.ai/your-username/pixelated-empathy-moe
```

### Key Metrics to Watch
- **GPU Memory**: Should stay <75GB
- **Training Loss**: Should decrease steadily
- **Eval Loss**: Should decrease without overfitting
- **Perplexity**: Target <15

## 🚀 Deployment

After training, the models are ready for inference:

```python
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load base model
base_model = AutoModelForCausalLM.from_pretrained("microsoft/DialoGPT-medium")

# Load expert adapter
model = PeftModel.from_pretrained(base_model, "/path/to/therapeutic_expert")

# Generate response
inputs = tokenizer("I'm struggling with trauma", return_tensors="pt")
outputs = model.generate(**inputs, max_length=200)
```

## 📝 Notes

- **Sequential Training**: Experts are trained one at a time to maximize H100 utilization
- **Automatic Checkpointing**: Models are saved every 500 steps
- **Early Stopping**: Training stops if validation loss doesn't improve
- **Quality Filtering**: Only high-quality examples are used for training

## 🆘 Support

If you encounter issues:
1. Check the troubleshooting section above
2. Review training logs in `/teamspace/studios/this_studio/training.log`
3. Monitor GPU usage with `nvidia-smi`
4. Verify data files are properly formatted

## 🎯 Success Criteria

Training is successful when:
- ✅ All 4 experts complete training without errors
- ✅ Validation perplexity < 15 for each expert
- ✅ Generated samples show appropriate style characteristics
- ✅ Models are saved and ready for inference

**Ready to train world-class therapeutic AI on H100! 🚀**
