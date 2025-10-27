# Therapeutic AI MoE Training Guide

## 🚀 Quick Start

### Prerequisites
- Lightning.ai H100 GPU access
- Training dataset prepared (`training_dataset.json`)
- WandB account configured (`wandb_config.json`)

### Installation

```bash
# Install dependencies
pip install -r requirements_moe.txt

# Or with uv (recommended)
uv pip install -r requirements_moe.txt
```

### Training

```bash
# Start MoE training with H100 optimizations
python train_moe_h100.py
```

## 🧠 Architecture Overview

### Multi-Expert Mixture of Experts (MoE)

The therapeutic AI uses a 4-expert MoE architecture with domain specialization:

1. **Psychology Expert**: General psychological concepts and theories
2. **Mental Health Expert**: Clinical mental health, disorders, treatments
3. **Bias Detection Expert**: Identifies and mitigates biases in responses
4. **General Therapeutic Expert**: Broad therapeutic conversation skills

### Expert Routing

- **Dynamic Routing**: Each token is routed to the top-2 most relevant experts
- **Load Balancing**: Ensures even distribution across experts
- **Domain Classification**: Interpretable routing based on content domain

### LoRA Fine-Tuning

- **Rank**: 16 (efficient parameter updates)
- **Alpha**: 32 (scaling factor)
- **Target Modules**: q_proj, v_proj, k_proj, o_proj
- **Trainable Parameters**: ~1-2% of total model parameters

### Extended Context

- **Training Length**: 2048 tokens
- **Maximum Context**: 8192 tokens (4x training length)
- **Session Continuity**: Supports long therapeutic conversations

## ⚙️ Configuration

### Training Parameters

Edit `moe_training_config.json`:

```json
{
  "num_train_epochs": 3,
  "per_device_train_batch_size": 4,
  "gradient_accumulation_steps": 8,
  "learning_rate": 3e-4,
  "warmup_steps": 1000
}
```

**Effective Batch Size**: `batch_size × gradient_accumulation_steps = 32`

### H100 Optimizations

- **BFloat16**: Native H100 precision for faster training
- **Fused Optimizer**: `adamw_torch_fused` for H100
- **Gradient Checkpointing**: Reduces memory usage
- **Dataloader Workers**: 4 parallel workers
- **Pin Memory**: Faster CPU-GPU transfers

### Time Constraints

- **Maximum Duration**: 12 hours
- **Checkpoint Interval**: 30 minutes
- **Early Stopping**: 3 epochs patience
- **Auto-save**: Before time limit reached

## 📊 Monitoring

### WandB Metrics

Training metrics logged to Weights & Biases:

- **Loss**: Training and validation loss
- **Expert Usage**: Distribution across experts
- **Routing Entropy**: Expert selection diversity
- **Time Progress**: Elapsed and remaining hours
- **Model Stats**: Parameters, memory usage

### Real-time Progress

```
📊 Progress: 45.2% | Loss: 1.234 | Step: 1500
⏰ Elapsed: 5.5 hours | Remaining: 6.5 hours
💾 Checkpoint at 5.5 hours
```

## 🎯 Training Workflow

### 1. Data Preparation

Ensure your training data is in the correct format:

```json
{
  "conversations": [
    {
      "text": "Therapist: How are you feeling today?\nClient: I've been struggling with anxiety..."
    }
  ]
}
```

### 2. Configuration

Review and adjust:
- `moe_training_config.json` - Training parameters
- `wandb_config.json` - Logging configuration
- `safety_config.json` - Safety constraints

### 3. Launch Training

```bash
# Start training
python train_moe_h100.py

# Monitor in WandB dashboard
# https://wandb.ai/your-project
```

### 4. Checkpoints

Automatic checkpoints saved every 30 minutes:
```
therapeutic_moe_model/
├── checkpoint-500/
├── checkpoint-1000/
├── checkpoint-1500/
└── moe_layers.pt
```

### 5. Model Export

After training completes:
```bash
# Model saved to:
therapeutic_moe_model/
├── adapter_config.json
├── adapter_model.bin
├── moe_layers.pt
└── tokenizer files
```

## 🔧 Advanced Configuration

### Adjusting Expert Count

```python
moe_config = MoEConfig(
    num_experts=8,  # Increase for more specialization
    expert_domains=[
        "psychology",
        "mental_health",
        "bias_detection",
        "general_therapeutic",
        "crisis_intervention",
        "trauma_therapy",
        "cognitive_behavioral",
        "mindfulness"
    ]
)
```

### Custom Expert Domains

Define your own expert specializations:

```python
expert_domains = [
    "child_psychology",
    "couples_therapy",
    "addiction_counseling",
    "grief_counseling"
]
```

### LoRA Rank Tuning

- **Lower Rank (8)**: Faster training, less capacity
- **Medium Rank (16)**: Balanced (recommended)
- **Higher Rank (32)**: More capacity, slower training

### Context Length

```python
max_position_embeddings = 16384  # For very long sessions
```

## 📈 Performance Expectations

### Training Time

- **Dataset Size**: 8,000 conversations
- **Expected Duration**: 8-10 hours on H100
- **Checkpoints**: Every 30 minutes
- **Final Model Size**: ~1.5GB (LoRA adapters)

### Model Quality

- **Target Loss**: < 1.5
- **Perplexity**: < 2.5
- **Expert Balance**: 20-30% per expert
- **Routing Entropy**: > 1.0

### Resource Usage

- **GPU Memory**: ~40-60GB (H100 has 80GB)
- **Training Speed**: ~500-1000 tokens/sec
- **Checkpoint Size**: ~1.5GB per checkpoint

## 🐛 Troubleshooting

### Out of Memory

Reduce batch size:
```json
{
  "per_device_train_batch_size": 2,
  "gradient_accumulation_steps": 16
}
```

### Slow Training

Increase batch size (if memory allows):
```json
{
  "per_device_train_batch_size": 8,
  "gradient_accumulation_steps": 4
}
```

### Poor Expert Balance

Increase load balancing weight:
```json
{
  "load_balancing_weight": 0.05
}
```

### Training Timeout

Reduce epochs or enable early stopping:
```json
{
  "num_train_epochs": 2,
  "early_stopping_patience": 2
}
```

## 🔒 Safety Features

### Time Constraints

- Automatic stop before 12-hour limit
- Graceful shutdown with checkpoint save
- Resume capability from last checkpoint

### Early Stopping

- Monitors validation loss
- Stops if no improvement for 3 epochs
- Saves best model automatically

### Checkpoint Recovery

```bash
# Resume from checkpoint
python train_moe_h100.py --resume_from_checkpoint therapeutic_moe_model/checkpoint-1500
```

## 📚 Additional Resources

- **MoE Architecture**: `moe_architecture.py`
- **Training Script**: `train_moe_h100.py`
- **Configuration**: `moe_training_config.json`
- **Requirements**: `requirements_moe.txt`

## 🎉 Next Steps

After training completes:

1. **Evaluate Model**: Test on held-out therapeutic conversations
2. **Deploy**: Use deployment scripts in `ai/deployment/`
3. **Monitor**: Set up production monitoring
4. **Iterate**: Fine-tune based on performance metrics

---

**Questions?** Check the main documentation or open an issue.
