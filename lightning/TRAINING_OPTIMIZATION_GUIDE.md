# Training Optimization Guide - 12-Hour Window

## 🎯 Overview

This guide explains how to optimize therapeutic AI training to complete within a 12-hour window on H100 GPUs while maximizing model quality.

## 🚀 Quick Start

### Automatic Optimization (Recommended)

```bash
# Automatically optimizes for your dataset
python train_optimized.py
```

The optimizer will:
1. Analyze your dataset (size, token distribution)
2. Select optimal batch size and accumulation steps
3. Estimate training time
4. Adjust parameters if needed to fit 12-hour window
5. Train with optimal configuration

### Manual Optimization

```python
from training_optimizer import optimize_for_dataset

# Optimize for your dataset
profile, estimate, training_args = optimize_for_dataset(
    num_samples=8000,
    avg_tokens_per_sample=500,
    num_epochs=3,
    priority='balanced',  # 'fast', 'balanced', 'quality', 'memory_efficient'
    max_hours=12.0
)

# Use the optimized training_args
```

## 📊 Optimization Profiles

### Fast Profile
**Best for**: Large datasets, tight time constraints

```python
{
    'batch_size': 8,
    'gradient_accumulation_steps': 4,
    'max_length': 1024,
    'estimated_throughput': 1200 tokens/sec,
    'memory_usage': 75 GB
}
```

**Characteristics**:
- Highest throughput (~1200 tokens/sec)
- Shorter sequences (1024 tokens)
- Higher memory usage (75GB)
- Completes fastest

### Balanced Profile (Default)
**Best for**: Most use cases

```python
{
    'batch_size': 4,
    'gradient_accumulation_steps': 8,
    'max_length': 2048,
    'estimated_throughput': 800 tokens/sec,
    'memory_usage': 60 GB
}
```

**Characteristics**:
- Good throughput (~800 tokens/sec)
- Standard sequences (2048 tokens)
- Moderate memory usage (60GB)
- Best quality/speed tradeoff

### Quality Profile
**Best for**: Smaller datasets, maximum quality

```python
{
    'batch_size': 2,
    'gradient_accumulation_steps': 16,
    'max_length': 4096,
    'estimated_throughput': 400 tokens/sec,
    'memory_usage': 70 GB
}
```

**Characteristics**:
- Lower throughput (~400 tokens/sec)
- Longer sequences (4096 tokens)
- Higher memory usage (70GB)
- Best model quality

### Memory Efficient Profile
**Best for**: Memory constraints, very large models

```python
{
    'batch_size': 1,
    'gradient_accumulation_steps': 32,
    'max_length': 2048,
    'estimated_throughput': 300 tokens/sec,
    'memory_usage': 45 GB
}
```

**Characteristics**:
- Lowest throughput (~300 tokens/sec)
- Standard sequences (2048 tokens)
- Lowest memory usage (45GB)
- Fits in smaller GPUs

## ⏰ Time Estimation

### How It Works

The optimizer estimates training time using:

```python
total_tokens = num_samples × avg_tokens × num_epochs
training_time = total_tokens / throughput
```

### Example Calculation

For 8,000 samples, 500 tokens average, 3 epochs:

```
Total tokens = 8,000 × 500 × 3 = 12,000,000 tokens

With balanced profile (800 tokens/sec):
Training time = 12,000,000 / 800 = 15,000 seconds = 4.17 hours ✅

With quality profile (400 tokens/sec):
Training time = 12,000,000 / 400 = 30,000 seconds = 8.33 hours ✅

With 16,000 samples:
Training time = 24,000,000 / 800 = 30,000 seconds = 8.33 hours ✅
```

### Safety Margin

The optimizer includes a 30-minute safety margin:
- **Available time**: 11.5 hours (12h - 0.5h margin)
- **Ensures**: Time for final checkpoint and cleanup

## 🔧 Configuration

### In training_config.json

```json
{
  "num_train_epochs": 3,
  "optimization_priority": "balanced",
  "max_training_hours": 12.0,
  "base_model": "LatitudeGames/Wayfarer-2-12B"
}
```

### Priority Options

- **`fast`**: Maximize speed, may sacrifice some quality
- **`balanced`**: Best tradeoff (recommended)
- **`quality`**: Maximize quality, slower training
- **`memory_efficient`**: Minimize memory usage

## 📈 Monitoring

### Real-time Progress

The optimizer tracks:

```
⏰ Elapsed: 3.5 hours | Remaining: 8.5 hours
📊 Progress: 29.2% | On track: ✅
🎯 Steps/hour: 428 | ETA: 2024-10-27 18:30:00
```

### WandB Metrics

Logged automatically:
- `optimization/estimated_hours`: Predicted duration
- `optimization/actual_hours`: Actual duration
- `optimization/time_accuracy`: Prediction accuracy
- `training/elapsed_hours`: Current elapsed time
- `training/remaining_hours`: Estimated remaining time
- `training/on_track`: Whether on schedule

## 🎛️ Advanced Tuning

### Custom Profile

```python
from training_optimizer import H100OptimizationProfile

custom_profile = H100OptimizationProfile(
    batch_size=6,
    gradient_accumulation_steps=6,
    max_length=3072,
    num_workers=4,
    use_gradient_checkpointing=True,
    use_bf16=True,
    use_fused_optimizer=True,
    estimated_throughput=600.0,
    memory_usage_gb=65.0
)
```

### Adjusting for Dataset Size

**Small datasets (<5,000 samples)**:
- Use `quality` profile
- Increase epochs (4-5)
- Longer sequences (4096)

**Medium datasets (5,000-15,000 samples)**:
- Use `balanced` profile (default)
- Standard epochs (3)
- Standard sequences (2048)

**Large datasets (>15,000 samples)**:
- Use `fast` profile
- Reduce epochs (2)
- Shorter sequences (1024)

### Adjusting for Token Length

**Short conversations (<300 tokens)**:
- Increase batch size
- Use shorter max_length (1024)
- Higher throughput possible

**Long conversations (>800 tokens)**:
- Decrease batch size
- Use longer max_length (4096)
- Lower throughput expected

## 🐛 Troubleshooting

### Training Too Slow

**Symptoms**: Estimated time > 12 hours

**Solutions**:
1. Switch to `fast` profile
2. Reduce epochs
3. Reduce max_length
4. Increase batch size (if memory allows)

```python
# Example adjustment
profile, estimate, args = optimize_for_dataset(
    num_samples=8000,
    avg_tokens_per_sample=500,
    num_epochs=2,  # Reduced from 3
    priority='fast',  # Changed from 'balanced'
    max_hours=12.0
)
```

### Out of Memory

**Symptoms**: CUDA OOM error

**Solutions**:
1. Switch to `memory_efficient` profile
2. Reduce batch size
3. Reduce max_length
4. Enable gradient checkpointing

```python
# Memory-efficient settings
profile = H100OptimizationProfile(
    batch_size=1,
    gradient_accumulation_steps=32,
    max_length=2048,
    use_gradient_checkpointing=True
)
```

### Training Too Fast

**Symptoms**: Completes in <6 hours, poor quality

**Solutions**:
1. Switch to `quality` profile
2. Increase epochs
3. Increase max_length
4. Decrease batch size

```python
# Quality-focused settings
profile, estimate, args = optimize_for_dataset(
    num_samples=8000,
    avg_tokens_per_sample=500,
    num_epochs=4,  # Increased from 3
    priority='quality',
    max_hours=12.0
)
```

### Inaccurate Time Estimates

**Symptoms**: Actual time differs significantly from estimate

**Causes**:
- First run (no baseline throughput)
- Different hardware than expected
- Dataset characteristics vary

**Solutions**:
1. Run with actual throughput measurement
2. Adjust throughput estimates in profile
3. Use conservative estimates (0.7x baseline)

## 📊 Optimization Report

After optimization, a report is saved:

```json
{
  "timestamp": "2024-10-27T10:00:00",
  "max_training_hours": 12.0,
  "profile_name": "balanced",
  "profile": {
    "batch_size": 4,
    "gradient_accumulation_steps": 8,
    "estimated_throughput": 800.0
  },
  "estimate": {
    "total_steps": 750,
    "estimated_hours": 4.17,
    "fits_in_window": true
  },
  "dataset_info": {
    "num_samples": 8000,
    "avg_tokens_per_sample": 500
  }
}
```

## 🎯 Best Practices

### 1. Start with Balanced Profile

```bash
python train_optimized.py
```

Let the optimizer choose the best configuration.

### 2. Monitor First Run

Watch the first training run to validate estimates:
- Check actual throughput
- Verify memory usage
- Confirm time estimates

### 3. Adjust Based on Results

If first run shows:
- **Faster than expected**: Switch to quality profile
- **Slower than expected**: Switch to fast profile
- **OOM errors**: Switch to memory_efficient profile

### 4. Use Checkpoints

Training saves checkpoints every 30 minutes:
- Resume if interrupted
- Evaluate intermediate models
- Stop early if quality sufficient

### 5. Validate Time Estimates

After first run, check accuracy:

```python
actual_hours = 4.5
estimated_hours = 4.17
accuracy = (estimated_hours / actual_hours) * 100  # 92.7%
```

Good accuracy: 85-115%

## 📚 Examples

### Example 1: Standard Training

```bash
# 8,000 samples, balanced profile
python train_optimized.py
```

**Result**:
- Profile: balanced
- Estimated: 4.2 hours
- Actual: 4.5 hours
- Fits in window: ✅

### Example 2: Large Dataset

```python
# 20,000 samples, need fast profile
from training_optimizer import optimize_for_dataset

profile, estimate, args = optimize_for_dataset(
    num_samples=20000,
    avg_tokens_per_sample=500,
    num_epochs=2,
    priority='fast',
    max_hours=12.0
)
```

**Result**:
- Profile: fast
- Estimated: 9.7 hours
- Fits in window: ✅

### Example 3: High Quality

```python
# 5,000 samples, maximize quality
profile, estimate, args = optimize_for_dataset(
    num_samples=5000,
    avg_tokens_per_sample=600,
    num_epochs=4,
    priority='quality',
    max_hours=12.0
)
```

**Result**:
- Profile: quality
- Estimated: 10.8 hours
- Fits in window: ✅

## 🔍 Understanding the Math

### Throughput Calculation

```
Throughput (tokens/sec) = 
    (batch_size × seq_length × devices) / 
    (forward_time + backward_time + optimizer_time)
```

### H100 Baseline

- **Peak**: ~1500 tokens/sec (theoretical)
- **Practical**: ~1000 tokens/sec (with overhead)
- **Conservative**: ~700 tokens/sec (safe estimate)

### Profile Throughput

- **Fast**: 1200 tokens/sec (80% of peak)
- **Balanced**: 800 tokens/sec (53% of peak)
- **Quality**: 400 tokens/sec (27% of peak)
- **Memory Efficient**: 300 tokens/sec (20% of peak)

## ✅ Checklist

Before training:
- [ ] Dataset analyzed
- [ ] Profile selected
- [ ] Time estimate reviewed
- [ ] Fits in 12-hour window
- [ ] WandB configured
- [ ] Checkpoints enabled

During training:
- [ ] Monitor progress
- [ ] Check time estimates
- [ ] Verify memory usage
- [ ] Watch for errors

After training:
- [ ] Review actual vs estimated time
- [ ] Check model quality
- [ ] Save optimization report
- [ ] Update estimates for next run

---

**Questions?** Check the main documentation or optimization report.
