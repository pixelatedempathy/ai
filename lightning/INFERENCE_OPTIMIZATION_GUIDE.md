# Inference Optimization Guide - <2 Second Latency

## 🎯 Target: P95 Latency < 2 Seconds

This guide explains how to achieve sub-2-second response times for therapeutic AI inference.

## 🚀 Quick Start

### Start Inference Service

```bash
# Start the optimized inference service
python inference_service.py
```

Service will be available at `http://localhost:8000`

### Test Inference

```bash
# Test with curl
curl -X POST http://localhost:8000/api/v1/inference \
  -H "Content-Type: application/json" \
  -d '{
    "user_input": "I'\''ve been feeling anxious lately.",
    "use_cache": true
  }'
```

### Run Benchmark

```bash
# Benchmark performance
python benchmark_inference.py
```

## 🔧 Optimization Techniques

### 1. Model Compilation

**PyTorch 2.0+ torch.compile**:
- Reduces overhead by 20-30%
- Optimizes computation graph
- Automatic kernel fusion

```python
model = torch.compile(model, mode='reduce-overhead')
```

### 2. Response Caching

**LRU Cache with TTL**:
- Caches recent responses
- 1-hour TTL (configurable)
- 1000 entry limit (configurable)

**Performance Impact**:
- Cache hit: ~10-50ms
- Cache miss: ~500-1500ms
- Hit rate: 30-50% typical

### 3. Flash Attention

**Memory-efficient attention**:
- 2-4x faster than standard attention
- Lower memory usage
- Supported on A100/H100

```python
model.config.use_flash_attention_2 = True
```

### 4. BFloat16 Precision

**Half-precision inference**:
- 2x faster than FP32
- Native H100 support
- Minimal quality loss

```python
model = model.to(torch.bfloat16)
```

### 5. KV Cache

**Key-Value caching**:
- Reuses attention keys/values
- Faster for long contexts
- Enabled by default

```python
generation_config = GenerationConfig(use_cache=True)
```

### 6. Batch Processing

**Process multiple requests**:
- Better GPU utilization
- Higher throughput
- Slightly higher latency per request

### 7. Context Truncation

**Limit context length**:
- Faster processing
- Lower memory usage
- Keep last N messages

```python
config = InferenceConfig(
    max_context_length=2048,
    context_window=10  # Last 10 messages
)
```

## 📊 Performance Characteristics

### Expected Latencies

| Configuration | P50 | P95 | P99 | Cache Hit Rate |
|--------------|-----|-----|-----|----------------|
| No optimizations | 2.5s | 3.5s | 4.0s | 0% |
| With caching | 0.8s | 1.5s | 2.0s | 40% |
| Full optimizations | 0.5s | 1.2s | 1.8s | 50% |

### Breakdown by Component

```
Total Latency (1.2s):
├── Cache lookup:        10ms
├── Tokenization:        50ms
├── Model forward:      800ms
├── Generation:         300ms
└── Decoding:            40ms
```

### Throughput

- **Sequential**: 1-2 requests/second
- **Concurrent (5x)**: 3-5 requests/second
- **Batched (8x)**: 6-10 requests/second

## 🎛️ Configuration

### InferenceConfig

```python
config = InferenceConfig(
    # Generation
    max_new_tokens=256,        # Response length
    temperature=0.7,           # Randomness
    top_p=0.9,                 # Nucleus sampling
    top_k=50,                  # Top-k sampling
    repetition_penalty=1.1,    # Prevent repetition
    
    # Performance
    use_cache=True,            # Enable KV cache
    use_flash_attention=True,  # Flash attention
    compile_model=True,        # torch.compile
    
    # Context
    max_context_length=2048,   # Max input length
    context_window=10,         # Messages to keep
    
    # Caching
    enable_response_cache=True,
    cache_size=1000,
    cache_ttl_seconds=3600
)
```

### Environment Variables

```bash
# Model path
export MODEL_PATH="./therapeutic_moe_model"

# Device
export DEVICE="cuda"  # or "cpu"

# Cache settings
export CACHE_SIZE=1000
export CACHE_TTL=3600

# Service settings
export HOST="0.0.0.0"
export PORT=8000
export WORKERS=1
```

## 🔍 Monitoring

### Health Check

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "uptime_seconds": 3600.5,
  "total_requests": 1250,
  "avg_latency": 0.85,
  "p95_latency": 1.45,
  "cache_hit_rate": 0.42,
  "meets_sla": true
}
```

### Metrics

```bash
curl http://localhost:8000/metrics
```

Response:
```json
{
  "total_requests": 1250,
  "avg_latency": 0.85,
  "p50_latency": 0.65,
  "p95_latency": 1.45,
  "p99_latency": 1.85,
  "cache_hit_rate": 0.42,
  "cache_stats": {
    "size": 420,
    "max_size": 1000,
    "utilization": 0.42
  },
  "meets_sla": true
}
```

## 🐛 Troubleshooting

### High Latency (>2s)

**Symptoms**: P95 latency exceeds 2 seconds

**Solutions**:
1. Enable response caching
2. Reduce max_new_tokens
3. Enable model compilation
4. Use Flash Attention
5. Reduce context_window

```python
# Aggressive optimization
config = InferenceConfig(
    max_new_tokens=128,      # Shorter responses
    context_window=5,        # Less context
    compile_model=True,
    use_flash_attention=True,
    enable_response_cache=True
)
```

### Low Cache Hit Rate (<20%)

**Symptoms**: Most requests are cache misses

**Causes**:
- Unique user inputs
- Short cache TTL
- Small cache size

**Solutions**:
1. Increase cache size
2. Increase cache TTL
3. Normalize inputs before caching

### Out of Memory

**Symptoms**: CUDA OOM errors

**Solutions**:
1. Reduce max_context_length
2. Reduce batch_size
3. Enable gradient checkpointing (training only)
4. Use smaller model

```python
config = InferenceConfig(
    max_context_length=1024,  # Shorter context
    batch_size=1              # No batching
)
```

### Slow First Request

**Symptoms**: First request takes 5-10 seconds

**Cause**: Model compilation and warmup

**Solution**: Run warmup on startup

```python
engine.warmup(num_requests=10)
```

## 📈 Optimization Checklist

Before deployment:
- [ ] Model compiled with torch.compile
- [ ] Flash Attention enabled
- [ ] BFloat16 precision
- [ ] Response caching enabled
- [ ] KV cache enabled
- [ ] Warmup completed
- [ ] Benchmark run (P95 < 2s)

During operation:
- [ ] Monitor latency metrics
- [ ] Track cache hit rate
- [ ] Watch memory usage
- [ ] Check error rates

## 🎯 Best Practices

### 1. Warmup on Startup

```python
# Warmup model before serving
engine.warmup(num_requests=10)
```

### 2. Monitor Continuously

```python
# Check metrics regularly
metrics = engine.get_metrics()
if not metrics['meets_sla']:
    alert_ops_team()
```

### 3. Tune Cache Size

```python
# Adjust based on traffic
if cache_hit_rate < 0.3:
    increase_cache_size()
```

### 4. Use Async for Concurrency

```python
# Handle multiple requests
response = await engine.generate_async(user_input)
```

### 5. Limit Context Length

```python
# Keep only recent messages
conversation_history = conversation_history[-10:]
```

## 📊 Benchmarking

### Run Full Benchmark

```bash
python benchmark_inference.py
```

### Custom Benchmark

```python
from benchmark_inference import InferenceBenchmark

benchmark = InferenceBenchmark(engine)

# Sequential test
result = benchmark.run_sequential_benchmark(
    num_requests=100,
    use_cache=True
)

# Concurrent test
result = await benchmark.run_concurrent_benchmark(
    num_requests=100,
    concurrency=10
)

benchmark.print_results(result)
```

## 🚀 Production Deployment

### Docker

```dockerfile
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Install dependencies
RUN pip install torch transformers fastapi uvicorn

# Copy model and code
COPY therapeutic_moe_model /app/model
COPY inference_*.py /app/

# Run service
CMD ["python", "/app/inference_service.py"]
```

### Kubernetes

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: therapeutic-ai-inference
spec:
  replicas: 3
  template:
    spec:
      containers:
      - name: inference
        image: therapeutic-ai:latest
        resources:
          limits:
            nvidia.com/gpu: 1
            memory: 32Gi
        env:
        - name: MODEL_PATH
          value: "/app/model"
```

### Load Balancing

```nginx
upstream inference_backend {
    least_conn;
    server inference-1:8000;
    server inference-2:8000;
    server inference-3:8000;
}

server {
    location /api/v1/inference {
        proxy_pass http://inference_backend;
        proxy_timeout 5s;
    }
}
```

## 📚 API Reference

### POST /api/v1/inference

Generate therapeutic response.

**Request**:
```json
{
  "user_input": "I've been feeling anxious.",
  "conversation_history": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi, how can I help?"}
  ],
  "system_prompt": "You are a compassionate therapist.",
  "use_cache": true
}
```

**Response**:
```json
{
  "response": "I understand you're feeling anxious...",
  "latency": 0.85,
  "cache_hit": false,
  "tokens_generated": 45,
  "metadata": {
    "generation_time": 0.75,
    "input_tokens": 120
  }
}
```

### GET /health

Health check endpoint.

### GET /metrics

Detailed performance metrics.

### POST /cache/clear

Clear response cache.

---

**Questions?** Check the main documentation or run benchmarks.
