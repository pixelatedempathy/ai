#!/bin/bash

# Set memory management environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

# Clear GPU memory
.venv/bin/python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

# Run training
.venv/bin/python train.py