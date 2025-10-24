#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false

uv run python -c "import torch; torch.cuda.empty_cache()" 2>/dev/null || true

uv run python train_qlora.py