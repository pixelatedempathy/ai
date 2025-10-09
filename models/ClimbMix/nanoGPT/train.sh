#!/bin/bash
set -x

export CUDA_DEVICE_MAX_CONNECTIONS=1
export WANDB_RUN_ID="nanogpt-climbmix"

torchrun --nproc_per_node=$SUBMIT_GPUS --nnodes $NUM_NODES --node_rank $NODE_RANK --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT train.py config/train_gpt2_climbmix.py
