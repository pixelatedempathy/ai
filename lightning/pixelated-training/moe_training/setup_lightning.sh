#!/bin/bash

# Lightning.ai H100 Setup Script for MoE Training
# Run this script in your Lightning.ai studio

set -e

echo "🚀 Setting up MoE training environment on Lightning.ai H100..."

# Check if we're in Lightning.ai environment
if [ ! -d "/teamspace/studios/this_studio" ]; then
    echo "❌ Error: Not in Lightning.ai environment"
    exit 1
fi

# Create directory structure
echo "📁 Creating directory structure..."
mkdir -p /teamspace/studios/this_studio/{models,logs,.cache}
mkdir -p /teamspace/studios/this_studio/moe_training_data

# Install Python dependencies
echo "📦 Installing Python dependencies..."
pip install -r moe_training/requirements.txt

# Check GPU
echo "🔍 Checking GPU availability..."
python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.1f}GB')
else:
    print('❌ No GPU detected')
    exit(1)
"

# Check if training data exists
echo "📊 Checking training data..."
if [ ! -f "/teamspace/studios/this_studio/moe_training_data/therapeutic_high_quality.json" ]; then
    echo "⚠️  Training data not found. Please upload training data to:"
    echo "   /teamspace/studios/this_studio/moe_training_data/"
    echo ""
    echo "Required files:"
    echo "   - therapeutic_high_quality.json"
    echo "   - educational_high_quality.json" 
    echo "   - empathetic_high_quality.json"
    echo "   - practical_high_quality.json"
    echo ""
    echo "You can upload these from your local pixelated-training/moe_training_data/ directory"
else
    echo "✅ Training data found"
    
    # Show data statistics
    python -c "
import json
import os

data_dir = '/teamspace/studios/this_studio/moe_training_data'
total_examples = 0

for file in ['therapeutic_high_quality.json', 'educational_high_quality.json', 
             'empathetic_high_quality.json', 'practical_high_quality.json']:
    file_path = os.path.join(data_dir, file)
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f'{file}: {len(data)} examples')
        total_examples += len(data)
    else:
        print(f'❌ Missing: {file}')

print(f'Total training examples: {total_examples}')
"
fi

# Set up Weights & Biases (optional)
echo "📈 Setting up Weights & Biases..."
echo "To use W&B logging, run: wandb login"
echo "Or set WANDB_MODE=offline for offline logging"

# Create launch script
echo "🎯 Creating launch script..."
cat > /teamspace/studios/this_studio/launch_training.sh << 'EOF'
#!/bin/bash

# Launch MoE training
echo "🚀 Launching MoE training on H100..."

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export WANDB_PROJECT="pixelated-empathy-moe"

# Optional: Set W&B to offline mode if no internet
# export WANDB_MODE=offline

# Run training
cd /teamspace/studios/this_studio
python -m moe_training.lightning_train

echo "✅ Training complete!"
EOF

chmod +x /teamspace/studios/this_studio/launch_training.sh

# Create monitoring script
echo "📊 Creating monitoring script..."
cat > /teamspace/studios/this_studio/monitor_training.py << 'EOF'
#!/usr/bin/env python3
"""Monitor training progress and GPU usage."""

import time
import psutil
import torch
import json
from pathlib import Path

def monitor_training():
    """Monitor training progress."""
    print("🔍 Monitoring training progress...")
    print("Press Ctrl+C to stop monitoring\n")
    
    try:
        while True:
            # GPU info
            if torch.cuda.is_available():
                gpu_memory = torch.cuda.memory_allocated(0) / (1024**3)
                gpu_memory_max = torch.cuda.max_memory_allocated(0) / (1024**3)
                gpu_util = torch.cuda.utilization(0) if hasattr(torch.cuda, 'utilization') else 'N/A'
                
                print(f"GPU Memory: {gpu_memory:.1f}GB / {gpu_memory_max:.1f}GB")
                print(f"GPU Utilization: {gpu_util}%")
            
            # CPU and system info
            cpu_percent = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            
            print(f"CPU: {cpu_percent}%")
            print(f"RAM: {memory.percent}% ({memory.used / (1024**3):.1f}GB / {memory.total / (1024**3):.1f}GB)")
            
            # Check for training logs
            log_file = Path("/teamspace/studios/this_studio/training.log")
            if log_file.exists():
                # Get last few lines
                with open(log_file, 'r') as f:
                    lines = f.readlines()
                    if lines:
                        print(f"Latest log: {lines[-1].strip()}")
            
            print("-" * 50)
            time.sleep(30)  # Update every 30 seconds
            
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped")

if __name__ == "__main__":
    monitor_training()
EOF

chmod +x /teamspace/studios/this_studio/monitor_training.py

echo ""
echo "✅ Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Upload training data to /teamspace/studios/this_studio/moe_training_data/"
echo "2. (Optional) Run: wandb login"
echo "3. Start training: ./launch_training.sh"
echo "4. Monitor progress: python monitor_training.py"
echo ""
echo "📁 Important directories:"
echo "   - Models: /teamspace/studios/this_studio/models/"
echo "   - Logs: /teamspace/studios/this_studio/logs/"
echo "   - Data: /teamspace/studios/this_studio/moe_training_data/"
echo ""
echo "🎯 Ready for H100 training!"
