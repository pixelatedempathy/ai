#!/usr/bin/env python3
"""
Complete Post-Training Pipeline
Export GGUF + Upload to HuggingFace
"""

import os
import sys
import subprocess
from pathlib import Path

def run_complete_pipeline(model_path="./wayfarer-finetuned"):
    """Run complete post-training pipeline"""
    
    print("🚀 Starting Complete Post-Training Pipeline")
    print("=" * 60)
    
    # Step 1: Export to GGUF
    print("📦 Step 1: Exporting to GGUF formats...")
    gguf_result = subprocess.run([
        sys.executable, "export_gguf.py", model_path, "./gguf-exports"
    ], capture_output=True, text=True)
    
    if gguf_result.returncode == 0:
        print("✅ GGUF export completed")
    else:
        print(f"❌ GGUF export failed: {gguf_result.stderr}")
        return False
    
    # Step 2: Upload to HuggingFace
    print("\n📤 Step 2: Uploading to HuggingFace...")
    hf_result = subprocess.run([
        sys.executable, "upload_hf.py", model_path, "./gguf-exports", "Wayfarer2-Pixelated"
    ], capture_output=True, text=True)
    
    if hf_result.returncode == 0:
        print("✅ HuggingFace upload completed")
    else:
        print(f"❌ HuggingFace upload failed: {hf_result.stderr}")
        return False
    
    print("\n🎯 Complete pipeline finished successfully!")
    print("📊 Results:")
    print("   • Model exported to multiple GGUF formats")
    print("   • All variants uploaded to HuggingFace")
    print("   • Model card and documentation included")
    print("   • Ready for public use")
    
    return True

def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "./wayfarer-finetuned"
    
    if not os.path.exists(model_path):
        print(f"❌ Model path not found: {model_path}")
        print("Make sure training completed successfully")
        return False
    
    success = run_complete_pipeline(model_path)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
