#!/usr/bin/env python3
"""
HuggingFace Upload Script - Model + GGUF Variants
"""

import os
import json
from pathlib import Path
from huggingface_hub import HfApi, create_repo, upload_folder, upload_file

def upload_to_huggingface(model_path, gguf_path, model_name="Wayfarer2-Pixelated"):
    """Upload model and GGUF variants to HuggingFace"""
    
    print(f"🚀 Uploading {model_name} to HuggingFace...")
    
    # Initialize HF API
    api = HfApi()
    
    # Get username from token
    try:
        user_info = api.whoami()
        username = user_info['name']
        repo_id = f"{username}/{model_name}"
        print(f"📦 Repository: {repo_id}")
    except Exception as e:
        print(f"❌ HuggingFace authentication failed: {e}")
        print("Make sure HF_TOKEN is set or run 'huggingface-cli login'")
        return False
    
    # Create repository
    try:
        create_repo(
            repo_id=repo_id,
            exist_ok=True,
            repo_type="model",
            private=False
        )
        print(f"✅ Repository created/verified: {repo_id}")
    except Exception as e:
        print(f"❌ Repository creation failed: {e}")
        return False
    
    # Upload main model
    if os.path.exists(model_path):
        print(f"📤 Uploading main model from {model_path}...")
        try:
            upload_folder(
                folder_path=model_path,
                repo_id=repo_id,
                repo_type="model",
                commit_message="Upload fine-tuned Wayfarer2-Pixelated model"
            )
            print("✅ Main model uploaded successfully")
        except Exception as e:
            print(f"❌ Main model upload failed: {e}")
            return False
    else:
        print(f"⚠️ Model path not found: {model_path}")
    
    # Upload GGUF variants
    if os.path.exists(gguf_path):
        print(f"📤 Uploading GGUF variants from {gguf_path}...")
        
        gguf_files = list(Path(gguf_path).glob("*.gguf"))
        
        for gguf_file in gguf_files:
            print(f"📤 Uploading {gguf_file.name}...")
            try:
                upload_file(
                    path_or_fileobj=str(gguf_file),
                    path_in_repo=gguf_file.name,
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=f"Add GGUF variant: {gguf_file.name}"
                )
                print(f"✅ {gguf_file.name} uploaded")
            except Exception as e:
                print(f"❌ {gguf_file.name} upload failed: {e}")
    else:
        print(f"⚠️ GGUF path not found: {gguf_path}")
    
    # Upload model card
    model_card_path = "README.md"
    if os.path.exists(model_card_path):
        print("📤 Uploading model card...")
        try:
            upload_file(
                path_or_fileobj=model_card_path,
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="model",
                commit_message="Add comprehensive model card"
            )
            print("✅ Model card uploaded")
        except Exception as e:
            print(f"❌ Model card upload failed: {e}")
    
    print(f"🎯 Upload complete! Model available at: https://huggingface.co/{repo_id}")
    return True

def main():
    import sys
    
    model_path = sys.argv[1] if len(sys.argv) > 1 else "./wayfarer-finetuned"
    gguf_path = sys.argv[2] if len(sys.argv) > 2 else "./gguf-exports"
    model_name = sys.argv[3] if len(sys.argv) > 3 else "Wayfarer2-Pixelated"
    
    success = upload_to_huggingface(model_path, gguf_path, model_name)
    
    if success:
        print("✅ HuggingFace upload completed successfully")
    else:
        print("❌ HuggingFace upload failed")
    
    return success

if __name__ == "__main__":
    main()
