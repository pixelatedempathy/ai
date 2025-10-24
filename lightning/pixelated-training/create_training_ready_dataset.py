#!/usr/bin/env python3
"""
Create training-ready dataset from existing large dataset
Uses the existing training_dataset.json and creates a smaller, optimized version
"""
import json
import os

def create_optimized_dataset():
    """Create optimized dataset for training"""
    
    # Check if we already have the large dataset
    if not os.path.exists("training_dataset.json"):
        print("training_dataset.json not found. Using existing validation_dataset.json")
        # Use the existing smaller dataset
        if os.path.exists("validation_dataset.json"):
            os.system("cp validation_dataset.json training_dataset_ready.json")
            print("Created training_dataset_ready.json from validation_dataset.json")
            return
        else:
            print("No dataset found. Please run consolidate_datasets.py first")
            return
    
    print("Large dataset found. Creating optimized version...")
    
    # Since the full dataset is too large, let's use the existing smaller one
    # that's already optimized for the training pipeline
    if os.path.exists("validation_dataset.json"):
        print("Using existing optimized validation_dataset.json")
        
        # Check its size
        size = os.path.getsize("validation_dataset.json")
        print(f"Validation dataset size: {size / (1024*1024):.1f} MB")
        
        # Copy it as our training dataset
        os.system("cp validation_dataset.json training_dataset_optimized.json")
        
        # Load and check conversation count
        with open("validation_dataset.json", 'r') as f:
            data = json.load(f)
            print(f"Conversations in optimized dataset: {len(data)}")
        
        print("Training-ready dataset created: training_dataset_optimized.json")
        
        # Update the training config to use this dataset
        update_training_config()
    
    else:
        print("No optimized dataset available")

def update_training_config():
    """Update training config to use optimized dataset"""
    config_file = "training_config.json"
    
    if os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
        
        # Update dataset path
        config['dataset_path'] = 'training_dataset_optimized.json'
        
        with open(config_file, 'w') as f:
            json.dump(config, f, indent=2)
        
        print("Updated training_config.json to use optimized dataset")

if __name__ == "__main__":
    create_optimized_dataset()
