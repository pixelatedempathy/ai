#!/usr/bin/env python3
"""
Dry Run Test - Validate Training Pipeline Without GPU
"""

import json
import os
import sys
from pathlib import Path

def test_file_integrity():
    """Test all required files exist and are valid"""
    print("🔍 Testing file integrity...")
    
    required_files = [
        'training_dataset.json',
        'train.py',
        'training_config.json',
        'wandb_config.json',
        'safety_config.json'
    ]
    
    missing_files = []
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    if missing_files:
        print(f"❌ Missing files: {missing_files}")
        return False
    
    # Test JSON files are valid
    json_files = [f for f in required_files if f.endswith('.json')]
    for file in json_files:
        try:
            with open(file, 'r') as f:
                json.load(f)
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON in {file}: {e}")
            return False
    
    print("✅ All files present and valid")
    return True

def test_dataset_format():
    """Test dataset format and structure"""
    print("🔍 Testing dataset format...")
    
    try:
        with open('training_dataset.json', 'r') as f:
            dataset = json.load(f)
        
        # Check structure
        if 'conversations' not in dataset:
            print("❌ Dataset missing 'conversations' key")
            return False
        
        conversations = dataset['conversations']
        if not isinstance(conversations, list) or len(conversations) == 0:
            print("❌ No conversations found")
            return False
        
        # Test first conversation
        conv = conversations[0]
        required_keys = ['conversation_id', 'messages', 'text']
        for key in required_keys:
            if key not in conv:
                print(f"❌ Conversation missing '{key}' key")
                return False
        
        # Test ChatML format
        if '<|im_start|>' not in conv['text'] or '<|im_end|>' not in conv['text']:
            print("❌ Invalid ChatML format")
            return False
        
        print(f"✅ Dataset valid: {len(conversations):,} conversations")
        return True
        
    except Exception as e:
        print(f"❌ Dataset test failed: {e}")
        return False

def test_training_script_syntax():
    """Test training script syntax without execution"""
    print("🔍 Testing training script syntax...")
    
    try:
        with open('train.py', 'r') as f:
            script_content = f.read()
        
        # Basic syntax check
        compile(script_content, 'train.py', 'exec')
        
        # Check for required imports
        required_imports = ['torch', 'transformers', 'wandb', 'datasets']
        for imp in required_imports:
            if f'import {imp}' not in script_content and f'from {imp}' not in script_content:
                print(f"❌ Missing import: {imp}")
                return False
        
        print("✅ Training script syntax valid")
        return True
        
    except SyntaxError as e:
        print(f"❌ Syntax error in training script: {e}")
        return False
    except Exception as e:
        print(f"❌ Training script test failed: {e}")
        return False

def test_config_compatibility():
    """Test configuration file compatibility"""
    print("🔍 Testing configuration compatibility...")
    
    try:
        # Load configs
        with open('training_config.json', 'r') as f:
            training_config = json.load(f)
        
        with open('wandb_config.json', 'r') as f:
            wandb_config = json.load(f)
        
        # Check required training config keys
        required_keys = ['learning_rate', 'num_train_epochs', 'per_device_train_batch_size']
        for key in required_keys:
            if key not in training_config:
                print(f"❌ Missing training config: {key}")
                return False
        
        # Check WandB config
        if 'project' not in wandb_config:
            print("❌ Missing WandB project config")
            return False
        
        print("✅ Configuration files compatible")
        return True
        
    except Exception as e:
        print(f"❌ Config test failed: {e}")
        return False

def test_import_dependencies():
    """Test if required dependencies can be imported"""
    print("🔍 Testing dependency imports...")
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('datasets', 'Datasets'),
        ('wandb', 'WandB'),
        ('numpy', 'NumPy')
    ]
    
    missing_packages = []
    for package, name in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(name)
    
    if missing_packages:
        print(f"❌ Missing packages: {missing_packages}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    print("✅ All dependencies available")
    return True

def create_mock_training_test():
    """Create a mock training test that runs without GPU"""
    
    mock_script = '''#!/usr/bin/env python3
"""
Mock Training Test - CPU Only
"""

import json
import torch
from datasets import Dataset

def mock_training_test():
    """Test training pipeline on CPU with small sample"""
    print("🧪 Running mock training test...")
    
    # Load small sample
    with open('training_dataset.json', 'r') as f:
        data = json.load(f)
    
    # Use only first 10 conversations
    sample_texts = [conv['text'] for conv in data['conversations'][:10]]
    dataset = Dataset.from_dict({"text": sample_texts})
    
    print(f"✅ Sample dataset created: {len(sample_texts)} conversations")
    
    # Test tokenization (without model loading)
    try:
        from transformers import AutoTokenizer
        
        # Use a small tokenizer for testing
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Test tokenization
        def tokenize_function(examples):
            return tokenizer(examples["text"], truncation=True, padding=True, max_length=512)
        
        tokenized_sample = dataset.map(tokenize_function, batched=True)
        print(f"✅ Tokenization test passed: {len(tokenized_sample)} samples")
        
        return True
        
    except Exception as e:
        print(f"❌ Mock training test failed: {e}")
        return False

if __name__ == "__main__":
    mock_training_test()
'''
    
    with open('mock_training_test.py', 'w') as f:
        f.write(mock_script)
    
    return 'mock_training_test.py'

def run_dry_run():
    """Run complete dry run validation"""
    print("🧪 Wayfarer Training Pipeline Dry Run")
    print("=" * 45)
    
    tests = [
        ("File Integrity", test_file_integrity),
        ("Dataset Format", test_dataset_format),
        ("Script Syntax", test_training_script_syntax),
        ("Config Compatibility", test_config_compatibility),
        ("Dependencies", test_import_dependencies)
    ]
    
    results = {}
    all_passed = True
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results[test_name] = result
            if not result:
                all_passed = False
        except Exception as e:
            print(f"❌ {test_name} test crashed: {e}")
            results[test_name] = False
            all_passed = False
        print()
    
    # Create mock training test
    mock_script = create_mock_training_test()
    print(f"✅ Created {mock_script} for CPU testing")
    
    # Summary
    print("📋 Dry Run Results:")
    print("=" * 25)
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
    
    print(f"\n🎯 Overall Result: {'✅ READY FOR TRAINING' if all_passed else '❌ ISSUES FOUND'}")
    
    if all_passed:
        print("🚀 Training pipeline validated - ready for GPU execution")
    else:
        print("🔧 Fix issues before proceeding to GPU training")
    
    return all_passed, results

if __name__ == "__main__":
    run_dry_run()
