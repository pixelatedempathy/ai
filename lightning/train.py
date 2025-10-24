#!/usr/bin/env python3
"""
Wayfarer Training - Fixed Version
"""

import json
import torch
import wandb
import os
import signal
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from datasets import Dataset
from tqdm import tqdm

# Global shutdown flag
shutdown_requested = False

def signal_handler(signum, frame):
    global shutdown_requested
    print(f"\n🛑 Shutdown requested")
    shutdown_requested = True

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

class WandBCallback:
    def __init__(self, safety_config):
        self.safety_config = safety_config
        self.step_count = 0
        
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        global shutdown_requested
        
        if shutdown_requested:
            print("🛑 Stopping training")
            control.should_training_stop = True
            return
        
        if logs:
            self.step_count += 1
            current_loss = logs.get('train_loss', 0)
            
            if 'epoch' in logs:
                progress = (logs['epoch'] / args.num_train_epochs) * 100
                print(f"📊 Progress: {progress:.1f}% | Loss: {current_loss:.4f} | Step: {self.step_count}")
            
            enhanced_logs = logs.copy()
            enhanced_logs.update({
                'training/steps_completed': self.step_count,
                'system/shutdown_requested': shutdown_requested
            })
            
            wandb.log(enhanced_logs, step=state.global_step)

def setup_wandb():
    with open('wandb_config.json', 'r') as f:
        config = json.load(f)
    
    if not torch.cuda.is_available():
        print("⚠️ No CUDA - using offline mode")
        os.environ['WANDB_MODE'] = 'offline'
    
    run = wandb.init(
        project=config['project'],
        entity=config.get('entity'),
        name=config['name'],
        tags=config['tags'],
        notes=config['notes'],
        config=config['config']
    )
    
    return run

def main():
    global shutdown_requested
    
    print("🚀 Wayfarer Training")
    
    wandb_run = None
    
    try:
        # Setup WandB
        wandb_run = setup_wandb()
        
        # Load configs
        print("📋 Loading configurations...")
        with open('training_config.json', 'r') as f:
            training_config = json.load(f)
        
        with open('safety_config.json', 'r') as f:
            safety_config = json.load(f)
        
        # Load dataset
        print("📊 Loading dataset...")
        with open('training_dataset.json', 'r') as f:
            data = json.load(f)
        
        texts = [conv['text'] for conv in tqdm(data['conversations'], desc="Loading")]
        dataset = Dataset.from_dict({"text": texts})
        
        print(f"📊 Dataset: {len(dataset)} samples")
        
        wandb.log({
            'dataset/total_conversations': len(texts),
            'dataset/avg_length': sum(len(text.split()) for text in texts) / len(texts)
        })
        
        # Setup model
        MODEL_NAME = "LatitudeGames/Wayfarer-2-12B"
        device_available = torch.cuda.is_available()
        
        if device_available:
            print("🚀 Loading model...")
            tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.bfloat16,
                device_map="auto"
            )
        else:
            print("⚠️ CPU mode")
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            model = None
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Tokenize
        def tokenize_function(examples):
            result = tokenizer(
                examples["text"], 
                truncation=True, 
                padding="max_length",
                max_length=2048
            )
            result["labels"] = result["input_ids"].copy()
            return result
        
        print("🔤 Tokenizing...")
        tokenized_dataset = dataset.map(
            tokenize_function, 
            batched=True, 
            remove_columns=["text"]
        )
        
        print(f"📊 Tokenized: {len(tokenized_dataset)} samples")
        
        if len(tokenized_dataset) == 0:
            raise ValueError("Empty dataset!")
        
        # Training args
        config_copy = training_config.copy()
        config_copy.pop('report_to', None)
        
        if device_available:
            training_args = TrainingArguments(
                report_to="wandb",
                run_name=wandb_run.name,
                **config_copy
            )
        else:
            config_copy['bf16'] = False
            config_copy['report_to'] = 'none'
            training_args = TrainingArguments(**config_copy)
        
        # Trainer
        if model is not None:
            trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=tokenized_dataset,
                tokenizer=tokenizer,
                callbacks=[WandBCallback(safety_config)]
            )
        else:
            trainer = None
        
        # Train
        if trainer is not None and device_available and not shutdown_requested:
            wandb.log({
                'training/status': 'started',
                'training/total_parameters': sum(p.numel() for p in model.parameters())
            })
            
            print("🎯 Training...")
            trainer.train()
            
            print("💾 Saving...")
            trainer.save_model()
            tokenizer.save_pretrained(training_args.output_dir)
            
            wandb.log({'training/status': 'completed'})
            print("✅ Training completed!")
            
        else:
            print("⚠️ Testing mode")
    
    except KeyboardInterrupt:
        print("\n🛑 Interrupted")
    except Exception as e:
        print(f"❌ Failed: {e}")
        if wandb_run:
            try:
                wandb.log({'training/status': 'failed', 'training/error': str(e)})
            except:
                pass
        raise
    finally:
        if wandb_run:
            try:
                wandb.finish()
            except:
                pass
        print("🧹 Done")

if __name__ == "__main__":
    main()