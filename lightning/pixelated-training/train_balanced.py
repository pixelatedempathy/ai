#!/usr/bin/env python3
"""
Balanced training - good loss, reasonable speed
"""

import json
import torch
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from transformers import BitsAndBytesConfig, DataCollatorForLanguageModeling
from transformers.trainer_callback import TrainerCallback
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset
from tqdm import tqdm

class WandBCallback(TrainerCallback):
    def __init__(self):
        self.step_count = 0
        
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if logs:
            self.step_count += 1
            current_loss = logs.get('train_loss', 0)
            
            if 'epoch' in logs:
                progress = (logs['epoch'] / args.num_train_epochs) * 100
                print(f"📊 Progress: {progress:.1f}% | Loss: {current_loss:.4f} | Step: {self.step_count}")
            
            enhanced_logs = logs.copy()
            enhanced_logs.update({
                'training/steps_completed': self.step_count,
                'training/progress_percent': (logs.get('epoch', 0) / args.num_train_epochs) * 100 if args.num_train_epochs else 0
            })
            
            wandb.log(enhanced_logs, step=state.global_step)

def setup_wandb():
    with open('wandb_config.json', 'r') as f:
        config = json.load(f)
    
    return wandb.init(
        project=config['project'],
        name=f"{config['name']}-balanced",
        tags=config['tags'] + ['balanced'],
        notes="Balanced training for good loss and speed",
        config=config['config']
    )

def main():
    print("⚖️ Balanced Training")
    
    try:
        wandb_run = setup_wandb()
        
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
        
        # 4-bit quantization
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        MODEL_NAME = "LatitudeGames/Wayfarer-2-12B"
        
        print("🚀 Loading model...")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Align tokenizer with model config
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.eos_token_id = tokenizer.eos_token_id
        
        model = prepare_model_for_kbit_training(model)
        
        # Disable cache for gradient checkpointing compatibility
        model.config.use_cache = False
        
        # Balanced LoRA config
        lora_config = LoraConfig(
            r=32,
            lora_alpha=64,
            target_modules=["q_proj", "v_proj", "o_proj", "gate_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)
        
        def tokenize_function(examples):
            result = tokenizer(
                examples["text"], 
                truncation=True, 
                padding="max_length",
                max_length=768
            )
            result["labels"] = result["input_ids"].copy()
            return result
        
        print("🔤 Tokenizing...")
        tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # Balanced training args
        training_args = TrainingArguments(
            output_dir="./wayfarer-balanced",
            per_device_train_batch_size=2,
            gradient_accumulation_steps=16,
            num_train_epochs=3,
            learning_rate=8e-5,
            weight_decay=0.01,
            warmup_steps=200,
            lr_scheduler_type="cosine",
            bf16=True,
            logging_steps=25,
            save_steps=50,
            report_to="wandb",
            gradient_checkpointing=True,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            adam_beta1=0.9,
            adam_beta2=0.95,
            max_grad_norm=1.0,
            dataloader_num_workers=0
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            processing_class=tokenizer,
            data_collator=data_collator,
            callbacks=[WandBCallback()]
        )
        
        wandb.log({
            'training/status': 'started',
            'training/total_parameters': sum(p.numel() for p in model.parameters()),
            'training/trainable_parameters': sum(p.numel() for p in model.parameters() if p.requires_grad)
        })
        
        print("🎯 Training...")
        trainer.train()
        
        wandb.log({'training/status': 'completed'})
        
        print("💾 Saving...")
        trainer.save_model()
        
        print("✅ Training completed!")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
        raise
    finally:
        wandb.finish()
        print("🧹 Done")

if __name__ == "__main__":
    main()