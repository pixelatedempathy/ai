#!/usr/bin/env python3
"""
Mixture of Experts (MoE) Trainer for Tim Fletcher Voice Styles
Trains separate LoRA experts for each communication style
"""

import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class StyleDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': encoding['input_ids'].flatten()
        }

class MoETrainer:
    def __init__(self, model_name="microsoft/DialoGPT-medium"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.base_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.experts = {}
        
    def create_lora_config(self, style_name):
        """Create LoRA configuration for a specific style"""
        return LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["c_attn", "c_proj"],  # For GPT-2 based models
            bias="none"
        )
    
    def load_style_data(self, style_files):
        """Load and prepare training data for a style"""
        texts = []
        for file_path in style_files:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    # Split into chunks for training
                    chunks = self.split_into_chunks(content)
                    texts.extend(chunks)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
        return texts
    
    def split_into_chunks(self, text, chunk_size=300):
        """Split text into training chunks"""
        sentences = text.split('. ')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            if len(current_chunk) + len(sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def train_expert(self, style_name, style_files, epochs=3):
        """Train a LoRA expert for a specific style"""
        print(f"\nTraining {style_name} expert...")
        
        # Load data
        texts = self.load_style_data(style_files)
        if not texts:
            print(f"No data found for {style_name}")
            return None
        
        print(f"Training on {len(texts)} text chunks")
        
        # Create LoRA model
        lora_config = self.create_lora_config(style_name)
        model = get_peft_model(self.base_model, lora_config)
        
        # Prepare dataset
        dataset = StyleDataset(texts, self.tokenizer)
        dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
        
        # Training setup
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        model.train()
        
        # Training loop
        for epoch in range(epochs):
            total_loss = 0
            for batch_idx, batch in enumerate(dataloader):
                optimizer.zero_grad()
                
                outputs = model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch['labels']
                )
                
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
                if batch_idx % 10 == 0:
                    print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}, Loss: {loss.item():.4f}")
            
            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch+1} completed. Average loss: {avg_loss:.4f}")
        
        # Save expert
        expert_path = Path(__file__).parent / f"experts/{style_name}_expert"
        expert_path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(expert_path)
        
        print(f"Saved {style_name} expert to {expert_path}")
        return model
    
    def train_all_experts(self, style_analysis_file):
        """Train experts for all styles"""
        with open(style_analysis_file, 'r') as f:
            style_data = json.load(f)
        
        for style_name, file_list in style_data.items():
            if file_list:  # Only train if we have data
                expert = self.train_expert(style_name, file_list)
                if expert:
                    self.experts[style_name] = expert
        
        print(f"\nTraining complete! Trained {len(self.experts)} experts.")

def main():
    trainer = MoETrainer()
    
    # Look for style analysis file
    analysis_file = Path(__file__).parent / "style_analysis.json"
    if not analysis_file.exists():
        print("Style analysis file not found. Run style_analyzer.py first.")
        return
    
    trainer.train_all_experts(analysis_file)

if __name__ == "__main__":
    main()
