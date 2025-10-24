#!/usr/bin/env python3
"""
Enhanced Training Script with Tim Fletcher Voice Integration
Combines base model training with MoE voice extraction
"""

import json
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

class VoiceIntegratedModel:
    def __init__(self, base_model_name="microsoft/DialoGPT-medium"):
        self.base_model_name = base_model_name
        self.tokenizer = AutoTokenizer.from_pretrained(base_model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
        self.experts = {}
        self.load_experts()
    
    def load_experts(self):
        """Load trained LoRA experts"""
        experts_dir = Path(__file__).parent / "voice_extraction/experts"
        
        if not experts_dir.exists():
            print("No experts found. Train experts first with moe_trainer.py")
            return
        
        for expert_dir in experts_dir.iterdir():
            if expert_dir.is_dir():
                style_name = expert_dir.name.replace("_expert", "")
                try:
                    expert = PeftModel.from_pretrained(self.base_model, expert_dir)
                    self.experts[style_name] = expert
                    print(f"Loaded {style_name} expert")
                except Exception as e:
                    print(f"Error loading {style_name} expert: {e}")
    
    def route_to_expert(self, input_text):
        """Determine which expert to use based on input"""
        style_keywords = {
            'therapeutic': ['trauma', 'healing', 'hurt', 'pain', 'recovery'],
            'educational': ['explain', 'understand', 'learn', 'concept'],
            'empathetic': ['feel', 'difficult', 'struggle', 'support'],
            'practical': ['do', 'action', 'step', 'practice', 'tool']
        }
        
        input_lower = input_text.lower()
        scores = {}
        
        for style, keywords in style_keywords.items():
            score = sum(1 for keyword in keywords if keyword in input_lower)
            scores[style] = score
        
        max_score = max(scores.values()) if scores.values() else 0
        if max_score == 0:
            return 'therapeutic'
        
        top_styles = [style for style, score in scores.items() if score == max_score]
        return top_styles[0] if len(top_styles) == 1 else 'therapeutic'
    
    def generate_response(self, input_text, max_length=200):
        """Generate response using appropriate expert"""
        if not self.experts:
            inputs = self.tokenizer.encode(input_text, return_tensors='pt')
            with torch.no_grad():
                outputs = self.base_model.generate(
                    inputs,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        selected_style = self.route_to_expert(input_text)
        expert = self.experts.get(selected_style, list(self.experts.values())[0])
        
        inputs = self.tokenizer.encode(input_text, return_tensors='pt')
        with torch.no_grad():
            outputs = expert.generate(
                inputs,
                max_length=max_length,
                num_return_sequences=1,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"Used {selected_style} expert for response")
        return response

def main():
    print("🎭 Initializing Voice-Integrated Training...")
    
    config_path = Path(__file__).parent / "training_config.json"
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
        print("Loaded training configuration")
    else:
        print("No training config found, using defaults")
        config = {}
    
    model = VoiceIntegratedModel()
    
    test_inputs = [
        "I'm struggling with childhood trauma",
        "Can you explain what complex PTSD is?",
        "I feel so alone in this",
        "What steps can I take to heal?"
    ]
    
    print("\n🧪 Testing Voice Integration...")
    for test_input in test_inputs:
        print(f"\nInput: {test_input}")
        response = model.generate_response(test_input)
        print(f"Response: {response}")
    
    print("\n✅ Voice integration test complete!")

if __name__ == "__main__":
    main()
