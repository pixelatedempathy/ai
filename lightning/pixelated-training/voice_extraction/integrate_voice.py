#!/usr/bin/env python3
"""
Voice Extraction Integration Script
Integrates MoE voice extraction into pixelated training setup
"""

import json
import toml
from pathlib import Path

def update_pyproject_toml():
    """Add voice extraction dependencies to pyproject.toml"""
    pyproject_path = Path(__file__).parent.parent / "pyproject.toml"
    
    if pyproject_path.exists():
        with open(pyproject_path, 'r') as f:
            data = toml.load(f)
    else:
        data = {
            "project": {
                "name": "pixelated-training",
                "version": "0.1.0",
                "dependencies": []
            }
        }
    
    # Add voice extraction dependencies
    new_deps = [
        "scikit-learn>=1.3.0",
        "nltk>=3.8",
        "textstat>=0.7.0",
        "pyphen>=0.14.0"
    ]
    
    if "dependencies" not in data["project"]:
        data["project"]["dependencies"] = []
    
    for dep in new_deps:
        if not any(dep.split(">=")[0] in existing for existing in data["project"]["dependencies"]):
            data["project"]["dependencies"].append(dep)
    
    with open(pyproject_path, 'w') as f:
        toml.dump(data, f)
    
    print("Adding voice extraction dependencies to pyproject.toml...")

def update_training_config():
    """Update training configuration with voice extraction settings"""
    config_path = Path(__file__).parent.parent / "training_config.json"
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = json.load(f)
    else:
        config = {}
    
    # Add voice extraction configuration
    voice_config = {
        "voice_extraction": {
            "enabled": True,
            "moe_experts": {
                "therapeutic": {
                    "weight": 0.3,
                    "description": "Trauma-focused, healing-oriented responses"
                },
                "educational": {
                    "weight": 0.25,
                    "description": "Explanatory, concept-teaching mode"
                },
                "empathetic": {
                    "weight": 0.25,
                    "description": "Understanding, supportive responses"
                },
                "practical": {
                    "weight": 0.2,
                    "description": "Action-oriented, tool-providing advice"
                }
            },
            "routing_strategy": "context_aware",
            "blend_threshold": 0.7
        }
    }
    
    config.update(voice_config)
    
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print("Updated training_config.json with voice extraction settings")

def create_enhanced_training_script():
    """Create enhanced training script with voice integration"""
    script_path = Path(__file__).parent.parent / "train_with_voice.py"
    with open(script_path, 'w') as f:
        f.write('''#!/usr/bin/env python3
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
    
    print("\\n🧪 Testing Voice Integration...")
    for test_input in test_inputs:
        print(f"\\nInput: {test_input}")
        response = model.generate_response(test_input)
        print(f"Response: {response}")
    
    print("\\n✅ Voice integration test complete!")

if __name__ == "__main__":
    main()
''')
    
    print("Created train_with_voice.py - Enhanced training script with voice integration")

def main():
    print("🎭 Integrating MoE Voice Extraction System...")
    
    # Create voice_extraction directory structure
    voice_dir = Path(__file__).parent
    voice_dir.mkdir(exist_ok=True)
    (voice_dir / "experts").mkdir(exist_ok=True)
    
    # Update project configuration
    update_pyproject_toml()
    update_training_config()
    create_enhanced_training_script()
    
    print("\nVoice extraction integration complete!")
    print("\nNext steps:")
    print("1. Run: uv sync  # Install new dependencies")
    print("2. Run: python voice_extraction/style_analyzer.py  # Analyze transcripts")
    print("3. Run: python voice_extraction/moe_trainer.py     # Train experts")
    print("4. Run: python train_with_voice.py                 # Train with voice integration")

if __name__ == "__main__":
    main()
