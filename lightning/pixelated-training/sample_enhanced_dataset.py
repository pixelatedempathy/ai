#!/usr/bin/env python3
"""
Sample enhanced dataset from the large consolidated dataset
Creates a training-ready dataset with therapeutic focus
"""
import json
import random
import sys
from pathlib import Path

def sample_therapeutic_conversations(input_file, output_file, sample_size=40000):
    """Sample therapeutic conversations from large dataset"""
    
    print(f"Sampling {sample_size} conversations from {input_file}")
    
    # Read file line by line to avoid memory issues
    conversations = []
    therapeutic_keywords = [
        'therapy', 'counseling', 'mental health', 'anxiety', 'depression',
        'stress', 'support', 'help', 'feelings', 'emotions', 'coping',
        'therapist', 'counselor', 'wellbeing', 'mindfulness', 'self-care'
    ]
    
    try:
        # Try to read the JSON file
        print("Reading dataset...")
        with open(input_file, 'r', encoding='utf-8') as f:
            # Read in chunks to manage memory
            chunk_size = 1000
            all_conversations = []
            
            # Load the JSON (this might still be large)
            data = json.load(f)
            print(f"Loaded {len(data)} total conversations")
            
            # Process in chunks
            therapeutic_conversations = []
            
            for i in range(0, len(data), chunk_size):
                chunk = data[i:i+chunk_size]
                print(f"Processing chunk {i//chunk_size + 1}/{(len(data)-1)//chunk_size + 1}")
                
                for conv in chunk:
                    # Check if conversation has therapeutic content
                    messages = conv.get('messages', [])
                    if not messages:
                        continue
                    
                    # Get text content
                    text_content = ' '.join([
                        msg.get('content', '') for msg in messages
                    ]).lower()
                    
                    # Check for therapeutic keywords
                    if any(keyword in text_content for keyword in therapeutic_keywords):
                        # Add metadata
                        conv['therapeutic_enhanced'] = True
                        conv['sample_id'] = len(therapeutic_conversations)
                        therapeutic_conversations.append(conv)
                        
                        # Stop if we have enough
                        if len(therapeutic_conversations) >= sample_size:
                            break
                
                if len(therapeutic_conversations) >= sample_size:
                    break
            
            print(f"Found {len(therapeutic_conversations)} therapeutic conversations")
            
            # Shuffle and take sample
            random.shuffle(therapeutic_conversations)
            final_sample = therapeutic_conversations[:sample_size]
            
            # Save the sample
            print(f"Saving {len(final_sample)} conversations to {output_file}")
            with open(output_file, 'w', encoding='utf-8') as out_f:
                json.dump(final_sample, out_f, separators=(',', ':'))
            
            print(f"Enhanced dataset created: {output_file}")
            print(f"Sample size: {len(final_sample)} conversations")
            
            return len(final_sample)
            
    except MemoryError:
        print("Memory error - dataset too large. Using alternative approach...")
        return create_smaller_sample(input_file, output_file, 10000)
    
    except Exception as e:
        print(f"Error processing dataset: {e}")
        return 0

def create_smaller_sample(input_file, output_file, sample_size):
    """Create smaller sample if memory is limited"""
    print(f"Creating smaller sample of {sample_size} conversations")
    
    # Use existing smaller datasets from processed folder
    sample_conversations = []
    
    # Load from smaller files
    smaller_files = [
        "processed/phase_2_professional_datasets/task_5_10_counsel_chat/counsel_chat_conversations.jsonl",
        "processed/phase_2_professional_datasets/task_5_11_llama3_mental_counseling/llama3_mental_counseling_conversations.jsonl",
        "processed/phase_2_professional_datasets/task_5_12_therapist_sft/therapist_sft_conversations.jsonl"
    ]
    
    for file_path in smaller_files:
        if Path(file_path).exists():
            print(f"Loading from {file_path}")
            with open(file_path, 'r') as f:
                for line_num, line in enumerate(f):
                    if line.strip():
                        try:
                            conv = json.loads(line)
                            conv['source_file'] = file_path
                            conv['sample_id'] = len(sample_conversations)
                            sample_conversations.append(conv)
                            
                            if len(sample_conversations) >= sample_size:
                                break
                        except json.JSONDecodeError:
                            continue
            
            if len(sample_conversations) >= sample_size:
                break
    
    # Save sample
    if sample_conversations:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(sample_conversations, f, separators=(',', ':'))
        
        print(f"Created sample dataset: {output_file}")
        print(f"Sample size: {len(sample_conversations)} conversations")
        
        return len(sample_conversations)
    
    return 0

def main():
    input_file = "training_dataset.json"
    output_file = "training_dataset_enhanced.json"
    
    if not Path(input_file).exists():
        print(f"Input file {input_file} not found")
        return
    
    # Try to sample from the large dataset
    sample_count = sample_therapeutic_conversations(input_file, output_file, 40000)
    
    if sample_count > 0:
        print(f"\nSuccess! Created enhanced dataset with {sample_count} conversations")
        
        # Update training config
        config_file = "training_config.json"
        if Path(config_file).exists():
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            config['dataset_path'] = output_file
            
            with open(config_file, 'w') as f:
                json.dump(config, f, indent=2)
            
            print(f"Updated {config_file} to use enhanced dataset")
    else:
        print("Failed to create enhanced dataset")

if __name__ == "__main__":
    main()
