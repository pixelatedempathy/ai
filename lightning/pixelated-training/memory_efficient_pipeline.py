#!/usr/bin/env python3
"""
Memory-efficient enhanced dataset pipeline
"""
import json
import os
from pathlib import Path

def process_in_chunks():
    """Process dataset in memory-efficient chunks"""
    input_file = "training_dataset.json"
    output_file = "training_dataset_enhanced.json"
    
    if not os.path.exists(input_file):
        print(f"Dataset file {input_file} not found")
        return
    
    print("Processing dataset in memory-efficient mode...")
    
    # Get file size
    file_size = os.path.getsize(input_file)
    print(f"Input file size: {file_size / (1024*1024):.1f} MB")
    
    # Read and process in one go but with minimal processing
    with open(input_file, 'r') as f:
        print("Loading conversations...")
        conversations = json.load(f)
    
    total_count = len(conversations)
    print(f"Total conversations: {total_count}")
    
    # Simple quality filter - keep conversations with therapeutic keywords
    therapeutic_keywords = [
        'therapy', 'counseling', 'mental health', 'anxiety', 'depression',
        'stress', 'support', 'help', 'feelings', 'emotions', 'coping'
    ]
    
    filtered_conversations = []
    
    for i, conv in enumerate(conversations):
        if i % 10000 == 0:
            print(f"Processed {i}/{total_count} conversations")
        
        # Check if conversation has therapeutic content
        messages = conv.get('messages', [])
        if not messages:
            continue
            
        text_content = ' '.join([msg.get('content', '') for msg in messages]).lower()
        
        # Must contain at least one therapeutic keyword
        if any(keyword in text_content for keyword in therapeutic_keywords):
            # Add minimal metadata
            conv['enhanced'] = True
            conv['conversation_id'] = f"pixelated_{len(filtered_conversations):06d}"
            filtered_conversations.append(conv)
    
    print(f"Filtered to {len(filtered_conversations)} therapeutic conversations")
    
    # Limit to target size if needed
    target_size = 40000
    if len(filtered_conversations) > target_size:
        filtered_conversations = filtered_conversations[:target_size]
        print(f"Limited to {target_size} conversations")
    
    # Save enhanced dataset
    print("Saving enhanced dataset...")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(filtered_conversations, f, separators=(',', ':'))
    
    print(f"Enhanced dataset saved: {output_file}")
    print(f"Final count: {len(filtered_conversations)} conversations")
    
    # Generate simple report
    report = {
        'original_count': total_count,
        'enhanced_count': len(filtered_conversations),
        'filter_ratio': len(filtered_conversations) / total_count if total_count > 0 else 0
    }
    
    with open("enhancement_report.json", 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"Enhancement ratio: {report['filter_ratio']:.2%}")

if __name__ == "__main__":
    process_in_chunks()
