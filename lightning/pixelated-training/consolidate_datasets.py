#!/usr/bin/env python3
"""
Consolidate all downloaded datasets into training_dataset.json
Integrates professional datasets, CoT reasoning, and priority conversations
"""
import json
import os
from pathlib import Path

def load_jsonl(file_path):
    """Load JSONL file"""
    conversations = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                conversations.append(json.loads(line))
    return conversations

def load_json(file_path):
    """Load JSON file"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def main():
    processed_dir = Path("processed")
    all_conversations = []
    
    # Priority conversations (Phase 1)
    priority_files = [
        "phase_1_priority_conversations/task_5_1_priority_1/priority_1_conversations.jsonl",
        "phase_1_priority_conversations/task_5_2_priority_2/priority_2_conversations.jsonl", 
        "phase_1_priority_conversations/task_5_3_priority_3/priority_3_conversations.jsonl",
        "phase_1_priority_conversations/task_5_6_unified_priority/unified_priority_conversations.jsonl"
    ]
    
    # Professional datasets (Phase 2)
    professional_files = [
        "phase_2_professional_datasets/task_5_9_soulchat/soulchat_2_0_conversations.jsonl",
        "phase_2_professional_datasets/task_5_10_counsel_chat/counsel_chat_conversations.jsonl",
        "phase_2_professional_datasets/task_5_11_llama3_mental_counseling/llama3_mental_counseling_conversations.jsonl",
        "phase_2_professional_datasets/task_5_12_therapist_sft/therapist_sft_conversations.jsonl",
        "phase_2_professional_datasets/task_5_13_neuro_qa_sft/neuro_qa_sft_conversations.jsonl"
    ]
    
    # CoT reasoning (Phase 3)
    cot_files = [
        "phase_3_cot_reasoning/task_5_15_cot_reasoning/cot_reasoning_conversations_consolidated.jsonl"
    ]
    
    # Natural conversations
    natural_files = [
        "natural_conversations/natural_multi_turn_conversations.json"
    ]
    
    # Load all datasets
    for file_list, phase_name in [
        (priority_files, "Priority"),
        (professional_files, "Professional"), 
        (cot_files, "CoT Reasoning"),
        (natural_files, "Natural")
    ]:
        for file_path in file_list:
            full_path = processed_dir / file_path
            if full_path.exists():
                print(f"Loading {phase_name}: {file_path}")
                if file_path.endswith('.jsonl'):
                    conversations = load_jsonl(full_path)
                else:
                    conversations = load_json(full_path)
                    if isinstance(conversations, dict) and 'conversations' in conversations:
                        conversations = conversations['conversations']
                
                all_conversations.extend(conversations)
                print(f"  Added {len(conversations)} conversations")
            else:
                print(f"  Missing: {file_path}")
    
    print(f"\nTotal conversations: {len(all_conversations)}")
    
    # Save consolidated dataset
    with open("training_dataset.json", 'w', encoding='utf-8') as f:
        json.dump(all_conversations, f, indent=2, ensure_ascii=False)
    
    print(f"Saved training_dataset.json ({len(all_conversations)} conversations)")

if __name__ == "__main__":
    main()
