"""Data loading and preprocessing for MoE training."""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from typing import Dict, List, Tuple, Optional
import random
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class TherapeuticDataset(Dataset):
    """Dataset for therapeutic conversation training."""
    
    def __init__(
        self,
        data_file: str,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        style: str = "therapeutic",
        min_length: int = 50,
        augment_data: bool = True
    ):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.style = style
        self.min_length = min_length
        self.augment_data = augment_data
        
        # Load and preprocess data
        self.examples = self._load_and_preprocess(data_file)
        
        # Set up special tokens
        self._setup_special_tokens()
        
        logger.info(f"Loaded {len(self.examples)} examples for {style} expert")
    
    def _load_and_preprocess(self, data_file: str) -> List[Dict]:
        """Load and preprocess training examples."""
        with open(data_file, 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        processed_examples = []
        
        for item in raw_data:
            text = item['text'].strip()
            
            # Filter by length
            if len(text.split()) < self.min_length:
                continue
            
            # Quality filtering
            if item.get('quality_score', 0) < 0.5:
                continue
            
            # Create training example
            example = {
                'text': text,
                'style': item['style'],
                'confidence': item.get('confidence', 1.0),
                'quality_score': item.get('quality_score', 1.0),
                'indicators': item.get('indicators', []),
                'metadata': item.get('metadata', {})
            }
            
            processed_examples.append(example)
        
        # Data augmentation if enabled
        if self.augment_data and len(processed_examples) < 1000:
            processed_examples = self._augment_data(processed_examples)
        
        return processed_examples
    
    def _augment_data(self, examples: List[Dict]) -> List[Dict]:
        """Augment training data for smaller datasets."""
        augmented = examples.copy()
        
        # Simple augmentation: add variations with different contexts
        context_prefixes = [
            "In therapeutic work, ",
            "When working with clients, ",
            "From a therapeutic perspective, ",
            "In my experience with trauma recovery, ",
        ]
        
        for example in examples[:min(200, len(examples))]:  # Limit augmentation
            for prefix in context_prefixes:
                if not example['text'].startswith(prefix.strip()):
                    augmented_text = prefix + example['text'].lower()
                    augmented_example = example.copy()
                    augmented_example['text'] = augmented_text
                    augmented_example['augmented'] = True
                    augmented.append(augmented_example)
        
        logger.info(f"Augmented dataset from {len(examples)} to {len(augmented)} examples")
        return augmented
    
    def _setup_special_tokens(self):
        """Set up special tokens for conversation format."""
        # Add special tokens if not present
        special_tokens = {
            'pad_token': '<pad>',
            'eos_token': '<eos>',
            'bos_token': '<bos>',
            'unk_token': '<unk>'
        }
        
        tokens_to_add = []
        for token_type, token in special_tokens.items():
            if getattr(self.tokenizer, token_type) is None:
                tokens_to_add.append(token)
        
        if tokens_to_add:
            self.tokenizer.add_special_tokens({
                'additional_special_tokens': tokens_to_add
            })
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get training example."""
        example = self.examples[idx]
        
        # Format as conversation
        text = self._format_conversation(example['text'], example['style'])
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Prepare labels (same as input_ids for language modeling)
        labels = encoding['input_ids'].clone()
        
        # Mask padding tokens in labels
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': labels.squeeze(),
            'style': example['style'],
            'quality_score': torch.tensor(example['quality_score'], dtype=torch.float)
        }
    
    def _format_conversation(self, text: str, style: str) -> str:
        """Format text as therapeutic conversation."""
        # Create a conversational format that encourages the desired style
        style_prompts = {
            'therapeutic': "As a trauma-informed therapist, I would respond: ",
            'educational': "From a clinical perspective, let me explain: ",
            'empathetic': "I understand this is difficult. ",
            'practical': "Here are some practical steps you can take: "
        }
        
        prompt = style_prompts.get(style, "")
        formatted_text = f"{prompt}{text}"
        
        return formatted_text
    
    def get_style_distribution(self) -> Dict[str, int]:
        """Get distribution of styles in dataset."""
        distribution = {}
        for example in self.examples:
            style = example['style']
            distribution[style] = distribution.get(style, 0) + 1
        return distribution


class MoEDataLoader:
    """Data loader manager for MoE training."""
    
    def __init__(self, config, tokenizer: AutoTokenizer):
        self.config = config
        self.tokenizer = tokenizer
        self.datasets = {}
        self.dataloaders = {}
    
    def prepare_expert_data(self, expert_name: str) -> Tuple[DataLoader, DataLoader]:
        """Prepare training and validation data for specific expert."""
        expert_config = self.config.get_expert_config(expert_name)
        data_file = Path(self.config.system.data_dir) / expert_config.data_file
        
        # Create dataset
        full_dataset = TherapeuticDataset(
            data_file=str(data_file),
            tokenizer=self.tokenizer,
            max_length=self.config.model.max_length,
            style=expert_config.style,
            min_length=expert_config.min_length
        )
        
        # Split into train/val
        train_size = int(expert_config.train_split * len(full_dataset))
        val_size = len(full_dataset) - train_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, val_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=True,
            num_workers=self.config.training.dataloader_num_workers,
            pin_memory=True,
            drop_last=True
        )
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.training.batch_size,
            shuffle=False,
            num_workers=self.config.training.dataloader_num_workers,
            pin_memory=True,
            drop_last=False
        )
        
        # Store for later use
        self.datasets[expert_name] = {
            'train': train_dataset,
            'val': val_dataset,
            'full': full_dataset
        }
        
        self.dataloaders[expert_name] = {
            'train': train_loader,
            'val': val_loader
        }
        
        logger.info(f"Prepared data for {expert_name}: {len(train_dataset)} train, {len(val_dataset)} val")
        
        return train_loader, val_loader
    
    def get_dataset_stats(self, expert_name: str) -> Dict:
        """Get statistics for expert dataset."""
        if expert_name not in self.datasets:
            raise ValueError(f"Dataset not prepared for expert: {expert_name}")
        
        full_dataset = self.datasets[expert_name]['full']
        
        stats = {
            'total_examples': len(full_dataset),
            'style_distribution': full_dataset.get_style_distribution(),
            'train_size': len(self.datasets[expert_name]['train']),
            'val_size': len(self.datasets[expert_name]['val']),
        }
        
        # Calculate average text length
        text_lengths = []
        for i in range(min(1000, len(full_dataset))):  # Sample for efficiency
            example = full_dataset[i]
            text_lengths.append(len(full_dataset.examples[i]['text'].split()))
        
        stats['avg_text_length'] = sum(text_lengths) / len(text_lengths)
        stats['min_text_length'] = min(text_lengths)
        stats['max_text_length'] = max(text_lengths)
        
        return stats
    
    def prepare_all_experts(self) -> Dict[str, Tuple[DataLoader, DataLoader]]:
        """Prepare data for all experts."""
        all_loaders = {}
        
        for expert_name in self.config.experts.keys():
            train_loader, val_loader = self.prepare_expert_data(expert_name)
            all_loaders[expert_name] = (train_loader, val_loader)
        
        return all_loaders
    
    def print_data_summary(self):
        """Print summary of all datasets."""
        print("=== Data Summary ===")
        
        total_train = 0
        total_val = 0
        
        for expert_name in self.datasets.keys():
            stats = self.get_dataset_stats(expert_name)
            print(f"\n{expert_name.upper()} Expert:")
            print(f"  Total examples: {stats['total_examples']}")
            print(f"  Train/Val split: {stats['train_size']}/{stats['val_size']}")
            print(f"  Avg text length: {stats['avg_text_length']:.1f} words")
            print(f"  Style distribution: {stats['style_distribution']}")
            
            total_train += stats['train_size']
            total_val += stats['val_size']
        
        print(f"\nOverall: {total_train} train, {total_val} val examples")
        print("=" * 30)


def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Custom collate function for batching."""
    # Stack tensors
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])
    quality_scores = torch.stack([item['quality_score'] for item in batch])
    
    # Collect styles
    styles = [item['style'] for item in batch]
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': labels,
        'styles': styles,
        'quality_scores': quality_scores
    }
