import torch
from torch.utils.data import Dataset

class ConversationDataset(Dataset):
    def __init__(self, conversations, tokenizer, max_length=1024):
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.conversations)
    
    def __getitem__(self, idx):
        conversation = self.conversations[idx]
        
        # Tokenize the conversation
        tokenized = self.tokenizer(
            conversation,
            max_length=self.max_length,
            truncation=True,
            padding=False,  # Don't pad here, let the collator handle it
            return_tensors=None  # Return lists, not tensors
        )
        
        # For causal LM, labels are the same as input_ids but shifted
        input_ids = tokenized['input_ids']
        labels = input_ids.copy()
        
        return {
            'input_ids': input_ids,
            'labels': labels
        }