"""
Transformer-based emotion classifier for therapeutic conversations.

Fine-tunes DistilBERT on therapeutic dialogue to predict:
- Valence (0-1: negative → positive)
- Arousal (0-1: calm → activated)
- Primary emotion class (anger, anxiety, sadness, joy, etc.)

Architecture:
- Backbone: DistilBERT-base-uncased
- Task heads:
  - Valence regression (MSELoss)
  - Arousal regression (MSELoss)
  - Emotion classification (CrossEntropyLoss)
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class TherapeuticEmotionClassifier(nn.Module):
    """Multi-task emotion classifier for therapeutic dialogue."""

    EMOTION_CLASSES = [
        "anger",
        "anxiety",
        "sadness",
        "joy",
        "shame",
        "guilt",
        "hope",
        "relief",
        "grief",
        "neutral",
    ]

    def __init__(
        self, model_name: str = "distilbert-base-uncased", dropout: float = 0.2
    ):
        """
        Initialize classifier.

        Args:
            model_name: HuggingFace model identifier
            dropout: Dropout rate for regularization
        """
        super().__init__()
        self.model_name = model_name
        self.num_emotions = len(self.EMOTION_CLASSES)

        # Load pretrained model
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size

        # Shared representation layers
        self.dropout = nn.Dropout(dropout)
        self.shared_dense = nn.Linear(hidden_size, 256)
        self.shared_activation = nn.ReLU()

        # Task-specific heads
        # Valence head (regression)
        self.valence_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),  # Output: 0-1
        )

        # Arousal head (regression)
        self.arousal_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
            nn.Sigmoid(),  # Output: 0-1
        )

        # Emotion classification head
        self.emotion_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, self.num_emotions),
        )

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            (valence, arousal, emotion_logits)
        """
        # Encode text
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        cls_output = self.dropout(cls_output)

        # Shared layers
        shared_repr = self.shared_dense(cls_output)
        shared_repr = self.shared_activation(shared_repr)

        # Task heads
        valence = self.valence_head(shared_repr)
        arousal = self.arousal_head(shared_repr)
        emotion_logits = self.emotion_head(shared_repr)

        return valence, arousal, emotion_logits


class EmotionClassifierConfig:
    """Configuration for emotion classifier training."""

    model_name: str = "distilbert-base-uncased"
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 3
    max_seq_length: int = 512
    dropout: float = 0.2

    # Loss weights for multi-task learning
    valence_weight: float = 1.0
    arousal_weight: float = 1.0
    emotion_weight: float = 1.0


class EmotionClassifierTrainer:
    """Trainer for therapeutic emotion classifier."""

    def __init__(
        self,
        model: TherapeuticEmotionClassifier,
        device: str = "cpu",
        learning_rate: float = 2e-5,
    ):
        self.model = model.to(device)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model.model_name)

        # Optimizers
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

    def encode_text(self, text: str, max_length: int = 512) -> Dict:
        """Encode text for model input."""
        encoding = self.tokenizer(
            text,
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in encoding.items()}

    def predict(self, text: str) -> Dict[str, float]:
        """
        Predict emotion for a single utterance.

        Args:
            text: Therapeutic utterance

        Returns:
            {
                "valence": float (0-1),
                "arousal": float (0-1),
                "primary_emotion": str,
                "emotion_scores": dict of emotion → probability
            }
        """
        self.model.eval()
        with torch.no_grad():
            encoding = self.encode_text(text)
            valence, arousal, emotion_logits = self.model(**encoding)

            # Convert to cpu numpy for output
            valence_val = valence.cpu().item()
            arousal_val = arousal.cpu().item()

            # Get emotion probabilities
            emotion_probs = torch.softmax(emotion_logits, dim=1)[0]
            primary_emotion_idx = emotion_probs.argmax().item()
            primary_emotion = self.model.EMOTION_CLASSES[primary_emotion_idx]

            emotion_scores = {
                emotion: prob.item()
                for emotion, prob in zip(self.model.EMOTION_CLASSES, emotion_probs)
            }

        return {
            "valence": valence_val,
            "arousal": arousal_val,
            "primary_emotion": primary_emotion,
            "emotion_scores": emotion_scores,
        }

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Predict emotions for multiple utterances."""
        return [self.predict(text) for text in texts]


if __name__ == "__main__":
    # Example usage
    model = TherapeuticEmotionClassifier()
    trainer = EmotionClassifierTrainer(model, device="cpu")

    # Test predictions
    examples = [
        "I'm feeling really anxious and overwhelmed right now.",
        "I think things are getting better. I'm cautiously hopeful.",
        "I'm so angry at myself for letting this happen.",
    ]

    print("=== Transformer Emotion Classifier Examples ===\n")
    for text in examples:
        result = trainer.predict(text)
        print(f"Text: {text}")
        print(f"Valence: {result['valence']:.3f}")
        print(f"Arousal: {result['arousal']:.3f}")
        print(f"Primary Emotion: {result['primary_emotion']}")
        print()
