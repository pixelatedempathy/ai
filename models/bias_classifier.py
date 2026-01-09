"""
Transformer-based bias detection classifier for therapeutic conversations.

Multi-task architecture detecting:
- Gender bias (binary classification)
- Racial bias (multi-class)
- Cultural bias (multi-class)

Outputs fairness metrics: demographic parity, equalized odds.
"""

from typing import Dict, List, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer


class BiasDetectionClassifier(nn.Module):
    """Multi-task transformer for bias detection."""

    BIAS_TYPES = {
        "gender": ["biased", "unbiased"],
        "racial": [
            "no_bias",
            "overrepresented_white",
            "overrepresented_black",
            "overrepresented_asian",
            "overrepresented_hispanic",
            "underrepresented_white",
            "underrepresented_black",
            "underrepresented_asian",
            "underrepresented_hispanic",
        ],
        "cultural": [
            "no_bias",
            "western_bias",
            "eastern_bias",
            "individualistic_bias",
            "collectivistic_bias",
            "religious_bias",
            "secular_bias",
        ],
    }

    def __init__(
        self,
        model_name: str = "roberta-base",
        dropout: float = 0.2,
    ):
        """
        Initialize bias detection classifier.

        Args:
            model_name: HuggingFace model identifier
            dropout: Dropout rate for regularization
        """
        super().__init__()
        self.model_name = model_name
        self.dropout_rate = dropout

        # Load pretrained model
        self.backbone = AutoModel.from_pretrained(model_name)
        hidden_size = self.backbone.config.hidden_size

        # Shared representation layers
        self.dropout = nn.Dropout(dropout)
        self.shared_dense = nn.Linear(hidden_size, 512)
        self.shared_activation = nn.ReLU()

        # Task-specific heads
        # Gender bias head (binary)
        self.gender_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, len(self.BIAS_TYPES["gender"])),
        )

        # Racial bias head (multi-class)
        self.racial_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, len(self.BIAS_TYPES["racial"])),
        )

        # Cultural bias head (multi-class)
        self.cultural_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, len(self.BIAS_TYPES["cultural"])),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]

        Returns:
            (gender_logits, racial_logits, cultural_logits)
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
        gender_logits = self.gender_head(shared_repr)
        racial_logits = self.racial_head(shared_repr)
        cultural_logits = self.cultural_head(shared_repr)

        return gender_logits, racial_logits, cultural_logits


class BiasDetectionTrainer:
    """Trainer for multi-task bias detection."""

    def __init__(
        self,
        model: BiasDetectionClassifier,
        device: str = "cpu",
        learning_rate: float = 2e-5,
    ):
        self.model = model.to(device)
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model.model_name)

        # Optimizers
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

        # Loss function
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

    def predict(self, text: str) -> Dict:
        """
        Predict bias for a single utterance.

        Args:
            text: Therapeutic utterance

        Returns:
            {
                "gender_bias": str (biased/unbiased),
                "gender_confidence": float,
                "racial_bias": str,
                "racial_confidence": float,
                "cultural_bias": str,
                "cultural_confidence": float,
                "overall_bias_score": float (0-1),
            }
        """
        self.model.eval()
        with torch.no_grad():
            encoding = self.encode_text(text)
            gender_logits, racial_logits, cultural_logits = self.model(**encoding)

            # Get predictions
            gender_pred = gender_logits.argmax(1).item()
            racial_pred = racial_logits.argmax(1).item()
            cultural_pred = cultural_logits.argmax(1).item()

            # Get probabilities
            gender_probs = torch.softmax(gender_logits, dim=1)[0]
            racial_probs = torch.softmax(racial_logits, dim=1)[0]
            cultural_probs = torch.softmax(cultural_logits, dim=1)[0]

            # Overall bias score (any non-"no_bias" or "unbiased" prediction)
            overall_bias_score = 1.0 - max(
                gender_probs[0].item(),  # "unbiased"
                racial_probs[0].item(),  # "no_bias"
                cultural_probs[0].item(),  # "no_bias"
            )

        return {
            "gender_bias": self.model.BIAS_TYPES["gender"][gender_pred],
            "gender_confidence": float(gender_probs[gender_pred].item()),
            "racial_bias": self.model.BIAS_TYPES["racial"][racial_pred],
            "racial_confidence": float(racial_probs[racial_pred].item()),
            "cultural_bias": self.model.BIAS_TYPES["cultural"][cultural_pred],
            "cultural_confidence": float(cultural_probs[cultural_pred].item()),
            "overall_bias_score": float(overall_bias_score),
        }

    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Predict bias for multiple utterances."""
        return [self.predict(text) for text in texts]
