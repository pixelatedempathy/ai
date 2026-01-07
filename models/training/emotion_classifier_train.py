"""
Training script for transformer-based emotion classifier.

Fine-tunes DistilBERT on therapeutic conversation data with emotion labels.
Supports multi-task learning: valence regression, arousal regression,
emotion classification.

Usage:
    uv run python -m ai.models.training.emotion_classifier_train \\
        --data-path data/therapeutic/processed/ \\
        --output-dir checkpoints/emotion-classifier \\
        --batch-size 16 \\
        --num-epochs 3
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from emotion_classifier import (
    EmotionClassifierConfig,
    TherapeuticEmotionClassifier,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TherapeuticEmotionDataset(Dataset):
    """Dataset for therapeutic conversations with emotion annotations."""

    def __init__(
        self,
        conversations: List[Dict],
        tokenizer,
        max_length: int = 512,
    ):
        """
        Initialize dataset.

        Args:
            conversations: List of dicts with 'text', 'valence', 'arousal',
                'emotion' keys
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Validate data
        self._validate_data()

    def _validate_data(self):
        """Validate required fields in conversations."""
        required_fields = {"text", "valence", "arousal", "emotion"}
        for i, conv in enumerate(self.conversations):
            if missing := required_fields - set(conv.keys()):
                raise ValueError(f"Conversation {i} missing fields: {missing}")

            # Validate ranges
            if not (0 <= conv["valence"] <= 1):
                raise ValueError(
                    f"Conversation {i}: valence must be in [0, 1], "
                    f"got {conv['valence']}"
                )
            if not (0 <= conv["arousal"] <= 1):
                raise ValueError(
                    f"Conversation {i}: arousal must be in [0, 1], "
                    f"got {conv['arousal']}"
                )

    def __len__(self) -> int:
        return len(self.conversations)

    def __getitem__(self, idx: int) -> Dict:
        """Get single example."""
        conv = self.conversations[idx]

        # Tokenize
        encoding = self.tokenizer(
            conv["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        # Squeeze batch dimension from tokenizer output
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "valence": torch.tensor(conv["valence"], dtype=torch.float32),
            "arousal": torch.tensor(conv["arousal"], dtype=torch.float32),
            "emotion": torch.tensor(
                list(TherapeuticEmotionClassifier.EMOTION_CLASSES).index(
                    conv["emotion"]
                ),
                dtype=torch.long,
            ),
        }


class EmotionClassifierTrainer:
    """Trainer for emotion classifier with multi-task learning."""

    def __init__(
        self,
        model: TherapeuticEmotionClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        config: Optional[EmotionClassifierConfig] = None,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.config = config or EmotionClassifierConfig()

        # Loss functions
        self.mse_loss = nn.MSELoss()
        self.ce_loss = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = AdamW(
            model.parameters(),
            lr=self.config.learning_rate,
        )

        # Learning rate scheduler
        total_steps = len(train_loader) * self.config.num_epochs
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=total_steps // 10,
            num_training_steps=total_steps,
        )

        # Metrics tracking
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float("inf")

    def _compute_loss(
        self,
        valence_pred: torch.Tensor,
        arousal_pred: torch.Tensor,
        emotion_pred: torch.Tensor,
        valence_true: torch.Tensor,
        arousal_true: torch.Tensor,
        emotion_true: torch.Tensor,
    ) -> torch.Tensor:
        """Compute multi-task loss."""
        valence_loss = self.mse_loss(valence_pred.squeeze(), valence_true)
        arousal_loss = self.mse_loss(arousal_pred.squeeze(), arousal_true)
        emotion_loss = self.ce_loss(emotion_pred, emotion_true)

        return (
            self.config.valence_weight * valence_loss
            + self.config.arousal_weight * arousal_loss
            + self.config.emotion_weight * emotion_loss
        )

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch in progress_bar:
            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            valence_true = batch["valence"].to(self.device)
            arousal_true = batch["arousal"].to(self.device)
            emotion_true = batch["emotion"].to(self.device)

            # Forward pass
            valence_pred, arousal_pred, emotion_pred = self.model(
                input_ids, attention_mask
            )

            # Compute loss
            loss = self._compute_loss(
                valence_pred,
                arousal_pred,
                emotion_pred,
                valence_true,
                arousal_true,
                emotion_true,
            )

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"loss": loss.item()})

        epoch_loss = total_loss / len(self.train_loader)
        self.train_losses.append(epoch_loss)
        return epoch_loss

    def validate(self) -> Tuple[float, Dict]:
        """Validate on validation set."""
        self.model.eval()
        total_loss = 0.0

        valence_preds = []
        arousal_preds = []
        emotion_preds = []
        emotion_trues = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", leave=False):
                # Move to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                valence_true = batch["valence"].to(self.device)
                arousal_true = batch["arousal"].to(self.device)
                emotion_true = batch["emotion"].to(self.device)

                # Forward pass
                valence_pred, arousal_pred, emotion_pred = self.model(
                    input_ids, attention_mask
                )

                # Compute loss
                loss = self._compute_loss(
                    valence_pred,
                    arousal_pred,
                    emotion_pred,
                    valence_true,
                    arousal_true,
                    emotion_true,
                )
                total_loss += loss.item()

                # Collect predictions
                valence_preds.extend(valence_pred.squeeze().cpu().numpy())
                arousal_preds.extend(arousal_pred.squeeze().cpu().numpy())
                emotion_preds.extend(emotion_pred.argmax(1).cpu().numpy())
                emotion_trues.extend(emotion_true.cpu().numpy())

        epoch_loss = total_loss / len(self.val_loader)
        self.val_losses.append(epoch_loss)

        # Compute metrics
        metrics = {
            "val_loss": epoch_loss,
            "emotion_f1": f1_score(emotion_trues, emotion_preds, average="weighted"),
            "emotion_accuracy": accuracy_score(emotion_trues, emotion_preds),
        }

        return epoch_loss, metrics

    def train(self, num_epochs: Optional[int] = None) -> Dict:
        """
        Train for specified number of epochs.

        Args:
            num_epochs: Number of epochs (uses config if None)

        Returns:
            Training history dictionary
        """
        num_epochs = num_epochs or self.config.num_epochs

        logger.info(f"Starting training for {num_epochs} epochs on {self.device}")

        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss, metrics = self.validate()

            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Emotion F1: {metrics['emotion_f1']:.4f} | "
                f"Accuracy: {metrics['emotion_accuracy']:.4f}"
            )

            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                logger.info("New best model! Saving checkpoint...")
                self.save_checkpoint(metrics)

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
        }

    def save_checkpoint(self, metrics: Dict, path: Optional[Path] = None):
        """Save model checkpoint."""
        if path is None:
            path = Path("checkpoints/emotion-classifier/best_model.pt")
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state": self.model.state_dict(),
            "model_name": self.model.model_name,
            "metrics": metrics,
        }

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")


def load_therapeutic_data(data_path: str) -> List[Dict]:
    """
    Load therapeutic conversation data.

    Expected format: JSON file with list of dicts containing:
    - 'text': utterance text
    - 'valence': float in [0, 1]
    - 'arousal': float in [0, 1]
    - 'emotion': emotion class string
    """
    data_path = Path(data_path)

    if not data_path.exists():
        logger.warning(
            f"Data path {data_path} not found. Using synthetic data for demonstration."
        )
        # Generate synthetic data for demonstration
        return _generate_synthetic_data(100)

    # Load from JSON
    if data_path.suffix == ".json":
        with open(data_path) as f:
            return json.load(f)

    # Load from directory of JSON files
    conversations = []
    for json_file in data_path.glob("*.json"):
        with open(json_file) as f:
            conversations.extend(json.load(f))

    return conversations


def _generate_synthetic_data(num_samples: int = 100) -> List[Dict]:
    """Generate synthetic therapeutic data for demonstration."""
    emotions = TherapeuticEmotionClassifier.EMOTION_CLASSES
    synthetic_utterances = {
        "anger": [
            "I'm so frustrated with this situation!",
            "This makes me absolutely furious.",
            "I can't believe how angry I am right now.",
        ],
        "anxiety": [
            "I'm feeling really anxious and worried.",
            "What if everything goes wrong?",
            "I have this constant sense of dread.",
        ],
        "sadness": [
            "I feel so sad and empty inside.",
            "Everything feels hopeless right now.",
            "I just want to cry.",
        ],
        "joy": [
            "I'm so happy about this!",
            "This brings me so much joy.",
            "I feel wonderful today!",
        ],
        "shame": [
            "I feel so ashamed of myself.",
            "I'm embarrassed about what I did.",
            "I feel like such a failure.",
        ],
        "guilt": [
            "I feel guilty for what happened.",
            "I blame myself for this.",
            "I should have done better.",
        ],
        "hope": [
            "I think things might get better.",
            "I have hope for the future.",
            "Things are looking up.",
        ],
        "relief": [
            "I'm so relieved this is over.",
            "Finally, some peace and quiet.",
            "I feel much better now.",
        ],
        "grief": [
            "I miss them so much.",
            "The grief feels overwhelming.",
            "I can't accept they're gone.",
        ],
        "neutral": [
            "I went to the store today.",
            "The weather is nice.",
            "That happened yesterday.",
        ],
    }

    # Map emotions to valence/arousal
    emotion_to_dims = {
        "anger": (0.2, 0.9),  # negative, high arousal
        "anxiety": (0.3, 0.8),  # negative, high arousal
        "sadness": (0.2, 0.3),  # negative, low arousal
        "joy": (0.9, 0.7),  # positive, high arousal
        "shame": (0.1, 0.6),  # negative, medium arousal
        "guilt": (0.2, 0.5),  # negative, medium arousal
        "hope": (0.8, 0.5),  # positive, medium arousal
        "relief": (0.8, 0.2),  # positive, low arousal
        "grief": (0.1, 0.4),  # negative, low arousal
        "neutral": (0.5, 0.5),  # neutral
    }

    data = []
    for _ in range(num_samples):
        emotion = np.random.choice(emotions)
        text = np.random.choice(synthetic_utterances[emotion])
        valence, arousal = emotion_to_dims[emotion]

        # Add noise
        valence += np.random.normal(0, 0.1)
        arousal += np.random.normal(0, 0.1)
        valence = np.clip(valence, 0, 1)
        arousal = np.clip(arousal, 0, 1)

        data.append(
            {
                "text": text,
                "valence": float(valence),
                "arousal": float(arousal),
                "emotion": emotion,
            }
        )

    return data


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train emotion classifier")
    parser.add_argument(
        "--data-path",
        type=str,
        default="data/therapeutic/processed/",
        help="Path to therapeutic conversation data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/emotion-classifier",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-5,
        help="Learning rate",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Validation split ratio",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info(f"Loading data from {args.data_path}")
    conversations = load_therapeutic_data(args.data_path)
    logger.info(f"Loaded {len(conversations)} conversations")

    # Split data
    train_size = int(len(conversations) * (1 - args.val_split))
    train_data = conversations[:train_size]
    val_data = conversations[train_size:]

    logger.info(f"Train/Val split: {len(train_data)}/{len(val_data)}")

    # Create datasets
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    train_dataset = TherapeuticEmotionDataset(
        train_data, tokenizer, max_length=args.max_seq_length
    )
    val_dataset = TherapeuticEmotionDataset(
        val_data, tokenizer, max_length=args.max_seq_length
    )

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model
    logger.info("Creating emotion classifier model")
    model = TherapeuticEmotionClassifier()

    # Create config
    config = EmotionClassifierConfig()
    config.learning_rate = args.learning_rate
    config.batch_size = args.batch_size
    config.num_epochs = args.num_epochs

    # Create trainer
    trainer = EmotionClassifierTrainer(
        model, train_loader, val_loader, device=args.device, config=config
    )

    # Train
    history = trainer.train(num_epochs=args.num_epochs)

    # Save history
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"Training complete! History saved to {history_path}")
    logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")


if __name__ == "__main__":
    main()
