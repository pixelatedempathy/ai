"""
Training script for bias detection classifier.

Trains multi-task transformer on biased/unbiased conversation pairs.
Detects gender, racial, and cultural bias with fairness metrics.

Usage:
    uv run python -m ai.models.training.bias_classifier_train \\
        --biased-data data/therapeutic/synthetic/biased \\
        --output-dir checkpoints/bias-classifier \\
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
from sklearn.metrics import f1_score
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from models.bias_classifier import BiasDetectionClassifier

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class BiasDataset(Dataset):
    """Dataset for biased/unbiased conversation pairs."""

    def __init__(
        self,
        conversations: List[Dict],
        tokenizer,
        max_length: int = 512,
    ):
        """
        Initialize dataset.

        Args:
            conversations: List of dicts with 'text', 'gender_bias',
                'racial_bias', 'cultural_bias' keys
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length
        """
        self.conversations = conversations
        self.tokenizer = tokenizer
        self.max_length = max_length

        self._validate_data()

    def _validate_data(self):
        """Validate required fields."""
        required_fields = {
            "text",
            "gender_bias",
            "racial_bias",
            "cultural_bias",
        }
        for i, conv in enumerate(self.conversations):
            if missing := required_fields - set(conv.keys()):
                raise ValueError(f"Conversation {i} missing fields: {missing}")

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

        # Map bias labels to indices
        gender_classes = BiasDetectionClassifier.BIAS_TYPES["gender"]
        racial_classes = BiasDetectionClassifier.BIAS_TYPES["racial"]
        cultural_classes = BiasDetectionClassifier.BIAS_TYPES["cultural"]

        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "gender_bias": torch.tensor(
                gender_classes.index(conv["gender_bias"]),
                dtype=torch.long,
            ),
            "racial_bias": torch.tensor(
                racial_classes.index(conv["racial_bias"]),
                dtype=torch.long,
            ),
            "cultural_bias": torch.tensor(
                cultural_classes.index(conv["cultural_bias"]),
                dtype=torch.long,
            ),
        }


class BiasDetectionTrainer:
    """Trainer for multi-task bias detection."""

    def __init__(
        self,
        model: BiasDetectionClassifier,
        train_loader: DataLoader,
        val_loader: DataLoader,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 2e-5,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device

        # Loss function
        self.ce_loss = nn.CrossEntropyLoss()

        # Optimizer
        self.optimizer = AdamW(model.parameters(), lr=learning_rate)

        # Learning rate scheduler
        total_steps = len(train_loader) * 3  # Assuming 3 epochs
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
        gender_pred: torch.Tensor,
        racial_pred: torch.Tensor,
        cultural_pred: torch.Tensor,
        gender_true: torch.Tensor,
        racial_true: torch.Tensor,
        cultural_true: torch.Tensor,
    ) -> torch.Tensor:
        """Compute multi-task loss."""
        gender_loss = self.ce_loss(gender_pred, gender_true)
        racial_loss = self.ce_loss(racial_pred, racial_true)
        cultural_loss = self.ce_loss(cultural_pred, cultural_true)

        return gender_loss + racial_loss + cultural_loss

    def train_epoch(self) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch in progress_bar:
            # Move to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            gender_true = batch["gender_bias"].to(self.device)
            racial_true = batch["racial_bias"].to(self.device)
            cultural_true = batch["cultural_bias"].to(self.device)

            # Forward pass
            gender_pred, racial_pred, cultural_pred = self.model(
                input_ids, attention_mask
            )

            # Compute loss
            loss = self._compute_loss(
                gender_pred,
                racial_pred,
                cultural_pred,
                gender_true,
                racial_true,
                cultural_true,
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

        gender_preds = []
        racial_preds = []
        cultural_preds = []
        gender_trues = []
        racial_trues = []
        cultural_trues = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating", leave=False):
                # Move to device
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                gender_true = batch["gender_bias"].to(self.device)
                racial_true = batch["racial_bias"].to(self.device)
                cultural_true = batch["cultural_bias"].to(self.device)

                # Forward pass
                gender_pred, racial_pred, cultural_pred = self.model(
                    input_ids, attention_mask
                )

                # Compute loss
                loss = self._compute_loss(
                    gender_pred,
                    racial_pred,
                    cultural_pred,
                    gender_true,
                    racial_true,
                    cultural_true,
                )
                total_loss += loss.item()

                # Collect predictions
                gender_preds.extend(gender_pred.argmax(1).cpu().numpy())
                racial_preds.extend(racial_pred.argmax(1).cpu().numpy())
                cultural_preds.extend(cultural_pred.argmax(1).cpu().numpy())
                gender_trues.extend(gender_true.cpu().numpy())
                racial_trues.extend(racial_true.cpu().numpy())
                cultural_trues.extend(cultural_true.cpu().numpy())

        epoch_loss = total_loss / len(self.val_loader)
        self.val_losses.append(epoch_loss)

        # Compute metrics
        metrics = {
            "val_loss": epoch_loss,
            "gender_f1": f1_score(gender_trues, gender_preds, average="weighted"),
            "racial_f1": f1_score(racial_trues, racial_preds, average="weighted"),
            "cultural_f1": f1_score(cultural_trues, cultural_preds, average="weighted"),
        }

        return epoch_loss, metrics

    def train(self, num_epochs: int = 3) -> Dict:
        """Train for specified epochs."""
        logger.info(f"Starting training for {num_epochs} epochs on {self.device}")

        for epoch in range(num_epochs):
            train_loss = self.train_epoch()
            val_loss, metrics = self.validate()

            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Gender F1: {metrics['gender_f1']:.4f} | "
                f"Racial F1: {metrics['racial_f1']:.4f} | "
                f"Cultural F1: {metrics['cultural_f1']:.4f}"
            )

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
            path = Path("checkpoints/bias-classifier/best_model.pt")
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "model_state": self.model.state_dict(),
            "model_name": self.model.model_name,
            "metrics": metrics,
        }

        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")


def generate_synthetic_biased_data(num_samples: int = 100) -> List[Dict]:
    """Generate synthetic biased conversation pairs."""
    biased_utterances = {
        # Gender bias examples
        "gender_biased": [
            "Women are naturally better at caregiving than men.",
            "A man should be the primary breadwinner in a family.",
            "Women are too emotional to handle technical roles.",
        ],
        "gender_unbiased": [
            "People have different strengths regardless of gender.",
            "Caregiving roles can be filled by anyone capable.",
            "Technical ability varies by individual, not gender.",
        ],
        # Racial bias examples
        "racial_overrep_white": [
            "White people have always been the most successful.",
            "The default leader should be white.",
            "White perspectives are the most valid.",
        ],
        "racial_underrep_white": [
            "White people never face discrimination.",
            "Only minorities can experience systemic racism.",
            "White people don't understand struggle.",
        ],
        "racial_no_bias": [
            "People of all races have valuable perspectives.",
            "Success depends on individual effort and opportunity.",
            "Every community has unique strengths.",
        ],
        # Cultural bias examples
        "cultural_western_bias": [
            "Western values are the most advanced.",
            "Individualism is the only path to success.",
            "Non-Western traditions are outdated.",
        ],
        "cultural_no_bias": [
            "Different cultures have different but equally valid values.",
            "Success can be defined in many ways.",
            "Both individual and community values matter.",
        ],
    }

    data = []
    for _ in range(num_samples):
        # Gender bias
        gender_bias = np.random.choice(["biased", "unbiased"])
        gender_text = np.random.choice(biased_utterances[f"gender_{gender_bias}"])

        # Racial bias
        racial_options = [
            "no_bias",
            "overrepresented_white",
            "underrepresented_white",
        ]
        racial_bias = np.random.choice(racial_options)
        if racial_bias == "no_bias":
            racial_text = np.random.choice(biased_utterances["racial_no_bias"])
        elif racial_bias == "overrepresented_white":
            racial_text = np.random.choice(biased_utterances["racial_overrep_white"])
        else:
            racial_text = np.random.choice(biased_utterances["racial_underrep_white"])

        # Cultural bias
        cultural_options = ["no_bias", "western_bias"]
        cultural_bias = np.random.choice(cultural_options)
        cultural_text = (
            np.random.choice(biased_utterances["cultural_no_bias"])
            if cultural_bias == "no_bias"
            else np.random.choice(biased_utterances["cultural_western_bias"])
        )
        # Combine texts
        text = f"{gender_text} {racial_text} {cultural_text}"

        data.append(
            {
                "text": text,
                "gender_bias": gender_bias,
                "racial_bias": racial_bias,
                "cultural_bias": cultural_bias,
            }
        )

    return data


def main():
    """Main training script."""
    parser = argparse.ArgumentParser(description="Train bias detection classifier")
    parser.add_argument(
        "--biased-data",
        type=str,
        default="data/therapeutic/synthetic/biased/",
        help="Path to biased conversation data",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="checkpoints/bias-classifier",
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
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on",
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load or generate data
    logger.info(f"Loading biased data from {args.biased_data}")
    data_path = Path(args.biased_data)

    if not data_path.exists():
        logger.info("Data path not found. Generating synthetic biased data...")
        conversations = generate_synthetic_biased_data(200)
    else:
        # Load from JSON files
        conversations = []
        for json_file in data_path.glob("*.json"):
            with open(json_file) as f:
                conversations.extend(json.load(f))

    logger.info(f"Loaded {len(conversations)} conversations")

    # Split train/val
    val_size = int(len(conversations) * 0.2)
    train_data = conversations[:-val_size]
    val_data = conversations[-val_size:]

    logger.info(f"Train/Val split: {len(train_data)}/{len(val_data)}")

    # Create datasets and loaders
    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    train_dataset = BiasDataset(train_data, tokenizer)
    val_dataset = BiasDataset(val_data, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model
    logger.info("Creating bias detection classifier")
    model = BiasDetectionClassifier()

    # Create trainer
    trainer = BiasDetectionTrainer(
        model,
        train_loader,
        val_loader,
        device=args.device,
        learning_rate=args.learning_rate,
    )

    # Train
    history = trainer.train(num_epochs=args.num_epochs)

    # Save history
    history_path = output_dir / "training_history.json"
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"Training complete! Saved to {history_path}")
    logger.info(f"Best validation loss: {trainer.best_val_loss:.4f}")


if __name__ == "__main__":
    main()
