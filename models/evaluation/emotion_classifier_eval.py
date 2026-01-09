"""
Evaluation script for emotion classifier.

Evaluates trained emotion classifier on test set and generates metrics report.

Usage:
    uv run python -m ai.models.evaluation.emotion_classifier_eval \\
        --checkpoint checkpoints/emotion-classifier \\
        --test-data data/therapeutic/test/ \\
        --output reports/emotion_eval.json
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from emotion_classifier import TherapeuticEmotionClassifier

from models.training.emotion_classifier_train import (
    TherapeuticEmotionDataset,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class EmotionClassifierEvaluator:
    """Evaluator for emotion classifier."""

    def __init__(
        self,
        model: TherapeuticEmotionClassifier,
        test_loader: DataLoader,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.model = model.to(device)
        self.test_loader = test_loader
        self.device = device

    def evaluate(self) -> Dict:
        """
        Evaluate model on test set.

        Returns:
            Dictionary with metrics:
            - emotion_accuracy, emotion_f1, emotion_precision, emotion_recall
            - valence_mse, arousal_mse
            - confusion_matrix
        """
        self.model.eval()

        valence_preds = []
        arousal_preds = []
        emotion_preds = []
        emotion_trues = []
        valence_trues = []
        arousal_trues = []

        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                valence_pred, arousal_pred, emotion_pred = self.model(
                    input_ids, attention_mask
                )

                # Collect predictions
                valence_preds.extend(valence_pred.squeeze().cpu().numpy())
                arousal_preds.extend(arousal_pred.squeeze().cpu().numpy())
                emotion_preds.extend(emotion_pred.argmax(1).cpu().numpy())

                # Collect true labels
                valence_trues.extend(batch["valence"].cpu().numpy())
                arousal_trues.extend(batch["arousal"].cpu().numpy())
                emotion_trues.extend(batch["emotion"].cpu().numpy())

        # Compute metrics
        return {
            "emotion": {
                "accuracy": float(accuracy_score(emotion_trues, emotion_preds)),
                "f1_weighted": float(
                    f1_score(
                        emotion_trues,
                        emotion_preds,
                        average="weighted",
                    )
                ),
                "f1_macro": float(
                    f1_score(
                        emotion_trues,
                        emotion_preds,
                        average="macro",
                    )
                ),
                "precision": float(
                    precision_score(
                        emotion_trues,
                        emotion_preds,
                        average="weighted",
                    )
                ),
                "recall": float(
                    recall_score(
                        emotion_trues,
                        emotion_preds,
                        average="weighted",
                    )
                ),
                "confusion_matrix": confusion_matrix(
                    emotion_trues, emotion_preds
                ).tolist(),
            },
            "valence": {
                "mse": float(
                    np.mean(
                        np.square(np.array(valence_preds) - np.array(valence_trues))
                    )
                ),
                "mae": float(
                    np.mean(np.abs(np.array(valence_preds) - np.array(valence_trues)))
                ),
            },
            "arousal": {
                "mse": float(
                    np.mean(
                        np.square(np.array(arousal_preds) - np.array(arousal_trues))
                    )
                ),
                "mae": float(
                    np.mean(np.abs(np.array(arousal_preds) - np.array(arousal_trues)))
                ),
            },
        }

    def evaluate_per_emotion(self) -> Dict:
        """Evaluate metrics per emotion class."""
        self.model.eval()

        emotion_predictions = {
            emotion: {"preds": [], "trues": []}
            for emotion in self.model.EMOTION_CLASSES
        }

        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)

                _, _, emotion_pred = self.model(input_ids, attention_mask)

                emotion_preds = emotion_pred.argmax(1).cpu().numpy()
                emotion_trues = batch["emotion"].cpu().numpy()

                for pred, true in zip(emotion_preds, emotion_trues):
                    emotion_name = self.model.EMOTION_CLASSES[true]
                    emotion_predictions[emotion_name]["trues"].append(true)
                    emotion_predictions[emotion_name]["preds"].append(pred)

        return {
            emotion: {
                "f1": float(
                    f1_score(
                        data["trues"],
                        data["preds"],
                        average="weighted",
                        zero_division=0,
                    )
                ),
                "accuracy": float(accuracy_score(data["trues"], data["preds"])),
                "samples": len(data["trues"]),
            }
            for emotion, data in emotion_predictions.items()
            if len(data["trues"]) != 0
        }


def load_checkpoint(checkpoint_path: str) -> TherapeuticEmotionClassifier:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = TherapeuticEmotionClassifier()
    model.load_state_dict(checkpoint["model_state"])
    return model


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate emotion classifier")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint directory",
    )
    parser.add_argument(
        "--test-data",
        type=str,
        default="data/therapeutic/test/",
        help="Path to test data",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="reports/emotion_eval.json",
        help="Output path for evaluation report",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to evaluate on",
    )

    args = parser.parse_args()

    # Load checkpoint
    checkpoint_path = Path(args.checkpoint) / "best_model.pt"
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found at {checkpoint_path}")
        return

    logger.info(f"Loading checkpoint from {checkpoint_path}")
    model = load_checkpoint(str(checkpoint_path))

    # Load test data
    logger.info(f"Loading test data from {args.test_data}")
    from ai.models.training.emotion_classifier_train import (
        load_therapeutic_data,
    )

    test_conversations = load_therapeutic_data(args.test_data)
    logger.info(f"Loaded {len(test_conversations)} test examples")

    # Create dataset and loader
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    test_dataset = TherapeuticEmotionDataset(test_conversations, tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # Evaluate
    evaluator = EmotionClassifierEvaluator(model, test_loader, device=args.device)
    metrics = evaluator.evaluate()
    per_emotion_metrics = evaluator.evaluate_per_emotion()

    # Prepare report
    report = {
        "overall_metrics": metrics,
        "per_emotion_metrics": per_emotion_metrics,
    }

    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Evaluation report saved to {output_path}")

    # Print summary
    print("\n=== Emotion Classifier Evaluation ===\n")
    print(f"Accuracy: {metrics['emotion']['accuracy']:.4f}")
    print(f"F1 (Weighted): {metrics['emotion']['f1_weighted']:.4f}")
    print(f"F1 (Macro): {metrics['emotion']['f1_macro']:.4f}")
    print(
        f"\nValence MSE: {metrics['valence']['mse']:.4f}, "
        f"MAE: {metrics['valence']['mae']:.4f}"
    )
    print(
        f"Arousal MSE: {metrics['arousal']['mse']:.4f}, "
        f"MAE: {metrics['arousal']['mae']:.4f}"
    )
    print("\n=== Per-Emotion Metrics ===\n")
    for emotion, emo_metrics in per_emotion_metrics.items():
        print(
            f"{emotion:12s} | F1: {emo_metrics['f1']:.4f} | "
            f"Acc: {emo_metrics['accuracy']:.4f} | "
            f"Samples: {emo_metrics['samples']}"
        )


if __name__ == "__main__":
    main()
