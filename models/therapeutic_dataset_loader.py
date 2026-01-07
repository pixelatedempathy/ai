"""
Therapeutic Dialogue Dataset Loader

Loads therapeutic conversations from various sources including:
- Pixelated Empathy API
- Local JSON/JSONL files
- HuggingFace datasets

Includes filtering, splitting, and preprocessing capabilities.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DatasetSplit:
    """Container for dataset splits."""

    train: List[Dict[str, Any]]
    validation: List[Dict[str, Any]]
    test: List[Dict[str, Any]]


class TherapeuticConversationLoader:
    """Loads and manages therapeutic conversation datasets."""

    def __init__(
        self,
        min_quality_score: float = 0.7,
        min_conversation_length: int = 10,
        filter_crisis_escalations: bool = True,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize loader.

        Args:
            min_quality_score: Minimum quality score filter
            min_conversation_length: Minimum character length for conversations
            filter_crisis_escalations: Remove conversations with harm escalation
            cache_dir: Directory for caching datasets
        """
        self.min_quality_score = min_quality_score
        self.min_conversation_length = min_conversation_length
        self.filter_crisis_escalations = filter_crisis_escalations
        self.cache_dir = Path(cache_dir) if cache_dir else Path("./cache/datasets")
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def load_from_json_files(
        self,
        directory: str,
        pattern: str = "*.json",
    ) -> List[Dict[str, Any]]:
        """
        Load conversations from JSON files.

        Args:
            directory: Directory containing JSON files
            pattern: File pattern to match

        Returns:
            List of conversation dictionaries
        """
        conversations = []
        directory_path = Path(directory)

        logger.info(f"Loading conversations from {directory}")

        for file_path in directory_path.glob(pattern):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)

                    # Handle both single conversation and list of conversations
                    if isinstance(data, list):
                        conversations.extend(data)
                    else:
                        conversations.append(data)

                logger.info(f"Loaded {len(data)} conversations from {file_path}")
            except Exception as e:
                logger.error(f"Error loading {file_path}: {e}")

        return self._filter_conversations(conversations)

    def load_from_jsonl_file(self, file_path: str) -> List[Dict[str, Any]]:
        """
        Load conversations from JSONL file (one per line).

        Args:
            file_path: Path to JSONL file

        Returns:
            List of conversation dictionaries
        """
        conversations = []
        file_path = Path(file_path)

        logger.info(f"Loading conversations from {file_path}")

        with open(file_path, "r") as f:
            for line_num, line in enumerate(f, 1):
                try:
                    conversation = json.loads(line)
                    conversations.append(conversation)
                except json.JSONDecodeError as e:
                    logger.error(f"JSON error at line {line_num}: {e}")

        logger.info(f"Loaded {len(conversations)} conversations from {file_path}")
        return self._filter_conversations(conversations)

    def load_from_api(
        self,
        api_client: Any,
        dataset_name: Optional[str] = None,
        tier: str = "professional",
        limit: int = 10000,
        batch_size: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Load conversations from Pixelated Empathy API.

        Args:
            api_client: Initialized PixelatedEmpathyAPI client
            dataset_name: Optional specific dataset
            tier: Quality tier (professional, quality, standard)
            limit: Maximum conversations to load
            batch_size: Batch size for API requests

        Returns:
            List of conversation dictionaries
        """
        conversations = []
        loaded = 0

        logger.info(
            f"Loading from API (dataset={dataset_name}, tier={tier}, limit={limit})"
        )

        try:
            for conversation in api_client.iterConversations(
                dataset=dataset_name,
                tier=tier,
                batchSize=batch_size,
            ):
                conversations.append(conversation)
                loaded += 1

                if loaded >= limit:
                    break

                if loaded % 1000 == 0:
                    logger.info(f"Loaded {loaded} conversations...")

        except Exception as e:
            logger.error(f"Error loading from API: {e}")

        logger.info(f"Loaded {loaded} conversations from API")
        return self._filter_conversations(conversations)

    def load_from_huggingface(
        self,
        dataset_name: str,
        split: str = "train",
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Load conversations from HuggingFace datasets.

        Args:
            dataset_name: HuggingFace dataset identifier
            split: Dataset split to load
            limit: Maximum conversations to load

        Returns:
            List of conversation dictionaries
        """
        try:
            from datasets import load_dataset
        except ImportError:
            logger.error("datasets package required: pip install datasets")
            return []

        logger.info(f"Loading from HuggingFace: {dataset_name}/{split}")

        try:
            dataset = load_dataset(dataset_name, split=split)

            # Convert to list and limit
            conversations = dataset.to_list()
            if limit:
                conversations = conversations[:limit]

            logger.info(f"Loaded {len(conversations)} conversations from HuggingFace")
            return self._filter_conversations(conversations)

        except Exception as e:
            logger.error(f"Error loading from HuggingFace: {e}")
            return []

    def split_conversations(
        self,
        conversations: List[Dict[str, Any]],
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> DatasetSplit:
        """
        Split conversations into train/val/test.

        Args:
            conversations: List of conversations
            train_ratio: Training set ratio
            val_ratio: Validation set ratio
            test_ratio: Test set ratio
            seed: Random seed for reproducibility

        Returns:
            DatasetSplit with train/val/test splits
        """
        assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-6

        np.random.seed(seed)
        indices = np.random.permutation(len(conversations))

        train_size = int(len(conversations) * train_ratio)
        val_size = int(len(conversations) * val_ratio)

        train_indices = indices[:train_size]
        val_indices = indices[train_size : train_size + val_size]
        test_indices = indices[train_size + val_size :]

        train_convs = [conversations[i] for i in train_indices]
        val_convs = [conversations[i] for i in val_indices]
        test_convs = [conversations[i] for i in test_indices]

        logger.info(
            f"Split conversations: train={len(train_convs)}, "
            f"val={len(val_convs)}, test={len(test_convs)}"
        )

        return DatasetSplit(
            train=train_convs,
            validation=val_convs,
            test=test_convs,
        )

    def iterate_conversations(
        self,
        conversations: List[Dict[str, Any]],
        batch_size: int = 32,
    ) -> Generator[List[Dict[str, Any]], None, None]:
        """
        Iterate conversations in batches.

        Args:
            conversations: List of conversations
            batch_size: Batch size

        Yields:
            Batches of conversations
        """
        for i in range(0, len(conversations), batch_size):
            yield conversations[i : i + batch_size]

    def validate_conversation_format(
        self,
        conversation: Dict[str, Any],
    ) -> Tuple[bool, Optional[str]]:
        """
        Validate conversation format.

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check required fields
        if "messages" not in conversation:
            return False, "Missing 'messages' field"

        messages = conversation["messages"]
        if not isinstance(messages, list) or len(messages) == 0:
            return False, "'messages' must be non-empty list"

        # Check message format
        for msg in messages:
            if not isinstance(msg, dict):
                return False, "Each message must be a dictionary"

            if "role" not in msg or "content" not in msg:
                return False, "Each message must have 'role' and 'content'"

            if msg["role"] not in ["user", "assistant", "system"]:
                return False, f"Invalid role: {msg['role']}"

        return True, None

    def _filter_conversations(
        self,
        conversations: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Filter conversations based on quality criteria.

        Args:
            conversations: List of conversations

        Returns:
            Filtered list of conversations
        """
        filtered = []
        skipped = 0

        for conv in conversations:
            # Validate format
            is_valid, error = self.validate_conversation_format(conv)
            if not is_valid:
                skipped += 1
                continue

            # Check quality score
            quality = conv.get("quality_score", 0.5)
            if quality < self.min_quality_score:
                skipped += 1
                continue

            # Check minimum length
            total_length = sum(
                len(msg.get("content", "")) for msg in conv.get("messages", [])
            )
            if total_length < self.min_conversation_length:
                skipped += 1
                continue

            # Check for crisis escalation
            if self.filter_crisis_escalations and self._has_crisis_escalation(conv):
                skipped += 1
                continue

            filtered.append(conv)

        logger.info(
            f"Filtered {len(filtered)} conversations "
            f"(skipped {skipped} invalid/low quality)"
        )

        return filtered

    def _has_crisis_escalation(self, conversation: Dict[str, Any]) -> bool:
        """Check if conversation has crisis escalation without resolution."""
        messages = conversation.get("messages", [])

        has_crisis = False
        last_response_helps = True

        for msg in messages:
            content = msg.get("content", "").lower()

            # Detect crisis signals
            crisis_keywords = [
                "suicide",
                "self harm",
                "kill myself",
                "dying",
            ]
            if any(keyword in content for keyword in crisis_keywords):
                has_crisis = True

            # Check if last response from therapist addresses it
            if has_crisis and msg.get("role") == "assistant":
                help_keywords = [
                    "emergency",
                    "crisis",
                    "professional",
                    "safe",
                    "help",
                    "therapist",
                ]
                last_response_helps = any(
                    keyword in content for keyword in help_keywords
                )

        return has_crisis and not last_response_helps

    def save_to_jsonl(
        self,
        conversations: List[Dict[str, Any]],
        output_path: str,
    ) -> None:
        """
        Save conversations to JSONL file.

        Args:
            conversations: List of conversations
            output_path: Path to output file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for conv in conversations:
                f.write(json.dumps(conv) + "\n")

        logger.info(f"Saved {len(conversations)} conversations to {output_path}")

    def get_dataset_statistics(
        self,
        conversations: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Get statistics about dataset.

        Args:
            conversations: List of conversations

        Returns:
            Dictionary of statistics
        """
        if not conversations:
            return {}

        lengths = []
        turn_counts = []
        quality_scores = []
        has_crisis_list = []

        for conv in conversations:
            messages = conv.get("messages", [])
            turn_counts.append(len(messages))

            total_length = sum(len(msg.get("content", "")) for msg in messages)
            lengths.append(total_length)

            quality_scores.append(conv.get("quality_score", 0.5))
            has_crisis_list.append(conv.get("has_crisis_signal", False))

        return {
            "total_conversations": len(conversations),
            "avg_conversation_length": float(np.mean(lengths)),
            "min_conversation_length": int(np.min(lengths)),
            "max_conversation_length": int(np.max(lengths)),
            "avg_turns": float(np.mean(turn_counts)),
            "avg_quality_score": float(np.mean(quality_scores)),
            "conversations_with_crisis": sum(has_crisis_list),
            "crisis_percentage": float(sum(has_crisis_list) / len(conversations) * 100),
        }
