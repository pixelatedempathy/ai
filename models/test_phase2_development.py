"""
Tests for Therapeutic Model Development (Phase 2)

Comprehensive test suite for:
- Fine-tuning pipeline
- Quality evaluation metrics
- Dataset loading
"""

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, Mock

import pytest
from ai.models.conversation_quality_metrics import (
    ConversationQualityEvaluator,
    ConversationQualityMetrics,
)
from ai.models.therapeutic_dataset_loader import (
    DatasetSplit,
    TherapeuticConversationLoader,
)
from ai.models.therapeutic_finetuning import (
    TherapeuticConversationDataset,
    TherapeuticFinetunConfig,
)


# Test fixtures
@pytest.fixture
def sample_conversations() -> List[Dict[str, Any]]:
    """Create sample therapeutic conversations."""
    return [
        {
            "id": "conv_001",
            "messages": [
                {
                    "role": "user",
                    "content": ("I've been feeling anxious about work lately."),
                },
                {
                    "role": "assistant",
                    "content": (
                        "I understand work anxiety can be challenging. "
                        "Can you tell me what specifically makes you "
                        "anxious?"
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        "I worry about making mistakes and disappointing my team."
                    ),
                },
                {
                    "role": "assistant",
                    "content": "That sounds like a lot of pressure. "
                    "Let's explore some coping strategies together.",
                },
            ],
            "quality_score": 0.85,
            "has_crisis_signal": False,
        },
        {
            "id": "conv_002",
            "messages": [
                {
                    "role": "user",
                    "content": ("I feel hopeless and don't see a point in trying."),
                },
                {
                    "role": "assistant",
                    "content": (
                        "I'm really sorry you're feeling this hopeless, and I'm "
                        "concerned about your safety. You deserve support right "
                        "now. If you're in immediate danger, please call 911. "
                        "I can stay with you while we connect with a crisis "
                        "counselor at the 988 Suicide & Crisis Lifeline or text "
                        "741741. Is there someone you trust nearby we can "
                        "involve for support?"
                    ),
                },
            ],
            "quality_score": 0.90,
            "has_crisis_signal": True,
        },
    ]


@pytest.fixture
def tokenizer_mock():
    """Create mock tokenizer."""
    mock = MagicMock()
    mock.encode = Mock(return_value=[1, 2, 3, 4, 5])
    mock.decode = Mock(return_value="test response")
    mock.pad_token = "<pad>"
    mock.eos_token = "</s>"
    return mock


class TestTherapeuticFinetunConfig:
    """Tests for fine-tuning configuration."""

    def test_default_config(self):
        """Test default configuration."""
        config = TherapeuticFinetunConfig()

        assert config.model_name == "meta-llama/Llama-2-7b-hf"
        assert config.use_lora is True
        assert config.lora_rank == 8
        assert config.batch_size == 8
        assert config.learning_rate == 2e-4

    def test_custom_config(self):
        """Test custom configuration."""
        config = TherapeuticFinetunConfig(
            model_name="gpt2",
            batch_size=16,
            learning_rate=1e-4,
        )

        assert config.model_name == "gpt2"
        assert config.batch_size == 16
        assert config.learning_rate == 1e-4


class TestTherapeuticConversationDataset:
    """Tests for conversation dataset."""

    def test_dataset_creation(self, sample_conversations, tokenizer_mock):
        """Test dataset creation."""
        dataset = TherapeuticConversationDataset(
            sample_conversations,
            tokenizer_mock,
            max_length=512,
        )

        assert len(dataset) == 2

    def test_dataset_getitem(self, sample_conversations, tokenizer_mock):
        """Test getting items from dataset."""
        tokenizer_mock.__call__ = Mock(
            return_value={
                "input_ids": [[1, 2, 3]],
                "attention_mask": [[1, 1, 1]],
            }
        )

        dataset = TherapeuticConversationDataset(
            sample_conversations,
            tokenizer_mock,
            max_length=512,
            include_metadata=True,
        )

        item = dataset[0]

        assert "input_ids" in item
        assert "attention_mask" in item
        assert "quality_score" in item
        assert "has_crisis_signal" in item

    def test_conversation_formatting(self, tokenizer_mock):
        """Test conversation formatting."""
        conversations = [
            {
                "messages": [
                    {"role": "user", "content": "Hello"},
                    {"role": "assistant", "content": "Hi there"},
                ]
            }
        ]

        dataset = TherapeuticConversationDataset(
            conversations,
            tokenizer_mock,
            max_length=512,
        )

        formatted = dataset._format_conversation(conversations[0]["messages"])

        assert "<|patient|>" in formatted
        assert "<|therapist|>" in formatted


class TestConversationQualityEvaluator:
    """Tests for quality evaluation."""

    def test_evaluator_initialization(self):
        """Test evaluator initialization."""
        evaluator = ConversationQualityEvaluator(use_llm_evaluation=False)

        assert evaluator is not None
        assert evaluator.use_llm_evaluation is False

    def test_empathy_evaluation(self):
        """Test empathy scoring."""
        evaluator = ConversationQualityEvaluator(use_llm_evaluation=False)

        responses = [
            "I understand how you feel and I'm here to help you through this.",
            "That sounds difficult. Tell me more about what's happening.",
            "I appreciate you sharing that with me.",
        ]

        score = evaluator._evaluate_empathy(responses)

        assert 0.0 <= score <= 1.0
        assert score > 0.5  # Should have good empathy score

    def test_safety_evaluation(self):
        """Test safety scoring."""
        evaluator = ConversationQualityEvaluator(use_llm_evaluation=False)

        messages = [
            {
                "role": "user",
                "content": "I'm having suicidal thoughts",
            },
            {
                "role": "assistant",
                "content": (
                    "This is serious. Please contact emergency services "
                    "or call the 988 Suicide & Crisis Lifeline "
                    "immediately."
                ),
            },
        ]

        score = evaluator._evaluate_safety(messages)

        assert 0.0 <= score <= 1.0
        assert score > 0.7  # Should have good safety score

    def test_coherence_evaluation(self):
        """Test coherence scoring."""
        evaluator = ConversationQualityEvaluator(use_llm_evaluation=False)

        responses = [
            "I'm having trouble sleeping and focusing at work.",
            (
                "Sleep issues can affect concentration. "
                "Have you tried sleep hygiene techniques?"
            ),
            "No, what kind of techniques would help?",
        ]

        score = evaluator._evaluate_coherence(responses)

        assert 0.0 <= score <= 1.0

    def test_crisis_detection(self):
        """Test crisis signal detection."""
        evaluator = ConversationQualityEvaluator(use_llm_evaluation=False)

        messages_with_crisis = [
            {"role": "user", "content": "I want to kill myself"},
            {
                "role": "assistant",
                "content": "I'm concerned about your safety. "
                "Please reach out to the 988 Suicide & Crisis Lifeline.",
            },
        ]

        has_crisis, intervention = evaluator._detect_crisis_signals(
            messages_with_crisis
        )

        assert has_crisis is True
        assert intervention is True

    def test_technique_analysis(self):
        """Test therapeutic technique detection."""
        evaluator = ConversationQualityEvaluator(use_llm_evaluation=False)

        responses = [
            "What thoughts come up when you think about this situation?",
            "How can we challenge that belief?",
            "What would be a more helpful thought?",
        ]

        techniques, consistency = evaluator._analyze_techniques(responses)

        assert len(techniques) > 0
        assert 0.0 <= consistency <= 1.0

    def test_full_evaluation(self, sample_conversations):
        """Test full conversation evaluation."""
        evaluator = ConversationQualityEvaluator(use_llm_evaluation=False)

        metrics = evaluator.evaluate_conversation(
            sample_conversations[0]["messages"],
            include_bias_analysis=False,
        )

        assert isinstance(metrics, ConversationQualityMetrics)
        assert 0.0 <= metrics.overall_quality <= 1.0
        assert 0.0 <= metrics.empathy_score <= 1.0
        assert 0.0 <= metrics.safety_score <= 1.0


class TestTherapeuticConversationLoader:
    """Tests for dataset loading."""

    def test_loader_initialization(self):
        """Test loader initialization."""
        loader = TherapeuticConversationLoader(
            min_quality_score=0.7,
            min_conversation_length=10,
        )

        assert loader.min_quality_score == 0.7
        assert loader.min_conversation_length == 10

    def test_validate_conversation_format(self, sample_conversations):
        """Test conversation format validation."""
        loader = TherapeuticConversationLoader()

        # Valid conversation
        is_valid, error = loader.validate_conversation_format(sample_conversations[0])
        assert is_valid is True
        assert error is None

        # Invalid conversation (no messages)
        is_valid, error = loader.validate_conversation_format({})
        assert is_valid is False
        assert "messages" in error.lower()

        # Invalid role
        is_valid, error = loader.validate_conversation_format(
            {"messages": [{"role": "invalid_role", "content": "test"}]}
        )
        assert is_valid is False

    def test_filtering_conversations(self):
        """Test conversation filtering."""
        loader = TherapeuticConversationLoader(min_quality_score=0.8)

        conversations = [
            {
                "messages": [
                    {"role": "user", "content": "long content " * 10},
                    {"role": "assistant", "content": "response " * 10},
                ],
                "quality_score": 0.9,
            },
            {
                "messages": [
                    {"role": "user", "content": "short"},
                ],
                "quality_score": 0.5,  # Below threshold
            },
        ]

        filtered = loader._filter_conversations(conversations)

        assert len(filtered) == 1  # Only first conversation passes

    def test_load_from_json_files(self, sample_conversations):
        """Test loading from JSON files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write test file
            test_file = Path(tmpdir) / "conversations.json"
            with open(test_file, "w") as f:
                json.dump(sample_conversations, f)

            loader = TherapeuticConversationLoader(min_quality_score=0.7)
            conversations = loader.load_from_json_files(tmpdir)

            assert len(conversations) == 2

    def test_load_from_jsonl_file(self, sample_conversations):
        """Test loading from JSONL file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Write test file
            test_file = Path(tmpdir) / "conversations.jsonl"
            with open(test_file, "w") as f:
                f.write("\n".join(map(json.dumps, sample_conversations)))
                f.write("\n")

            loader = TherapeuticConversationLoader(min_quality_score=0.7)
            conversations = loader.load_from_jsonl_file(str(test_file))

            assert len(conversations) == 2

    def test_split_conversations(self, sample_conversations):
        """Test train/val/test split."""
        loader = TherapeuticConversationLoader()

        # Create larger dataset
        conversations = sample_conversations * 10  # 20 conversations

        split = loader.split_conversations(
            conversations,
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
        )

        assert isinstance(split, DatasetSplit)
        assert len(split.train) == 16
        assert len(split.validation) == 2
        assert len(split.test) == 2

    def test_iterate_conversations(self, sample_conversations):
        """Test batch iteration."""
        loader = TherapeuticConversationLoader()

        conversations = sample_conversations * 5  # 10 conversations

        batches = list(loader.iterate_conversations(conversations, batch_size=3))

        assert len(batches) == 4  # 3+3+3+1
        assert len(batches[0]) == 3
        assert len(batches[-1]) == 1

    def test_save_to_jsonl(self, sample_conversations):
        """Test saving to JSONL."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "output.jsonl"

            loader = TherapeuticConversationLoader()
            loader.save_to_jsonl(sample_conversations, str(output_path))

            # Verify
            with open(output_path, "r") as f:
                lines = f.readlines()
                assert len(lines) == 2

    def test_dataset_statistics(self, sample_conversations):
        """Test statistics calculation."""
        loader = TherapeuticConversationLoader()

        stats = loader.get_dataset_statistics(sample_conversations)

        assert stats["total_conversations"] == 2
        assert "avg_conversation_length" in stats
        assert "avg_quality_score" in stats
        assert "conversations_with_crisis" in stats


# Integration tests
class TestPhase2Integration:
    """Integration tests for Phase 2 components."""

    def test_dataset_to_training_pipeline(self, sample_conversations):
        """Test data flow from loading to training dataset."""
        loader = TherapeuticConversationLoader(min_quality_score=0.7)

        # Filter conversations
        filtered = loader._filter_conversations(sample_conversations)

        # Split
        split = loader.split_conversations(filtered)

        # Should have data in train split
        assert len(split.train) > 0

    def test_quality_evaluation_on_dataset(self, sample_conversations):
        """Test evaluating conversations in dataset."""
        evaluator = ConversationQualityEvaluator(use_llm_evaluation=False)

        results = list(
            map(
                lambda conv: evaluator.evaluate_conversation(
                    conv["messages"],
                    include_bias_analysis=False,
                ),
                sample_conversations,
            )
        )

        assert len(results) == 2
        assert all(0.0 <= m.overall_quality <= 1.0 for m in results)
