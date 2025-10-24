# Pixelated Empathy

Therapeutic AI system with Tim Fletcher's communication style using Mixture of Experts architecture.

## Project Structure

```
src/
├── voice_extraction/    # Style analysis and pattern extraction
├── training/            # Model training and fine-tuning
├── api/                # FastAPI server and endpoints
├── database/           # Database models and operations
└── monitoring/         # Metrics and health monitoring

data/
├── transcripts/        # Raw transcript files
├── processed/          # Processed training data
└── models/            # Trained model artifacts

config/                # Configuration files
tests/                 # Test suite
logs/                  # Application logs
```

## Setup

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Start development server
uv run python -m src.api.main
```

## Development Status

See BLUEPRINT.md for detailed development plan and current progress.
