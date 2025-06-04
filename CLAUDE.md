# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CATastrophe is an autoencoder-based code vulnerability detector for Python code analysis. It uses machine learning (PyTorch) to detect potential vulnerabilities in Python code by training an autoencoder model on code features.

## Architecture

The project follows a standard ML project structure:

- **Feature Extraction**: Code is converted to numerical features using vectorization (`src/catastrphe/features/vectorizer.py`)
- **Model**: Autoencoder neural network with encoder (input → 512 → 128) and decoder (128 → 512 → input) layers
- **Training Pipeline**: `train.py` handles model training with configurable hyperparameters
- **Inference**: `predict.py` for making predictions on new code
- **Evaluation**: `evaluate.py` for model performance assessment

Key configuration is centralized in `src/catastrphe/config.py`:
- Data path: `data/dataset.json`
- Model weights: `hf_models/autoencoder_weights.pth`
- Vectorizer: `hf_models/vectorizer.pkl`
- Training params: batch_size=32, epochs=10, learning_rate=1e-3, max_features=2000

## Development Commands

### Linting
```bash
pylint $(git ls-files '*.py') --fail-under=8.0
```

### Training Model
```bash
# Using Docker
docker-compose --profile train up --build

# Locally
python -m catastrphe.train
```

### Running GitHub Bot
```bash
# Using Docker Compose
docker-compose up -d github-bot

# Check bot health
curl http://localhost:8080/health
```

### Dependencies
- Main project: See requirements.txt
- GitHub bot: See github_bot/requirements.txt
- Key packages: PyTorch, scikit-learn, huggingface-hub, PyGithub, Flask

## Important Notes

- Enhanced model architecture with dropout and batch normalization for better performance
- Model automatically publishes to Hugging Face Hub after training
- GitHub bot analyzes PR commits and posts vulnerability reports
- Docker Compose setup for easy deployment of both training and bot
- Early stopping implemented to prevent overfitting
- Vulnerability threshold configurable via environment variable