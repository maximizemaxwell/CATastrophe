# CATastrophe

A machine learning-based vulnerability detector for Python code using autoencoder neural networks.

## Overview

CATastrophe uses deep learning to identify potential security vulnerabilities in Python source code by training an autoencoder model to recognize patterns associated with vulnerable code segments.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment (required for training data access)
cp .env.example .env
# Edit .env and add your HUGGINGFACE_TOKEN

# Train the model
python3 run_training.py

# Analyze code for vulnerabilities
PYTHONPATH=./src python3 -m catastrophe.predict --message "test" --func "def test(): pass"
```

## Features

- **Autoencoder-based Detection**: Uses neural networks to learn vulnerability patterns
- **GitHub Integration**: Automated PR analysis via GitHub bot
- **Docker Support**: Easy deployment with Docker Compose
- **Hugging Face Hub**: Model sharing and versioning

## Documentation

For detailed documentation, see the [docs/](docs/) directory or build the mdBook:

```bash
cd docs
mdbook serve
```

## License

MIT License - see [LICENSE](LICENSE) file for details.
