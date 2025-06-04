# CATastrophe

A machine learning-based vulnerability detector for Python code using autoencoder neural networks.

## Overview

CATastrophe uses deep learning to identify potential security vulnerabilities in Python source code by training an autoencoder model to recognize patterns associated with vulnerable code segments. The system analyzes code through TF-IDF vectorization and neural network anomaly detection.

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd catastrophe

# Install dependencies
# For CPU-only (default)
pip install -r requirements.txt

# For GPU with CUDA 11.8
pip install -r requirements-gpu.txt

# For GPU with CUDA 12.1
pip install -r requirements-gpu-cu121.txt

# Set up environment (required for training data access)
cp .env.example .env
# Edit .env and add your HUGGINGFACE_TOKEN
```

## Environment Setup

Create a `.env` file with the following variables:

```bash
# Required for training data access and model publishing
HUGGINGFACE_TOKEN=your_hf_token_here

# Optional: For GitHub bot functionality
GITHUB_TOKEN=your_github_token
GITHUB_WEBHOOK_SECRET=your_webhook_secret
HF_REPO_ID=your_username/catastrophe-model
```

## Usage

### Training the Model

```bash
# Using Docker (recommended for deployment)
docker-compose --profile train up --build

# Local training (recommended for development)
python3 run_training.py

# Alternative methods
PYTHONPATH=./src python3 -m catastrophe.train
cd src && python3 -m catastrophe.train
```

### Analyzing Code for Vulnerabilities

```bash
# Analyze a single code snippet
PYTHONPATH=./src python3 -m catastrophe.predict \
  --message "commit message" \
  --func "def vulnerable_function(): sql = 'SELECT * FROM users WHERE id=' + user_id"

# From src directory
cd src && python3 -m catastrophe.predict \
  --message "database query" \
  --func "def get_user(id): return db.execute(f'SELECT * FROM users WHERE id={id}')"
```

### Running the GitHub Bot

```bash
# Using Docker Compose
docker-compose up -d github-bot

# Check bot health
curl http://localhost:8080/health

# View logs
docker-compose logs -f github-bot
```

### GPU Support

The system automatically detects and uses GPU when available:

- **CUDA Support**: Automatically uses GPU for training and inference
- **Memory Optimization**: Includes automatic cache clearing and memory management
- **CPU Fallback**: Gracefully falls back to CPU if GPU is unavailable

**Important**: The default `requirements.txt` installs CPU-only PyTorch. For GPU support, install using the appropriate GPU requirements file:

```bash
# Check your CUDA version
nvidia-smi

# Install GPU-enabled PyTorch
pip install -r requirements-gpu.txt      # For CUDA 11.8
pip install -r requirements-gpu-cu121.txt  # For CUDA 12.1

# Verify GPU is available
python3 test_gpu.py
```

## Features

- **Autoencoder-based Detection**: Uses neural networks to learn vulnerability patterns
- **Enhanced Architecture**: Dropout and batch normalization for improved performance
- **GitHub Integration**: Automated PR analysis via GitHub bot
- **Docker Support**: Easy deployment with Docker Compose
- **Hugging Face Hub**: Model sharing and versioning
- **GPU Acceleration**: Automatic CUDA detection and utilization
- **Early Stopping**: Prevents overfitting during training

## Model Architecture

- **Input Layer**: TF-IDF vectorized code features (2000 dimensions)
- **Encoder**: 1024 → 512 → 256 → 128 neurons with dropout (0.2) and batch normalization
- **Decoder**: 128 → 256 → 512 → 1024 → output neurons
- **Loss Function**: Mean Squared Error (MSE)
- **Optimizer**: Adam with weight decay (1e-5)

## Configuration

Key configuration options in `src/catastrophe/config.py`:

- `BATCH_SIZE = 32`: Training batch size
- `EPOCHS = 50`: Maximum training epochs
- `LEARNING_RATE = 1e-3`: Learning rate for Adam optimizer
- `MAX_FEATURES = 2000`: TF-IDF vocabulary size
- `EARLY_STOPPING_PATIENCE = 5`: Early stopping patience

## Testing

The project includes comprehensive tests with realistic C code examples:

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest

# Run tests with coverage report
pytest --cov=catastrophe --cov-report=html

# Run specific test categories
pytest tests/test_vectorizer.py        # Vectorizer tests
pytest tests/test_model.py             # Model architecture tests  
pytest tests/test_prediction.py        # Prediction functionality tests
pytest tests/test_integration.py       # End-to-end integration tests

# Run tests in verbose mode
pytest -v

# Run tests and stop on first failure
pytest -x
```

### Test Data

The test suite includes realistic C code examples:

- **Vulnerable C commits** (`tests/test_data/vulnerable_c_commits.json`):
  - Buffer overflow vulnerabilities (`strcpy`, `gets`, `sprintf`)
  - SQL injection patterns
  - Format string vulnerabilities
  - Command injection risks
  - Memory management issues

- **Safe C commits** (`tests/test_data/safe_c_commits.json`):
  - Secure string handling (`strncpy`, `fgets`, `snprintf`)
  - Prepared statements for SQL
  - Input validation and bounds checking
  - Proper memory management

### Test Coverage

Tests cover:
- **Unit Tests**: Individual component functionality
- **Integration Tests**: Complete pipeline from code to vulnerability detection
- **Edge Cases**: Empty inputs, malformed data, large datasets
- **Model Tests**: Neural network training, saving/loading, batch processing
- **Vectorizer Tests**: TF-IDF transformation and persistence

## Linting and Code Quality

```bash
# Run linting (must pass with score ≥ 8.0)
pylint $(git ls-files '*.py') --fail-under=8.0
```

## Documentation

For detailed documentation, see the [docs/](docs/) directory or build the mdBook:

```bash
cd docs
mdbook serve
```

## License

MIT License - see [LICENSE](LICENSE) file for details.
