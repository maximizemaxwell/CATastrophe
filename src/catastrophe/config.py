# src/catastrphe/config.py
"""
CATastrophe project configuration.
Defines model path, data path as constants
"""

from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Data Configuration
DATA_PATH = BASE_DIR / "data" / "dataset.json"  # Local fallback
HF_DATASET_REPO = "ewhk9887/commit-vulnerability"  # Hugging Face dataset repository
HF_DATASET_NAME = "default"  # Dataset configuration name

# Model save path
MODEL_DIR = BASE_DIR / "hf_model"
MODEL_WEIGHTS_PATH = MODEL_DIR / "autoencoder_weights.pth"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"

# Training configuration
MAX_FEATURES = 2000  # Maximum number of features for vectorization
BATCH_SIZE = 256  # Batch size for Training
EPOCHS = 50  # Maximum number of epochs for Training
LEARNING_RATE = 1e-3  # Learning rate for the optimizer
EARLY_STOPPING_PATIENCE = 5  # Early stopping patience
MIN_DELTA = 1e-4  # Minimum improvement for early stopping
