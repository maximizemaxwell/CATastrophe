# src/catastrphe/config.py
"""
CATastrophe project configuration.
Defines model path, data path as constants
"""

from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Data Path
DATA_PATH = BASE_DIR / "data" / "dataset.json"

# Model save path
MODEL_DIR = BASE_DIR / "hf_models"
MODEL_WEIGHTS_PATH = MODEL_DIR / "autoencoder_weights.pth"
VECTORIZER_PATH = MODEL_DIR / "vectorizer.pkl"

# Training configuration
MAX_FEATURES = 2000  # Maximum number of features for vectorization
BATCH_SIZE = 32  # Batch size for Training
EPOCHS = 10  # Number of epochs for Training
LEARNING_RATE = 1e-3  # Learning rate for the optimizer
