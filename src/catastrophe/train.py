# src/catastrophe/train.py
"""
- load dataset
- train and save TF-IDF vectorizer
- Autoencoder training
- publish model to Hugging Face Hub
"""

import json
import logging
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from dotenv import load_dotenv
from huggingface_hub import HfApi
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from .config import (
    DATA_PATH,
    HF_DATASET_REPO,
    HF_DATASET_NAME,
    MODEL_WEIGHTS_PATH,
    VECTORIZER_PATH,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    MAX_FEATURES,
    EARLY_STOPPING_PATIENCE,
    MIN_DELTA,
    BASE_DIR,
)
from .features.vectorizer import TFIDFVectorizerWrapper
from .model.autoencoder import Autoencoder

# Load .env from project root
load_dotenv(BASE_DIR / ".env")
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_texts_from_hf():
    """
    Load dataset from Hugging Face Hub
    """
    try:
        # Get HuggingFace token from environment
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        if not token:
            logging.warning("HF_TOKEN not found, falling back to local dataset")
            return load_texts_from_local()
        
        logging.info(f"Loading dataset from Hugging Face: {HF_DATASET_REPO}")
        
        # Load dataset from HF Hub
        dataset = load_dataset(
            HF_DATASET_REPO, 
            HF_DATASET_NAME,
            token=token,
            trust_remote_code=True
        )
        
        # Extract training data
        train_data = dataset['train']
        
        # Combine message and func columns
        texts = []
        for item in train_data:
            combined = item["message"] + " <SEP > " + item["func"]
            texts.append(combined)
        
        logging.info(f"Loaded {len(texts)} samples from Hugging Face dataset")
        return texts
        
    except Exception as e:
        logging.error(f"Failed to load from Hugging Face: {e}")
        logging.info("Falling back to local dataset")
        return load_texts_from_local()


def load_texts_from_local():
    """
    Load dataset from local JSON file (fallback)
    """
    texts = []
    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                item = json.loads(line)
                combined = item["message"] + " <SEP > " + item["func"]
                texts.append(combined)
        logging.info(f"Loaded {len(texts)} samples from local dataset")
    except FileNotFoundError:
        logging.error(f"Local dataset not found at {DATA_PATH}")
        raise
    return texts


def load_texts():
    """
    Load dataset, preferring Hugging Face Hub over local files
    """
    return load_texts_from_hf()


def train():
    """
    Train the model
    """
    # Load dataset from Hugging Face Hub (with local fallback)
    texts = load_texts()

    # TF-IDF Vectorizer training, save
    logging.info("Training TF-IDF Vectorizer with max features: %d", MAX_FEATURES)
    vect_wrapper = TFIDFVectorizerWrapper()
    vect_wrapper.fit(texts)
    vect_wrapper.save(VECTORIZER_PATH)
    X = vect_wrapper.transform(texts).toarray()

    # Split data into train and validation sets
    X_train, X_val = train_test_split(X, test_size=0.2, random_state=42)

    # Convert to PyTorch Tensors
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    X_val_tensor = torch.tensor(X_val, dtype=torch.float32)

    train_dataset = TensorDataset(X_train_tensor)
    val_dataset = TensorDataset(X_val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Autoencoder model with enhanced architecture
    model = Autoencoder(input_dim=X.shape[1], dropout_rate=0.2)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    criterion = nn.MSELoss()

    # Training loop with early stopping
    logging.info("Starting training...")
    best_val_loss = float("inf")
    patience_counter = 0

    for epoch in range(EPOCHS):
        # Training phase
        model.train()
        train_losses = []
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} / {EPOCHS}"):
            inputs = batch[0]
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())

        # Validation phase
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                inputs = batch[0]
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                val_losses.append(loss.item())

        avg_train_loss = sum(train_losses) / len(train_losses)
        avg_val_loss = sum(val_losses) / len(val_losses)

        logging.info(
            "Epoch %d - Train Loss: %.4f, Val Loss: %.4f",
            epoch + 1,
            avg_train_loss,
            avg_val_loss,
        )

        # Learning rate scheduling
        scheduler.step(avg_val_loss)

        # Early stopping
        if avg_val_loss < best_val_loss - MIN_DELTA:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(model.state_dict(), MODEL_WEIGHTS_PATH.with_suffix(".best.pth"))
        else:
            patience_counter += 1
            if patience_counter >= EARLY_STOPPING_PATIENCE:
                logging.info("Early stopping triggered at epoch %d", epoch + 1)
                break

    # Load best model if it exists
    best_model_path = MODEL_WEIGHTS_PATH.with_suffix(".best.pth")
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path))
        logging.info("Loaded best model from training")

    # Save the final model weights
    logging.info("Saving model weights locally...")
    MODEL_WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_WEIGHTS_PATH)
    logging.info("Model saved to %s", MODEL_WEIGHTS_PATH)

    # Publish to Hugging Face Hub
    if os.getenv("HF_TOKEN") and os.getenv("HF_REPO_ID"):
        publish_to_huggingface(model, vect_wrapper, best_val_loss)
    else:
        logging.warning(
            "HF_TOKEN or HF_REPO_ID not found in environment. "
            "Skipping HuggingFace upload."
        )

    return model, vect_wrapper


def publish_to_huggingface(model, vect_wrapper, final_loss):
    """
    Publish trained model to Hugging Face Hub
    """
    try:
        api = HfApi()
        token = os.getenv("HF_TOKEN")
        repo_id = os.getenv("HF_REPO_ID")

        logging.info("Publishing model to Hugging Face Hub: %s", repo_id)

        # Create repository if it doesn't exist
        api.create_repo(repo_id=repo_id, exist_ok=True, token=token)

        # Save model and vectorizer
        model_path = "catastrophe_model.pth"
        vectorizer_path = "vectorizer.pkl"

        torch.save(model.state_dict(), model_path)
        vect_wrapper.save(vectorizer_path)

        # Create model card
        model_card_content = f"""
---
language: en
tags:
- vulnerability-detection
- code-analysis
- autoencoder
- anomaly-detection
library_name: pytorch
metrics:
- mse
---

# CATastrophe - Code Vulnerability Detector

This model is an autoencoder-based vulnerability detector for Python code. It uses TF-IDF
vectorization and an autoencoder architecture to detect anomalies in code that may indicate
vulnerabilities.

## Model Details

- **Architecture**: Autoencoder (Input → 512 → 128 → 512 → Input)
- **Input Features**: {MAX_FEATURES} (TF-IDF)
- **Training Loss**: {final_loss:.4f}
- **Framework**: PyTorch

## Usage

```python
import torch
import pickle
from model import Autoencoder

# Load model
model = Autoencoder(input_dim={MAX_FEATURES})
model.load_state_dict(torch.load('catastrophe_model.pth'))
model.eval()

# Load vectorizer
with open('vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Analyze code
code_text = "your code here"
features = vectorizer.transform([code_text]).toarray()
features_tensor = torch.tensor(features, dtype=torch.float32)

with torch.no_grad():
    reconstructed = model(features_tensor)
    anomaly_score = torch.mean((features_tensor - reconstructed) ** 2, dim=1)
```

## Training Configuration

- Batch Size: {BATCH_SIZE}
- Epochs: {EPOCHS}
- Learning Rate: {LEARNING_RATE}
- Optimizer: Adam

## Limitations

This model is trained on vulnerable commits only and uses reconstruction error as an
anomaly score. High scores indicate potential vulnerabilities, but manual review is
recommended.
"""

        with open("README.md", "w", encoding="utf-8") as f:
            f.write(model_card_content)

        # Upload files
        api.upload_file(
            path_or_fileobj=model_path,
            path_in_repo="catastrophe_model.pth",
            repo_id=repo_id,
            token=token,
        )

        api.upload_file(
            path_or_fileobj=vectorizer_path,
            path_in_repo="vectorizer.pkl",
            repo_id=repo_id,
            token=token,
        )

        api.upload_file(
            path_or_fileobj="README.md",
            path_in_repo="README.md",
            repo_id=repo_id,
            token=token,
        )

        # Clean up temporary files
        os.remove(model_path)
        os.remove(vectorizer_path)
        os.remove("README.md")

        logging.info(
            "Model successfully published to: https://huggingface.co/%s", repo_id
        )

    except Exception as e:
        logging.error("Failed to publish to Hugging Face: %s", str(e))
        raise


if __name__ == "__main__":
    train()
