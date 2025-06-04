# src/catastrophe/evaluate.py
"""
Evaluate the trained model
- Load test dataset
- Calculate each predict_score()
"""

import argparse
import json
import logging
import os
import torch
import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
)
from tqdm import tqdm
from dotenv import load_dotenv
from datasets import load_dataset

from .config import DATA_PATH, HF_DATASET_REPO, HF_DATASET_NAME, BASE_DIR
from .model.loader import load_model

# Load .env from project root
load_dotenv(BASE_DIR / ".env")

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def load_test_data():
    """
    Load test dataset from Hugging Face or local file
    """
    try:
        # Get HuggingFace token from environment
        token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
        if not token:
            logging.warning("HF_TOKEN not found, falling back to local dataset")
            return load_test_data_local()

        logging.info(f"Loading test dataset from Hugging Face: {HF_DATASET_REPO}")

        # Load dataset from HF Hub
        dataset = load_dataset(
            HF_DATASET_REPO,
            HF_DATASET_NAME,
            token=token,
            trust_remote_code=True,
            split="train",  # Use train split and take last 20% for testing
        )

        # Extract test data (use last 20% for testing)
        dataset_size = len(dataset)
        test_start = int(dataset_size * 0.8)

        test_data = []
        for i, item in enumerate(dataset):
            if i >= test_start:  # Only use last 20% for testing
                test_data.append(
                    {
                        "message": item["message"],
                        "func": item["func"],
                        "label": item.get(
                            "label", 1
                        ),  # Assuming 1 for vulnerable if not specified
                    }
                )

        logging.info(f"Loaded {len(test_data)} test samples from Hugging Face")
        return test_data

    except Exception as e:
        logging.error(f"Failed to load from Hugging Face: {e}")
        logging.info("Falling back to local dataset")
        return load_test_data_local()


def load_test_data_local():
    """
    Load test dataset from local file
    """
    test_data = []
    try:
        with open(DATA_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
            # Use last 20% of data for testing
            test_lines = lines[int(len(lines) * 0.8) :]

            for line in test_lines:
                if not line.strip():
                    continue
                item = json.loads(line)
                test_data.append(
                    {
                        "message": item["message"],
                        "func": item["func"],
                        "label": item.get(
                            "label", 1
                        ),  # Assuming 1 for vulnerable if not specified
                    }
                )

        logging.info(f"Loaded {len(test_data)} test samples from local dataset")
    except FileNotFoundError:
        logging.error(f"Local dataset not found at {DATA_PATH}")
        raise
    return test_data


def evaluate_model(threshold=0.5):
    """
    Evaluate the model on test data
    """
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")

    # Load model and vectorizer
    logging.info("Loading model...")
    model, vectorizer = load_model(prefer_hub=True, device=device)

    # Load test data
    logging.info("Loading test data...")
    test_data = load_test_data()

    if not test_data:
        logging.error("No test data available")
        return

    # Calculate anomaly scores for all test samples
    logging.info("Calculating anomaly scores...")
    scores = []
    labels = []

    for item in tqdm(test_data, desc="Evaluating"):
        # Combine message and func
        input_text = item["message"] + " <SEP > " + item["func"]

        # Transform and predict
        vec = vectorizer.transform([input_text])
        x = torch.tensor(vec.toarray(), dtype=torch.float32).to(device)

        with torch.no_grad():
            reconstructed = model(x)
            anomaly_score = torch.mean((x - reconstructed) ** 2).item()

        scores.append(anomaly_score)
        labels.append(item["label"])

    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Convert to numpy arrays
    scores = np.array(scores)
    labels = np.array(labels)

    # Calculate metrics
    logging.info("\nEvaluation Results:")
    logging.info("-" * 50)

    # Basic statistics
    logging.info(f"Number of samples: {len(scores)}")
    logging.info(f"Number of vulnerable samples: {np.sum(labels)}")
    logging.info(f"Number of benign samples: {len(labels) - np.sum(labels)}")

    # Score statistics
    logging.info(f"\nAnomaly Score Statistics:")
    logging.info(f"Mean score (all): {np.mean(scores):.4f}")
    logging.info(f"Mean score (vulnerable): {np.mean(scores[labels == 1]):.4f}")
    logging.info(f"Mean score (benign): {np.mean(scores[labels == 0]):.4f}")
    logging.info(f"Std score (all): {np.std(scores):.4f}")

    # Binary classification metrics at given threshold
    predictions = (scores > threshold).astype(int)
    tp = np.sum((predictions == 1) & (labels == 1))
    tn = np.sum((predictions == 0) & (labels == 0))
    fp = np.sum((predictions == 1) & (labels == 0))
    fn = np.sum((predictions == 0) & (labels == 1))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = (
        2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    )
    accuracy = (tp + tn) / len(labels)

    logging.info(f"\nClassification Metrics (threshold={threshold}):")
    logging.info(f"Accuracy: {accuracy:.4f}")
    logging.info(f"Precision: {precision:.4f}")
    logging.info(f"Recall: {recall:.4f}")
    logging.info(f"F1-Score: {f1:.4f}")

    # ROC AUC (if we have both classes)
    if len(np.unique(labels)) > 1:
        auc = roc_auc_score(labels, scores)
        avg_precision = average_precision_score(labels, scores)
        logging.info(f"\nRanking Metrics:")
        logging.info(f"ROC AUC: {auc:.4f}")
        logging.info(f"Average Precision: {avg_precision:.4f}")

        # Find optimal threshold
        precisions, recalls, thresholds = precision_recall_curve(labels, scores)
        f1_scores = 2 * precisions * recalls / (precisions + recalls + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = (
            thresholds[optimal_idx] if optimal_idx < len(thresholds) else threshold
        )

        logging.info(f"\nOptimal threshold (by F1): {optimal_threshold:.4f}")
        logging.info(f"F1 at optimal threshold: {f1_scores[optimal_idx]:.4f}")

    logging.info("-" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate the CATastrophe model on test data"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Anomaly score threshold for classification (default: 0.5)",
    )
    args = parser.parse_args()

    evaluate_model(threshold=args.threshold)
