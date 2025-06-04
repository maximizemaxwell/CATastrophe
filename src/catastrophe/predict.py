# src/catastrophe/predict.py
"""
Predict module: Load model and vectorizer, predict scores for new texts.
Print the anomalies with scores above a threshold.
"""

import argparse
import torch
from .model.loader import load_model


def predict_score(message: str, func: str) -> float:
    """
    Predict the anomaly score for a given message and function.
    """
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model and vectorizer (preferring HuggingFace Hub)
    model, vectorizer = load_model(prefer_hub=True, device=device)

    # Text merge and transform
    input_text = message + " <SEP > " + func
    vec = vectorizer.transform([input_text])
    x = torch.tensor(vec.toarray(), dtype=torch.float32).to(device)

    # Predict the reconstruction
    with torch.no_grad():
        reconstructed = model(x)
        anomaly_score = torch.mean((x - reconstructed) ** 2).item()

    # Clear GPU cache if used
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return anomaly_score


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Predict anomaly score for a message and function."
    )
    parser.add_argument("--message", type=str, help="The message text to analyze.")
    parser.add_argument("--func", type=str, help="The function text to analyze.")
    args = parser.parse_args()

    # Predict the score
    score = predict_score(args.message, args.func)
    print(f"Anomaly Score: {score:.4f}")

    # Define a threshold for anomalies
    threshold = 0.5  # Example threshold, adjust as needed
    if score > threshold:
        print("Anomaly detected!")
    else:
        print("No anomaly detected.")
