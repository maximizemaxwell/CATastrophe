# src/catastrophe/predict.py
"""
Predict module: Load model and vectorizer, predict scores for new texts.
Print the anomalies with scores above a threshold.
"""

import argparse
import torch
from .config import MODEL_WEIGHTS_PATH
from .features.vectorizer import TFIDFVectorizerWrapper
from .model.autoencoder import Autoencoder


def predict_score(message: str, func: str) -> float:
    """
    Predict the anomaly score for a given message and function.
    """
    # Check for GPU availability
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load the vectorizer
    vectorizer = TFIDFVectorizerWrapper.load_from_file()

    # Text merge and transform
    input_text = message + " <SEP > " + func
    vec = vectorizer.transform([input_text])
    x = torch.tensor(vec.toarray(), dtype=torch.float32).to(device)

    # Load the Autoencoder model
    input_dim = vec.shape[1]
    model = Autoencoder(input_dim=input_dim)
    model.load_state_dict(torch.load(MODEL_WEIGHTS_PATH, map_location=device))
    model = model.to(device)
    model.eval()

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
