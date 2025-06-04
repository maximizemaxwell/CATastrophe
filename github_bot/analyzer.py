"""
Vulnerability Analyzer using the trained CATastrophe model
"""

import os
import torch
import logging
from dotenv import load_dotenv
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.catastrophe.model.loader import load_model
from src.catastrophe.config import HF_MODEL_REPO

load_dotenv()


class VulnerabilityAnalyzer:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._load_model()

    def _load_model(self):
        """Load model and vectorizer from Hugging Face Hub"""
        try:
            # Use HF_REPO_ID from environment or default to configured repo
            repo_id = os.getenv("HF_REPO_ID", HF_MODEL_REPO)

            logging.info(f"Loading model from Hugging Face: {repo_id}")

            # Load model and vectorizer using centralized loader
            self.model, self.vectorizer = load_model(
                prefer_hub=True, device=self.device
            )

            logging.info("Model loaded successfully")

        except Exception as e:
            logging.error(f"Failed to load model: {str(e)}")
            raise RuntimeError("Could not load model from any source")

    def is_loaded(self):
        """Check if model is loaded"""
        return self.model is not None and self.vectorizer is not None

    def analyze(self, code_text: str) -> float:
        """
        Analyze code text and return vulnerability score
        Higher score indicates higher likelihood of vulnerability
        """
        if not self.is_loaded():
            raise RuntimeError("Model not loaded")

        try:
            # Vectorize the input
            features = self.vectorizer.transform([code_text]).toarray()
            features_tensor = torch.tensor(features, dtype=torch.float32).to(
                self.device
            )

            # Get reconstruction
            with torch.no_grad():
                reconstructed = self.model(features_tensor)

                # Calculate anomaly score (MSE)
                anomaly_score = torch.mean(
                    (features_tensor - reconstructed) ** 2, dim=1
                )

                # Convert to Python float
                score = anomaly_score.item()

            return score

        except Exception as e:
            logging.error(f"Error analyzing code: {str(e)}")
            raise
