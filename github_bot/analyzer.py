"""
Vulnerability Analyzer using the trained CATastrophe model
"""

import os
import torch
import pickle
import logging
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.catastrphe.model.autoencoder import Autoencoder
from src.catastrphe.config import MAX_FEATURES

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
            repo_id = os.getenv("HF_REPO_ID")
            if not repo_id:
                raise ValueError("HF_REPO_ID environment variable is required")
            
            logging.info(f"Loading model from Hugging Face: {repo_id}")
            
            # Download files from Hugging Face
            model_path = hf_hub_download(repo_id=repo_id, filename="catastrophe_model.pth")
            vectorizer_path = hf_hub_download(repo_id=repo_id, filename="vectorizer.pkl")
            
            # Load vectorizer
            with open(vectorizer_path, 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            # Load model
            self.model = Autoencoder(input_dim=MAX_FEATURES)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.to(self.device)
            self.model.eval()
            
            logging.info("Model loaded successfully")
            
        except Exception as e:
            logging.error(f"Failed to load model: {str(e)}")
            # Fallback to local model if available
            self._load_local_model()
    
    def _load_local_model(self):
        """Load model from local filesystem as fallback"""
        try:
            local_model_path = "../hf_model/autoencoder_weights.pth"
            local_vectorizer_path = "../hf_model/vectorizer.pkl"
            
            if os.path.exists(local_model_path) and os.path.exists(local_vectorizer_path):
                logging.info("Loading local model as fallback")
                
                with open(local_vectorizer_path, 'rb') as f:
                    self.vectorizer = pickle.load(f)
                
                self.model = Autoencoder(input_dim=MAX_FEATURES)
                self.model.load_state_dict(torch.load(local_model_path, map_location=self.device))
                self.model.to(self.device)
                self.model.eval()
                
                logging.info("Local model loaded successfully")
            else:
                logging.error("No local model found")
                
        except Exception as e:
            logging.error(f"Failed to load local model: {str(e)}")
    
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
            features_tensor = torch.tensor(features, dtype=torch.float32).to(self.device)
            
            # Get reconstruction
            with torch.no_grad():
                reconstructed = self.model(features_tensor)
                
                # Calculate anomaly score (MSE)
                anomaly_score = torch.mean((features_tensor - reconstructed) ** 2, dim=1)
                
                # Convert to Python float
                score = anomaly_score.item()
            
            return score
            
        except Exception as e:
            logging.error(f"Error analyzing code: {str(e)}")
            raise
