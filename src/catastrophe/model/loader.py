# src/catastrophe/model/loader.py
"""
Model loader utilities for loading from HuggingFace Hub or local filesystem
"""

import os
import logging
import pickle
import torch
from pathlib import Path
from huggingface_hub import hf_hub_download
from typing import Tuple, Optional

from ..config import HF_MODEL_REPO, MODEL_WEIGHTS_PATH, VECTORIZER_PATH, MAX_FEATURES
from ..features.vectorizer import TFIDFVectorizerWrapper
from .autoencoder import Autoencoder


def load_model_from_hub(
    repo_id: str = HF_MODEL_REPO,
    token: Optional[str] = None,
    device: Optional[torch.device] = None,
) -> Tuple[Autoencoder, TFIDFVectorizerWrapper]:
    """
    Load model and vectorizer from Hugging Face Hub

    Args:
        repo_id: HuggingFace repository ID
        token: HuggingFace token (optional)
        device: Device to load model on

    Returns:
        Tuple of (model, vectorizer)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        logging.info(f"Loading model from Hugging Face: {repo_id}")

        # Download files from Hugging Face
        model_path = hf_hub_download(
            repo_id=repo_id, filename="catastrophe_model.pth", token=token
        )
        vectorizer_path = hf_hub_download(
            repo_id=repo_id, filename="vectorizer.pkl", token=token
        )

        # Load vectorizer
        wrapper = TFIDFVectorizerWrapper()
        with open(vectorizer_path, "rb") as f:
            wrapper.vectorizer = pickle.load(f)

        # Get input dimension from vectorizer
        input_dim = wrapper.vectorizer.get_feature_names_out().shape[0]

        # Load model
        model = Autoencoder(input_dim=input_dim)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()

        logging.info("Model loaded successfully from HuggingFace")
        return model, wrapper

    except Exception as e:
        logging.error(f"Failed to load model from HuggingFace: {str(e)}")
        raise


def load_model_from_local(
    model_path: Path = MODEL_WEIGHTS_PATH,
    vectorizer_path: Path = VECTORIZER_PATH,
    device: Optional[torch.device] = None,
) -> Tuple[Autoencoder, TFIDFVectorizerWrapper]:
    """
    Load model and vectorizer from local filesystem

    Args:
        model_path: Path to model weights
        vectorizer_path: Path to vectorizer
        device: Device to load model on

    Returns:
        Tuple of (model, vectorizer)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        logging.info("Loading model from local filesystem")

        # Load vectorizer
        wrapper = TFIDFVectorizerWrapper()
        with open(vectorizer_path, "rb") as f:
            wrapper.vectorizer = pickle.load(f)

        # Get input dimension from vectorizer
        input_dim = wrapper.vectorizer.get_feature_names_out().shape[0]

        # Load model
        model = Autoencoder(input_dim=input_dim)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model = model.to(device)
        model.eval()

        logging.info("Model loaded successfully from local filesystem")
        return model, wrapper

    except Exception as e:
        logging.error(f"Failed to load model from local filesystem: {str(e)}")
        raise


def load_model(
    prefer_hub: bool = True, device: Optional[torch.device] = None
) -> Tuple[Autoencoder, TFIDFVectorizerWrapper]:
    """
    Load model with fallback strategy

    Args:
        prefer_hub: Whether to prefer loading from HuggingFace Hub
        device: Device to load model on

    Returns:
        Tuple of (model, vectorizer)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Get HuggingFace token
    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")

    if prefer_hub:
        try:
            return load_model_from_hub(token=token, device=device)
        except Exception as e:
            logging.warning(
                f"Failed to load from HuggingFace, falling back to local: {e}"
            )
            if MODEL_WEIGHTS_PATH.exists() and VECTORIZER_PATH.exists():
                return load_model_from_local(device=device)
            else:
                raise RuntimeError(
                    "No model available: HuggingFace download failed and no local model found"
                )
    else:
        if MODEL_WEIGHTS_PATH.exists() and VECTORIZER_PATH.exists():
            return load_model_from_local(device=device)
        else:
            logging.info("Local model not found, attempting to load from HuggingFace")
            return load_model_from_hub(token=token, device=device)
