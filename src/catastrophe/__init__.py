"""
CATastrophe - Autoencoder-based code vulnerability detector for Python.
"""

__version__ = "0.1.0"
__author__ = "Max"

# Make key modules easily accessible
from .predict import predict_score
from .train import train

__all__ = ["predict_score", "train", "__version__"]