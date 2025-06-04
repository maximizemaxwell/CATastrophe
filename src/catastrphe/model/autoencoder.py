# src/catastrophe/model/autoencoder.py

"""
Autoencoder configuration module.
Gets input_dim and implementation of encoder and decoder
"""

import torch
from torch import nn


class Autoencoder(nn.Module):
    """
    Enhanced Autoencoder model with dropout and batch normalization for better performance.
    """

    def __init__(self, input_dim: int, dropout_rate: float = 0.2):
        """
        Initialize the Autoencoder model with improved architecture

        Args:
            input_dim: Dimension of input features
            dropout_rate: Dropout rate for regularization
        """
        super().__init__()

        # Enhanced encoder with batch normalization and dropout
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # Enhanced decoder (symmetric to encoder)
        self.decoder = nn.Sequential(
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(1024, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the autoencoder"""
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Get the latent representation (embedding) of input"""
        return self.encoder(x)
