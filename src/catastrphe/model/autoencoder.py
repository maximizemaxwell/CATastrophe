# src/catastrophe/model/autoencoder.py

"""
Autoencoder configuration module.
Gets input_dim and implementation of encoder and decoder
"""

import torch.nn as nn

class Autoencoder(nn.Module):
    """
    Autoencoder model implementation.
    """
    def __init__(self, input_dim: int):
        """
        Initialize the Autoencoder model
        """
        super(Autoencoder, self).__init__()
        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(128, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # input x to hidden layer and restore 
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed
