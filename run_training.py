#!/usr/bin/env python3
"""Run the training script with proper Python path setup."""

import sys
import os

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

# Now import and run the training
from catastrophe import train

if __name__ == "__main__":
    train()
