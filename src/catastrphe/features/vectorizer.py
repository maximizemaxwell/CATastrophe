# src/catastrophe/features/vectorizer.py
"""
Vectorizer module: TF-IDF Vectorization
"""

import pickle
from pathlib import Path
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer

from catastrphe.config import MAX_FEATURES, VECTORIZER_PATH


class TFIDFVectorizerWrapper:
    """
    TF-IDF Vectorizer Wrapper
    - Train: fit_transform
    - Predict: transform
    """

    def __init__(self):
        # Max features are from config.py
        self.vectorizer = TfidfVectorizer(max_features=MAX_FEATURES)

    def fit(self, texts: List[str]):
        """
        Fit the vectorizer with training data
        """
        self.vectorizer.fit(texts)

    def transform(self, texts: List[str]):
        """
        Transform texts to vectors using the fitted vectorizer
        """
        return self.vectorizer.transform(texts)

    def save(self, path: Path):
        """
        Save the vectorizer to a file
        """
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.vectorizer, f)

    @staticmethod
    def load_from_file() -> TfidfVectorizer:
        """
        Load the vectorizer from file
        """
        if not VECTORIZER_PATH.exists():
            raise FileNotFoundError(f"Vectorizer file not found: {VECTORIZER_PATH}")

        with open(VECTORIZER_PATH, "rb") as f:
            return pickle.load(f)
