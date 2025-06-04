"""
Pytest configuration and fixtures
"""

import os
import sys
import json
import tempfile
from pathlib import Path
import pytest

# Add src to Python path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture
def test_data_dir():
    """Return path to test data directory"""
    return Path(__file__).parent / "test_data"


@pytest.fixture
def vulnerable_c_commits(test_data_dir):
    """Load vulnerable C commits test data"""
    with open(test_data_dir / "vulnerable_c_commits.json", "r") as f:
        return json.load(f)


@pytest.fixture
def safe_c_commits(test_data_dir):
    """Load safe C commits test data"""
    with open(test_data_dir / "safe_c_commits.json", "r") as f:
        return json.load(f)


@pytest.fixture
def temp_model_dir():
    """Create temporary directory for model files"""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield Path(tmp_dir)


@pytest.fixture
def mock_vectorizer_path(temp_model_dir):
    """Mock vectorizer path for testing"""
    return temp_model_dir / "vectorizer.pkl"


@pytest.fixture
def mock_model_path(temp_model_dir):
    """Mock model weights path for testing"""
    return temp_model_dir / "autoencoder_weights.pth"
