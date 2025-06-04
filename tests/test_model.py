"""
Tests for autoencoder model
"""

import pytest
import torch
import tempfile
from pathlib import Path

from catastrophe.model.autoencoder import Autoencoder


class TestAutoencoder:
    """Test autoencoder model functionality"""

    def test_model_initialization(self):
        """Test model initialization with different input dimensions"""
        # Test standard initialization
        model = Autoencoder(input_dim=100)
        # Check that encoder first layer has correct input dimension
        assert model.encoder[0].in_features == 100

        # Test with dropout
        model_with_dropout = Autoencoder(input_dim=200, dropout_rate=0.3)
        assert model_with_dropout.encoder[0].in_features == 200

    def test_forward_pass(self):
        """Test forward pass through the model"""
        model = Autoencoder(input_dim=100)

        # Create test input
        batch_size = 5
        input_tensor = torch.randn(batch_size, 100)

        # Forward pass
        output = model(input_tensor)

        # Check output shape
        assert output.shape == input_tensor.shape
        assert output.shape == (batch_size, 100)

    def test_model_training_mode(self):
        """Test model behavior in training vs evaluation mode"""
        model = Autoencoder(input_dim=50, dropout_rate=0.5)
        input_tensor = torch.randn(10, 50)

        # Training mode (dropout should be active)
        model.train()
        output_train1 = model(input_tensor)
        output_train2 = model(input_tensor)

        # Outputs might be different due to dropout
        # (though not guaranteed with small tensors)

        # Evaluation mode (dropout should be inactive)
        model.eval()
        with torch.no_grad():
            output_eval1 = model(input_tensor)
            output_eval2 = model(input_tensor)

        # Outputs should be identical in eval mode
        assert torch.allclose(output_eval1, output_eval2)

    def test_model_save_load(self):
        """Test model save and load functionality"""
        model = Autoencoder(input_dim=75)
        input_tensor = torch.randn(3, 75)

        # Get initial output
        model.eval()
        with torch.no_grad():
            original_output = model(input_tensor)

        # Save model
        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp_file:
            torch.save(model.state_dict(), tmp_file.name)

            # Create new model and load weights
            new_model = Autoencoder(input_dim=75)
            new_model.load_state_dict(torch.load(tmp_file.name, map_location="cpu"))
            new_model.eval()

            # Test that loaded model produces same output
            with torch.no_grad():
                loaded_output = new_model(input_tensor)

            assert torch.allclose(original_output, loaded_output)

        # Clean up
        Path(tmp_file.name).unlink()

    def test_gradient_flow(self):
        """Test that gradients flow properly through the model"""
        model = Autoencoder(input_dim=50)
        input_tensor = torch.randn(5, 50, requires_grad=True)

        # Forward pass
        output = model(input_tensor)
        loss = torch.mean((output - input_tensor) ** 2)

        # Backward pass
        loss.backward()

        # Check that gradients exist
        assert input_tensor.grad is not None
        for param in model.parameters():
            assert param.grad is not None

    def test_different_input_dimensions(self):
        """Test model with various input dimensions"""
        dimensions = [10, 100, 500, 1000, 2000]

        for dim in dimensions:
            model = Autoencoder(input_dim=dim)
            input_tensor = torch.randn(2, dim)
            output = model(input_tensor)

            assert output.shape == (2, dim)

    def test_batch_normalization(self):
        """Test that batch normalization layers work correctly"""
        model = Autoencoder(input_dim=100)

        # Batch norm requires more than 1 sample
        input_tensor = torch.randn(10, 100)

        model.train()
        output = model(input_tensor)

        # Should not raise any errors
        assert output.shape == input_tensor.shape

    def test_reconstruction_basic_functionality(self):
        """Test basic reconstruction functionality without training convergence requirements"""
        model = Autoencoder(input_dim=20)

        # Create test input
        test_input = torch.randn(5, 20)

        # Test forward pass
        model.eval()
        with torch.no_grad():
            output = model(test_input)

        # Check output properties
        assert output.shape == test_input.shape
        assert torch.isfinite(output).all(), "Output should contain finite values"

        # Test that model can be trained (gradients flow)
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Single training step should not crash
        optimizer.zero_grad()
        output = model(test_input)
        loss = torch.mean((output - test_input) ** 2)
        loss.backward()
        optimizer.step()

        # Check that gradients were computed
        for param in model.parameters():
            assert param.grad is not None, "Gradients should be computed"
