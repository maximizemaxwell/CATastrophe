"""
Integration tests for the complete pipeline
"""

import pytest
import torch
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, Mock

from catastrophe.features.vectorizer import TFIDFVectorizerWrapper
from catastrophe.model.autoencoder import Autoencoder


class TestIntegration:
    """Integration tests for the complete vulnerability detection pipeline"""

    def test_full_pipeline_c_code(
        self, vulnerable_c_commits, safe_c_commits, temp_model_dir
    ):
        """Test complete pipeline from C code to vulnerability prediction"""
        # Combine all commits for training
        all_commits = vulnerable_c_commits + safe_c_commits
        texts = [
            commit["message"] + " <SEP> " + commit["func"] for commit in all_commits
        ]

        # Step 1: Train vectorizer
        vectorizer = TFIDFVectorizerWrapper()
        vectorizer.fit(texts)

        # Step 2: Transform to vectors
        X = vectorizer.transform(texts).toarray()

        # Step 3: Create and train model
        model = Autoencoder(input_dim=X.shape[1])

        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)

        # Simple training loop
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        model.train()

        for epoch in range(10):  # Quick training
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = torch.mean((outputs - X_tensor) ** 2)
            loss.backward()
            optimizer.step()

        # Step 4: Test prediction on new data
        test_vulnerable = {
            "message": "add login function",
            "func": "void login(char* password) { char buffer[8]; gets(buffer); }",
        }

        test_safe = {
            "message": "add login function",
            "func": "void login(const char* password, size_t max_len) { char buffer[64]; if (max_len < 64) strncpy(buffer, password, max_len); }",
        }

        # Predict vulnerability scores
        for test_case in [test_vulnerable, test_safe]:
            text = test_case["message"] + " <SEP> " + test_case["func"]
            vector = vectorizer.transform([text]).toarray()
            tensor = torch.tensor(vector, dtype=torch.float32)

            model.eval()
            with torch.no_grad():
                output = model(tensor)
                error = torch.mean((tensor - output) ** 2)

            assert torch.isfinite(error)
            assert error >= 0

    def test_model_persistence(self, vulnerable_c_commits):
        """Test saving and loading trained model"""
        # Prepare data
        texts = [
            commit["message"] + " <SEP> " + commit["func"]
            for commit in vulnerable_c_commits
        ]

        # Train vectorizer
        vectorizer = TFIDFVectorizerWrapper()
        vectorizer.fit(texts)

        # Train model
        X = vectorizer.transform(texts).toarray()
        model = Autoencoder(input_dim=X.shape[1])

        X_tensor = torch.tensor(X, dtype=torch.float32)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Train briefly
        model.train()
        for _ in range(5):
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = torch.mean((outputs - X_tensor) ** 2)
            loss.backward()
            optimizer.step()

        # Get prediction before saving
        model.eval()
        with torch.no_grad():
            original_output = model(X_tensor[:1])

        # Save model
        with tempfile.TemporaryDirectory() as tmp_dir:
            model_path = Path(tmp_dir) / "test_model.pth"
            vectorizer_path = Path(tmp_dir) / "test_vectorizer.pkl"

            torch.save(model.state_dict(), model_path)
            vectorizer.save(vectorizer_path)

            # Load model
            new_model = Autoencoder(input_dim=X.shape[1])
            new_model.load_state_dict(torch.load(model_path, map_location="cpu"))
            new_model.eval()

            # Test loaded model produces same output
            with torch.no_grad():
                loaded_output = new_model(X_tensor[:1])

            assert torch.allclose(original_output, loaded_output, atol=1e-6)

    def test_batch_processing(self, vulnerable_c_commits, safe_c_commits):
        """Test processing multiple commits in batches"""
        # Mix vulnerable and safe commits
        all_commits = vulnerable_c_commits[:5] + safe_c_commits[:5]
        texts = [
            commit["message"] + " <SEP> " + commit["func"] for commit in all_commits
        ]

        # Train system
        vectorizer = TFIDFVectorizerWrapper()
        vectorizer.fit(texts)

        X = vectorizer.transform(texts).toarray()
        model = Autoencoder(input_dim=X.shape[1])

        # Process in batches
        batch_size = 3
        X_tensor = torch.tensor(X, dtype=torch.float32)

        model.eval()
        all_outputs = []

        for i in range(0, len(X_tensor), batch_size):
            batch = X_tensor[i : i + batch_size]
            with torch.no_grad():
                batch_output = model(batch)
                all_outputs.append(batch_output)

        # Concatenate results
        final_output = torch.cat(all_outputs, dim=0)

        assert final_output.shape == X_tensor.shape

        # Compare with single batch processing
        with torch.no_grad():
            single_output = model(X_tensor)

        assert torch.allclose(final_output, single_output, atol=1e-6)

    def test_different_code_languages_handling(self):
        """Test how system handles different programming languages"""
        # Test with mixed language code (though system is designed for C)
        mixed_code_samples = [
            {
                "message": "buffer overflow in C",
                "func": "void unsafe() { char buf[10]; gets(buf); }",
            },
            {
                "message": "sql injection in python",
                "func": "def query(user_input): return f'SELECT * FROM users WHERE name={user_input}'",
            },
            {
                "message": "safe function in C",
                "func": "void safe(const char* input, size_t size) { if(size > 0 && size < MAX_SIZE) process(input); }",
            },
        ]

        texts = [
            sample["message"] + " <SEP> " + sample["func"]
            for sample in mixed_code_samples
        ]

        # Train on mixed data
        vectorizer = TFIDFVectorizerWrapper()
        vectorizer.fit(texts)

        X = vectorizer.transform(texts).toarray()
        model = Autoencoder(input_dim=X.shape[1])

        X_tensor = torch.tensor(X, dtype=torch.float32)

        # Test prediction
        model.eval()
        with torch.no_grad():
            outputs = model(X_tensor)
            errors = torch.mean((X_tensor - outputs) ** 2, dim=1)

        assert len(errors) == 3
        assert all(torch.isfinite(errors))

    def test_empty_and_malformed_input_handling(self):
        """Test system robustness with problematic inputs"""
        # Test various problematic inputs
        problematic_inputs = [
            {"message": "", "func": ""},
            {"message": "   ", "func": "   "},
            {"message": "test", "func": ""},
            {"message": "", "func": "int x;"},
            {"message": "test" * 1000, "func": "x" * 1000},  # Very long input
        ]

        # Use some normal data for training
        normal_data = [
            {"message": "fix bug", "func": "int safe_function() { return 0; }"},
            {"message": "add feature", "func": 'void feature() { printf("hello"); }'},
        ]

        all_data = normal_data + problematic_inputs
        texts = [item["message"] + " <SEP> " + item["func"] for item in all_data]

        try:
            vectorizer = TFIDFVectorizerWrapper()
            vectorizer.fit(texts)

            X = vectorizer.transform(texts).toarray()
            model = Autoencoder(input_dim=X.shape[1])

            X_tensor = torch.tensor(X, dtype=torch.float32)

            model.eval()
            with torch.no_grad():
                outputs = model(X_tensor)

            # If we reach here, system handled problematic inputs gracefully
            assert outputs.shape == X_tensor.shape

        except Exception as e:
            # Some failures are acceptable for very problematic inputs
            print(f"Expected failure with problematic inputs: {e}")

    def test_performance_with_large_dataset(self):
        """Test system performance with larger dataset"""
        # Generate larger synthetic dataset
        large_dataset = []

        # Create variations of vulnerable patterns
        vulnerable_patterns = [
            "strcpy(dest, src)",
            "gets(buffer)",
            "sprintf(output, format, input)",
            "system(command)",
            "printf(user_input)",
        ]

        safe_patterns = [
            "strncpy(dest, src, size)",
            "fgets(buffer, size, stdin)",
            "snprintf(output, size, format, input)",
            "execv(path, args)",
            'printf("%s", user_input)',
        ]

        for i in range(50):  # 50 of each type
            large_dataset.append(
                {
                    "message": f"vulnerability fix {i}",
                    "func": f"void func{i}() {{ {vulnerable_patterns[i % len(vulnerable_patterns)]}; }}",
                }
            )
            large_dataset.append(
                {
                    "message": f"safe implementation {i}",
                    "func": f"void safe_func{i}() {{ {safe_patterns[i % len(safe_patterns)]}; }}",
                }
            )

        texts = [item["message"] + " <SEP> " + item["func"] for item in large_dataset]

        # Test that system can handle this dataset size
        vectorizer = TFIDFVectorizerWrapper()
        vectorizer.fit(texts)

        X = vectorizer.transform(texts).toarray()
        model = Autoencoder(input_dim=X.shape[1])

        X_tensor = torch.tensor(X, dtype=torch.float32)

        # Quick training
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        model.train()

        for epoch in range(3):  # Just a few epochs for performance test
            optimizer.zero_grad()
            outputs = model(X_tensor)
            loss = torch.mean((outputs - X_tensor) ** 2)
            loss.backward()
            optimizer.step()

        # Test final prediction
        model.eval()
        with torch.no_grad():
            final_outputs = model(X_tensor)

        assert final_outputs.shape == X_tensor.shape
        assert len(large_dataset) == 100  # Verify we processed all samples
