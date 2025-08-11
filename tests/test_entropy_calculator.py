# tests/test_entropy_calculator.py
# Comprehensive tests for entropy calculator module

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from unittest.mock import patch
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.entropy_calculator import (
    calculate_dataset_entropy,
    calculate_model_entropy,
    calculate_joint_model_entropy,
    estimate_compression_rate,
    get_device,
    validate_sequences_tensor
)

class SimpleLanguageModel(nn.Module):
    """Simple language model for testing."""
    
    def __init__(self, vocab_size: int, embed_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.linear1 = nn.Linear(embed_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len)
        embeds = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        # Simple averaging for next token prediction
        pooled = embeds.mean(dim=1)  # (batch_size, embed_dim)
        hidden = F.relu(self.linear1(pooled))
        logits = self.linear2(hidden)  # (batch_size, vocab_size)
        # Expand to match input sequence length for each position
        return logits.unsqueeze(1).expand(-1, x.size(1), -1)

class DeterministicModel(nn.Module):
    """Deterministic model that always predicts token 0."""
    
    def __init__(self, vocab_size: int):
        super().__init__()
        self.vocab_size = vocab_size
        
    def forward(self, x):
        batch_size, seq_len = x.shape
        # Create logits that strongly predict token 0
        logits = torch.full((batch_size, seq_len, self.vocab_size), -10.0, device=x.device, dtype=torch.float32)
        logits[:, :, 0] = 10.0  # High logit for token 0
        return logits

class TestDatasetEntropy:
    """Test theoretical entropy calculation."""
    
    def test_basic_calculation(self):
        """Test basic entropy calculation H(X) = N * S * log2(V)."""
        result = calculate_dataset_entropy(
            dataset_size=100,
            sequence_length=64,
            vocab_size=2048
        )
        expected = 100 * 64 * math.log2(2048)
        assert abs(result - expected) < 1e-10
        
    def test_single_values(self):
        """Test with minimal values."""
        result = calculate_dataset_entropy(
            dataset_size=1,
            sequence_length=1,
            vocab_size=2
        )
        expected = 1.0  # log2(2) = 1
        assert abs(result - expected) < 1e-10
        
    def test_large_values(self):
        """Test with large values."""
        result = calculate_dataset_entropy(
            dataset_size=1000000,
            sequence_length=128,
            vocab_size=50000
        )
        expected = 1000000 * 128 * math.log2(50000)
        assert abs(result - expected) < 1e-6
        
    def test_invalid_inputs(self):
        """Test error handling for invalid inputs."""
        with pytest.raises(ValueError, match="dataset_size must be positive"):
            calculate_dataset_entropy(0, 64, 2048)
            
        with pytest.raises(ValueError, match="sequence_length must be positive"):
            calculate_dataset_entropy(100, 0, 2048)
            
        with pytest.raises(ValueError, match="vocab_size must be positive"):
            calculate_dataset_entropy(100, 64, 0)
            
        with pytest.raises(ValueError, match="dataset_size must be int"):
            calculate_dataset_entropy(100.5, 64, 2048)

class TestDeviceUtils:
    """Test device utility functions."""
    
    def test_device_auto_detection(self):
        """Test automatic device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
        assert device.type in ['cpu', 'cuda']
        
    def test_explicit_device(self):
        """Test explicit device specification."""
        cpu_device = torch.device('cpu')
        result = get_device(cpu_device)
        assert result == cpu_device
        
    def test_invalid_device(self):
        """Test invalid device handling."""
        with pytest.raises(ValueError, match="device must be torch.device"):
            get_device("cpu")

class TestTensorValidation:
    """Test tensor validation functions."""
    
    def test_valid_tensor(self):
        """Test validation of valid tensor."""
        sequences = torch.randint(0, 100, (10, 20), dtype=torch.long)
        validate_sequences_tensor(sequences)  # Should not raise
        
    def test_wrong_type(self):
        """Test non-tensor input."""
        with pytest.raises(ValueError, match="sequences must be torch.Tensor"):
            validate_sequences_tensor([[1, 2, 3]])
            
    def test_wrong_dimensions(self):
        """Test wrong tensor dimensions."""
        with pytest.raises(ValueError, match="sequences must be 2D tensor"):
            validate_sequences_tensor(torch.randint(0, 100, (10,)))
            
        with pytest.raises(ValueError, match="sequences must be 2D tensor"):
            validate_sequences_tensor(torch.randint(0, 100, (10, 20, 5)))
            
    def test_wrong_dtype(self):
        """Test wrong tensor dtype."""
        with pytest.raises(ValueError, match="sequences must have integer dtype"):
            validate_sequences_tensor(torch.randn(10, 20))
            
    def test_empty_tensor(self):
        """Test empty tensor."""
        with pytest.raises(ValueError, match="sequences tensor is empty"):
            validate_sequences_tensor(torch.empty(0, 0, dtype=torch.long))
            
    def test_negative_values(self):
        """Test tensor with negative values."""
        sequences = torch.tensor([[-1, 2, 3], [4, 5, 6]], dtype=torch.long)
        with pytest.raises(ValueError, match="sequences contains negative token ids"):
            validate_sequences_tensor(sequences)

class TestModelEntropy:
    """Test model entropy calculation."""
    
    @pytest.fixture
    def simple_model(self):
        """Create simple model for testing."""
        return SimpleLanguageModel(vocab_size=10)
        
    @pytest.fixture
    def deterministic_model(self):
        """Create deterministic model for testing."""
        return DeterministicModel(vocab_size=10)
        
    @pytest.fixture
    def sample_sequences(self):
        """Create sample sequences for testing."""
        return torch.randint(0, 10, (5, 8), dtype=torch.long)
        
    def test_basic_entropy_calculation(self, simple_model, sample_sequences):
        """Test basic entropy calculation."""
        entropy = calculate_model_entropy(
            sequences=sample_sequences,
            model=simple_model,
            batch_size=2
        )
        assert isinstance(entropy, float)
        assert entropy > 0
        assert not math.isnan(entropy)
        assert not math.isinf(entropy)
        
    def test_deterministic_model_low_entropy(self, sample_sequences):
        """Test that deterministic model gives low entropy."""
        det_model = DeterministicModel(vocab_size=10)
        
        # Create sequences of all zeros (what the deterministic model predicts)
        zero_sequences = torch.zeros_like(sample_sequences)
        
        entropy = calculate_model_entropy(
            sequences=zero_sequences,
            model=det_model,
            batch_size=2
        )
        
        # Should have very low entropy since model perfectly predicts zeros
        assert entropy < 1.0  # Should be close to 0
        
    def test_cpu_gpu_consistency(self, simple_model, sample_sequences):
        """Test that CPU and GPU give same results."""
        cpu_entropy = calculate_model_entropy(
            sequences=sample_sequences,
            model=simple_model,
            device=torch.device('cpu'),
            batch_size=2
        )
        
        if torch.cuda.is_available():
            gpu_entropy = calculate_model_entropy(
                sequences=sample_sequences,
                model=simple_model,
                device=torch.device('cuda'),
                batch_size=2
            )
            # Allow small numerical differences
            assert abs(cpu_entropy - gpu_entropy) < 1e-4
            
    def test_batch_size_consistency(self, simple_model, sample_sequences):
        """Test that different batch sizes give same results."""
        entropy1 = calculate_model_entropy(
            sequences=sample_sequences,
            model=simple_model,
            batch_size=1
        )
        
        entropy2 = calculate_model_entropy(
            sequences=sample_sequences,
            model=simple_model,
            batch_size=3
        )
        
        # Should be very close regardless of batch size (allow for floating point precision)
        assert abs(entropy1 - entropy2) < 1e-5
        
    def test_single_token_sequences(self, simple_model):
        """Test with single token sequences."""
        single_token_seqs = torch.randint(0, 10, (3, 1), dtype=torch.long)
        
        entropy = calculate_model_entropy(
            sequences=single_token_seqs,
            model=simple_model,
            batch_size=2
        )
        
        assert isinstance(entropy, float)
        assert entropy >= 0
        
    def test_invalid_inputs(self, simple_model):
        """Test error handling for invalid inputs."""
        valid_sequences = torch.randint(0, 10, (5, 8), dtype=torch.long)
        
        with pytest.raises(ValueError, match="model must be nn.Module"):
            calculate_model_entropy(valid_sequences, "not_a_model")
            
        with pytest.raises(ValueError, match="batch_size must be positive int"):
            calculate_model_entropy(valid_sequences, simple_model, batch_size=0)
            
        with pytest.raises(ValueError, match="batch_size must be positive int"):
            calculate_model_entropy(valid_sequences, simple_model, batch_size=2.5)

class TestJointModelEntropy:
    """Test joint model entropy calculation."""
    
    @pytest.fixture
    def model_pair(self):
        """Create pair of models for testing."""
        model1 = SimpleLanguageModel(vocab_size=10, hidden_dim=32)
        model2 = SimpleLanguageModel(vocab_size=10, hidden_dim=64)
        return model1, model2
        
    @pytest.fixture
    def sample_sequences(self):
        """Create sample sequences for testing."""
        return torch.randint(0, 10, (4, 6), dtype=torch.long)
        
    def test_basic_joint_entropy(self, model_pair, sample_sequences):
        """Test basic joint entropy calculation."""
        model1, model2 = model_pair
        
        entropy = calculate_joint_model_entropy(
            sequences=sample_sequences,
            model1=model1,
            model2=model2,
            batch_size=2
        )
        
        assert isinstance(entropy, float)
        assert entropy > 0
        assert not math.isnan(entropy)
        assert not math.isinf(entropy)
        
    def test_joint_entropy_bounds(self, model_pair, sample_sequences):
        """Test that joint entropy is bounded by individual entropies."""
        model1, model2 = model_pair
        
        entropy1 = calculate_model_entropy(sample_sequences, model1, batch_size=2)
        entropy2 = calculate_model_entropy(sample_sequences, model2, batch_size=2)
        joint_entropy = calculate_joint_model_entropy(
            sample_sequences, model1, model2, batch_size=2
        )
        
        # Joint entropy should be <= min of individual entropies
        assert joint_entropy <= min(entropy1, entropy2) + 1e-6  # Small tolerance
        
    def test_identical_models(self, sample_sequences):
        """Test joint entropy with identical models."""
        model1 = SimpleLanguageModel(vocab_size=10, hidden_dim=32)
        model2 = SimpleLanguageModel(vocab_size=10, hidden_dim=32)
        
        # Make models identical by copying weights
        model2.load_state_dict(model1.state_dict())
        
        entropy1 = calculate_model_entropy(sample_sequences, model1, batch_size=2)
        joint_entropy = calculate_joint_model_entropy(
            sample_sequences, model1, model2, batch_size=2
        )
        
        # Joint entropy should equal individual entropy for identical models
        assert abs(entropy1 - joint_entropy) < 1e-5
        
    def test_cpu_gpu_consistency(self, model_pair, sample_sequences):
        """Test CPU/GPU consistency for joint entropy."""
        model1, model2 = model_pair
        
        cpu_entropy = calculate_joint_model_entropy(
            sequences=sample_sequences,
            model1=model1,
            model2=model2,
            device=torch.device('cpu'),
            batch_size=2
        )
        
        if torch.cuda.is_available():
            gpu_entropy = calculate_joint_model_entropy(
                sequences=sample_sequences,
                model1=model1,
                model2=model2,
                device=torch.device('cuda'),
                batch_size=2
            )
            assert abs(cpu_entropy - gpu_entropy) < 1e-4

class TestCompressionRate:
    """Test compression rate estimation."""
    
    @pytest.fixture
    def simple_model(self):
        """Create simple model for testing."""
        return SimpleLanguageModel(vocab_size=10)
        
    @pytest.fixture
    def sample_sequences(self):
        """Create sample sequences for testing."""
        return torch.randint(0, 10, (5, 8), dtype=torch.long)
        
    def test_basic_compression_rate(self, simple_model, sample_sequences):
        """Test basic compression rate calculation."""
        rate = estimate_compression_rate(
            sequences=sample_sequences,
            model=simple_model,
            batch_size=2
        )
        
        assert isinstance(rate, float)
        assert rate > 0
        assert rate < 20  # Should be reasonable for 10-token vocab
        assert not math.isnan(rate)
        assert not math.isinf(rate)
        
    def test_compression_rate_vs_entropy(self, simple_model, sample_sequences):
        """Test relationship between compression rate and entropy."""
        rate = estimate_compression_rate(sample_sequences, simple_model)
        entropy = calculate_model_entropy(sample_sequences, simple_model)
        
        num_sequences, seq_len = sample_sequences.shape
        expected_rate = entropy / (num_sequences * (seq_len - 1))
        
        assert abs(rate - expected_rate) < 1e-6
        
    def test_deterministic_model_compression(self, sample_sequences):
        """Test compression rate with deterministic model."""
        det_model = DeterministicModel(vocab_size=10)
        
        # Use sequences of all zeros (what the model predicts perfectly)
        zero_sequences = torch.zeros_like(sample_sequences)
        
        rate = estimate_compression_rate(zero_sequences, det_model)
        
        # Should have very low compression rate
        assert rate < 0.1

class TestNumericalStability:
    """Test numerical stability edge cases."""
    
    def test_very_low_probabilities(self):
        """Test handling of very low probability tokens."""
        model = DeterministicModel(vocab_size=100)
        
        # Create sequences with tokens the model never predicts
        sequences = torch.full((3, 5), 99, dtype=torch.long)  # Token 99, model predicts 0
        
        entropy = calculate_model_entropy(sequences, model, batch_size=1)
        
        # Should handle very low probabilities without overflow
        assert not math.isnan(entropy)
        assert not math.isinf(entropy)
        assert entropy > 0
        
    def test_large_vocabulary(self):
        """Test with large vocabulary size."""
        large_vocab_model = SimpleLanguageModel(vocab_size=50000)
        sequences = torch.randint(0, 50000, (2, 10), dtype=torch.long)
        
        entropy = calculate_model_entropy(sequences, large_vocab_model, batch_size=1)
        
        assert not math.isnan(entropy)
        assert not math.isinf(entropy)
        assert entropy > 0

class TestMemoryEfficiency:
    """Test memory efficiency with large inputs."""
    
    def test_large_dataset_batching(self):
        """Test that large datasets are handled efficiently."""
        model = SimpleLanguageModel(vocab_size=100)
        
        # Large dataset
        large_sequences = torch.randint(0, 100, (1000, 20), dtype=torch.long)
        
        entropy = calculate_model_entropy(
            sequences=large_sequences,
            model=model,
            batch_size=32  # Process in batches
        )
        
        assert not math.isnan(entropy)
        assert not math.isinf(entropy)
        assert entropy > 0

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v", "--tb=short"])
