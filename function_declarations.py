# function_declarations.py
# Comprehensive function declarations with types and descriptions

from typing import Union, Optional, Tuple
import torch
import torch.nn as nn

# ===== src/core/entropy_calculator.py =====

def calculate_dataset_entropy(
    dataset_size: int,
    sequence_length: int, 
    vocab_size: int
) -> float:
    """
    Calculate theoretical entropy H(X) for uniform random dataset.
    
    Args:
        dataset_size: Number of sequences in dataset
        sequence_length: Length of each sequence in tokens
        vocab_size: Size of vocabulary
        
    Returns:
        Entropy in bits: dataset_size * sequence_length * log2(vocab_size)
        
    Raises:
        ValueError: If any parameter is non-positive
    """

def calculate_model_entropy(
    sequences: torch.Tensor,
    model: nn.Module,
    device: Optional[torch.device] = None,
    batch_size: int = 32,
    log_interval: int = 100
) -> float:
    """
    Calculate conditional entropy H(X | model) using model likelihoods.
    
    Args:
        sequences: Tensor of shape (num_sequences, sequence_length) containing token ids
        model: PyTorch model that returns logits for next token prediction
        device: Device to run computation on (auto-detected if None)
        batch_size: Batch size for processing sequences
        log_interval: Interval for progress logging
        
    Returns:
        Conditional entropy in bits: -sum(log2(p(x | model)))
        
    Raises:
        ValueError: If sequences tensor has wrong shape or dtype
        RuntimeError: If model forward pass fails
    """

def calculate_joint_model_entropy(
    sequences: torch.Tensor,
    model1: nn.Module,
    model2: nn.Module,
    device: Optional[torch.device] = None,
    batch_size: int = 32,
    log_interval: int = 100
) -> float:
    """
    Calculate joint conditional entropy H(X | model1, model2) using maximum likelihoods.
    
    Args:
        sequences: Tensor of shape (num_sequences, sequence_length) containing token ids
        model1: First PyTorch model
        model2: Second PyTorch model  
        device: Device to run computation on (auto-detected if None)
        batch_size: Batch size for processing sequences
        log_interval: Interval for progress logging
        
    Returns:
        Joint conditional entropy in bits: -sum(log2(max(p(x|model1), p(x|model2))))
        
    Raises:
        ValueError: If sequences tensor has wrong shape or dtype
        RuntimeError: If either model forward pass fails
    """

def estimate_compression_rate(
    sequences: torch.Tensor,
    model: nn.Module,
    device: Optional[torch.device] = None,
    batch_size: int = 32
) -> float:
    """
    Estimate compression rate in bits per token using model likelihoods.
    
    Args:
        sequences: Tensor of shape (num_sequences, sequence_length) containing token ids
        model: PyTorch model that returns logits for next token prediction
        device: Device to run computation on (auto-detected if None)  
        batch_size: Batch size for processing sequences
        
    Returns:
        Average bits per token needed to encode sequences given model
        
    Raises:
        ValueError: If sequences tensor has wrong shape or dtype
        RuntimeError: If model forward pass fails
    """

def get_device(device: Optional[torch.device] = None) -> torch.device:
    """
    Get appropriate compute device (CPU/GPU).
    
    Args:
        device: Specific device to use (auto-detected if None)
        
    Returns:
        torch.device object for computation
    """

def validate_sequences_tensor(sequences: torch.Tensor) -> None:
    """
    Validate sequences tensor has correct shape and dtype.
    
    Args:
        sequences: Tensor to validate
        
    Raises:
        ValueError: If tensor has wrong shape, dtype, or contains invalid values
    """
