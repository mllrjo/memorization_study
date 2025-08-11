# src/core/entropy_calculator.py
# Core entropy calculation functions for memorization research

import logging
import math
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from datetime import datetime
import os

# Configure logging
def setup_logging():
    """Setup logging configuration for entropy calculations."""
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'logs/entropy_calculator_{timestamp}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

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
    # Type validation
    if not isinstance(dataset_size, int):
        raise ValueError(f"dataset_size must be int, got {type(dataset_size)}")
    if not isinstance(sequence_length, int):
        raise ValueError(f"sequence_length must be int, got {type(sequence_length)}")
    if not isinstance(vocab_size, int):
        raise ValueError(f"vocab_size must be int, got {type(vocab_size)}")
    
    # Value validation
    if dataset_size <= 0:
        raise ValueError(f"dataset_size must be positive, got {dataset_size}")
    if sequence_length <= 0:
        raise ValueError(f"sequence_length must be positive, got {sequence_length}")
    if vocab_size <= 0:
        raise ValueError(f"vocab_size must be positive, got {vocab_size}")
    
    # Calculate entropy: H(X) = N * S * log2(V)
    entropy_bits = float(dataset_size * sequence_length * math.log2(vocab_size))
    
    logger.info(f"Calculated dataset entropy: {entropy_bits:.2f} bits "
                f"(N={dataset_size}, S={sequence_length}, V={vocab_size})")
    
    return entropy_bits

def get_device(device: Optional[torch.device] = None) -> torch.device:
    """
    Get appropriate compute device (CPU/GPU).
    
    Args:
        device: Specific device to use (auto-detected if None)
        
    Returns:
        torch.device object for computation
    """
    if device is not None:
        if not isinstance(device, torch.device):
            raise ValueError(f"device must be torch.device, got {type(device)}")
        return device
    
    # Auto-detect device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        logger.info(f"Using GPU: {torch.cuda.get_device_name()}")
    else:
        device = torch.device('cpu')
        logger.info("Using CPU")
    
    return device

def validate_sequences_tensor(sequences: torch.Tensor) -> None:
    """
    Validate sequences tensor has correct shape and dtype.
    
    Args:
        sequences: Tensor to validate
        
    Raises:
        ValueError: If tensor has wrong shape, dtype, or contains invalid values
    """
    if not isinstance(sequences, torch.Tensor):
        raise ValueError(f"sequences must be torch.Tensor, got {type(sequences)}")
    
    if sequences.dim() != 2:
        raise ValueError(f"sequences must be 2D tensor (num_sequences, sequence_length), "
                        f"got {sequences.dim()}D with shape {sequences.shape}")
    
    if sequences.dtype not in [torch.long, torch.int]:
        raise ValueError(f"sequences must have integer dtype for token ids, got {sequences.dtype}")
    
    if sequences.numel() == 0:
        raise ValueError("sequences tensor is empty")
    
    if torch.any(sequences < 0):
        raise ValueError("sequences contains negative token ids")
    
    logger.debug(f"Validated sequences tensor: shape {sequences.shape}, dtype {sequences.dtype}")

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
    # Input validation
    validate_sequences_tensor(sequences)
    
    if not isinstance(model, nn.Module):
        raise ValueError(f"model must be nn.Module, got {type(model)}")
    
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(f"batch_size must be positive int, got {batch_size}")
    
    if not isinstance(log_interval, int) or log_interval <= 0:
        raise ValueError(f"log_interval must be positive int, got {log_interval}")
    
    # Setup device and move data
    device = get_device(device)
    model = model.to(device)
    model.eval()
    sequences = sequences.to(device)
    
    num_sequences, sequence_length = sequences.shape
    total_log_prob = 0.0
    total_tokens = 0
    
    logger.info(f"Calculating model entropy for {num_sequences} sequences "
                f"of length {sequence_length} using batch_size {batch_size}")
    
    try:
        with torch.no_grad():
            for batch_idx in range(0, num_sequences, batch_size):
                batch_end = min(batch_idx + batch_size, num_sequences)
                batch_sequences = sequences[batch_idx:batch_end]
                batch_size_actual = batch_sequences.size(0)
                
                # Calculate log probabilities for each token in sequence
                # Model expects input tokens and predicts next tokens
                if sequence_length <= 1:
                    # Handle single token case
                    input_tokens = batch_sequences
                    target_tokens = batch_sequences
                else:
                    input_tokens = batch_sequences[:, :-1]  # All but last token
                    target_tokens = batch_sequences[:, 1:]  # All but first token
                
                # Forward pass
                try:
                    logits = model(input_tokens)
                    if isinstance(logits, tuple):
                        logits = logits[0]  # Handle models that return (logits, ...)
                    
                    # Calculate log probabilities
                    log_probs = F.log_softmax(logits, dim=-1)
                    
                    # Gather log probabilities for target tokens
                    target_log_probs = torch.gather(
                        log_probs.reshape(-1, log_probs.size(-1)),
                        1,
                        target_tokens.reshape(-1, 1)
                    ).squeeze(-1)
                    
                    # Sum log probabilities (in natural log)
                    batch_log_prob = target_log_probs.sum().item()
                    total_log_prob += batch_log_prob
                    total_tokens += target_tokens.numel()
                    
                except Exception as e:
                    raise RuntimeError(f"Model forward pass failed on batch {batch_idx}: {str(e)}")
                
                if (batch_idx // batch_size) % log_interval == 0:
                    logger.info(f"Processed batch {batch_idx // batch_size + 1}, "
                               f"sequences {batch_idx + 1}-{batch_end}")
    
    except Exception as e:
        logger.error(f"Error in entropy calculation: {str(e)}")
        raise
    
    # Convert from natural log to log2 and negate for entropy
    # H(X|model) = -sum(log2(p(x|model))) = -sum(ln(p(x|model))) / ln(2)
    entropy_bits = -total_log_prob / math.log(2)
    
    logger.info(f"Calculated model entropy: {entropy_bits:.2f} bits "
                f"({total_tokens} tokens, {entropy_bits/total_tokens:.4f} bits/token)")
    
    return entropy_bits

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
    # Input validation
    validate_sequences_tensor(sequences)
    
    if not isinstance(model1, nn.Module):
        raise ValueError(f"model1 must be nn.Module, got {type(model1)}")
    if not isinstance(model2, nn.Module):
        raise ValueError(f"model2 must be nn.Module, got {type(model2)}")
    
    if not isinstance(batch_size, int) or batch_size <= 0:
        raise ValueError(f"batch_size must be positive int, got {batch_size}")
    
    # Setup device and move data
    device = get_device(device)
    model1 = model1.to(device)
    model2 = model2.to(device)
    model1.eval()
    model2.eval()
    sequences = sequences.to(device)
    
    num_sequences, sequence_length = sequences.shape
    total_log_prob = 0.0
    total_tokens = 0
    
    logger.info(f"Calculating joint model entropy for {num_sequences} sequences "
                f"using batch_size {batch_size}")
    
    try:
        with torch.no_grad():
            for batch_idx in range(0, num_sequences, batch_size):
                batch_end = min(batch_idx + batch_size, num_sequences)
                batch_sequences = sequences[batch_idx:batch_end]
                
                # Prepare input and target tokens
                if sequence_length <= 1:
                    input_tokens = batch_sequences
                    target_tokens = batch_sequences
                else:
                    input_tokens = batch_sequences[:, :-1]
                    target_tokens = batch_sequences[:, 1:]
                
                # Forward pass for both models
                try:
                    # Model 1
                    logits1 = model1(input_tokens)
                    if isinstance(logits1, tuple):
                        logits1 = logits1[0]
                    log_probs1 = F.log_softmax(logits1, dim=-1)
                    target_log_probs1 = torch.gather(
                        log_probs1.reshape(-1, log_probs1.size(-1)),
                        1,
                        target_tokens.reshape(-1, 1)
                    ).squeeze(-1)
                    
                    # Model 2
                    logits2 = model2(input_tokens)
                    if isinstance(logits2, tuple):
                        logits2 = logits2[0]
                    log_probs2 = F.log_softmax(logits2, dim=-1)
                    target_log_probs2 = torch.gather(
                        log_probs2.reshape(-1, log_probs2.size(-1)),
                        1,
                        target_tokens.reshape(-1, 1)
                    ).squeeze(-1)
                    
                    # Take maximum log probability (in log space)
                    max_log_probs = torch.maximum(target_log_probs1, target_log_probs2)
                    
                    batch_log_prob = max_log_probs.sum().item()
                    total_log_prob += batch_log_prob
                    total_tokens += target_tokens.numel()
                    
                except Exception as e:
                    raise RuntimeError(f"Model forward pass failed on batch {batch_idx}: {str(e)}")
                
                if (batch_idx // batch_size) % log_interval == 0:
                    logger.info(f"Processed batch {batch_idx // batch_size + 1}")
    
    except Exception as e:
        logger.error(f"Error in joint entropy calculation: {str(e)}")
        raise
    
    # Convert to bits and negate for entropy
    entropy_bits = -total_log_prob / math.log(2)
    
    logger.info(f"Calculated joint model entropy: {entropy_bits:.2f} bits "
                f"({entropy_bits/total_tokens:.4f} bits/token)")
    
    return entropy_bits

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
    # Calculate entropy and convert to bits per token
    entropy_bits = calculate_model_entropy(
        sequences=sequences,
        model=model,
        device=device,
        batch_size=batch_size,
        log_interval=1000  # Less frequent logging for compression rate
    )
    
    num_sequences, sequence_length = sequences.shape
    total_tokens = num_sequences * (sequence_length - 1)  # Exclude first token
    
    if total_tokens == 0:
        raise ValueError("No tokens to calculate compression rate")
    
    compression_rate = entropy_bits / total_tokens
    
    logger.info(f"Estimated compression rate: {compression_rate:.4f} bits/token")
    
    return compression_rate
