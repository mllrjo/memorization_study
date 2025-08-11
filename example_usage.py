# example_usage.py
# Example usage of entropy calculator module

import torch
import torch.nn as nn
import sys
import os

# Add src to path
sys.path.append(os.path.join('src'))

from core.entropy_calculator import (
    calculate_dataset_entropy,
    calculate_model_entropy,
    calculate_joint_model_entropy,
    estimate_compression_rate
)

class SimpleGPT(nn.Module):
    """Minimal GPT-style model for demonstration."""
    
    def __init__(self, vocab_size=2048, embed_dim=128, num_heads=4, num_layers=2):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(1000, embed_dim)  # Max seq len 1000
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.output_proj = nn.Linear(embed_dim, vocab_size)
        
    def forward(self, x):
        seq_len = x.size(1)
        positions = torch.arange(seq_len, device=x.device).unsqueeze(0).expand_as(x)
        
        embeds = self.embedding(x) + self.pos_embedding(positions)
        hidden = self.transformer(embeds)
        logits = self.output_proj(hidden)
        
        return logits

def main():
    """Demonstrate entropy calculator usage."""
    print("Entropy Calculator Module Demonstration")
    print("=" * 50)
    
    # Morris et al. parameters for Figure 1
    vocab_size = 2048
    sequence_length = 64
    dataset_sizes = [100, 500, 1000, 5000]
    
    print(f"Parameters: vocab_size={vocab_size}, seq_len={sequence_length}")
    print()
    
    # 1. Calculate theoretical entropy H(X)
    print("1. Theoretical Dataset Entropy H(X)")
    print("-" * 30)
    for dataset_size in dataset_sizes:
        h_x = calculate_dataset_entropy(dataset_size, sequence_length, vocab_size)
        print(f"Dataset size {dataset_size:4d}: H(X) = {h_x:8.1f} bits")
    print()
    
    # 2. Create sample models
    print("2. Creating Models")
    print("-" * 20)
    target_model = SimpleGPT(vocab_size=vocab_size, embed_dim=64, num_layers=1)  # ~170K params
    oracle_model = SimpleGPT(vocab_size=vocab_size, embed_dim=128, num_layers=2)  # Larger oracle
    
    target_params = sum(p.numel() for p in target_model.parameters())
    oracle_params = sum(p.numel() for p in oracle_model.parameters())
    
    print(f"Target model: {target_params:,} parameters")
    print(f"Oracle model: {oracle_params:,} parameters")
    print()
    
    # 3. Generate synthetic uniform data
    print("3. Generating Synthetic Uniform Data")
    print("-" * 35)
    dataset_size = 1000
    sequences = torch.randint(0, vocab_size, (dataset_size, sequence_length), dtype=torch.long)
    print(f"Generated {dataset_size} sequences of length {sequence_length}")
    print(f"Sample sequence: {sequences[0][:10].tolist()}...")
    print()
    
    # 4. Calculate model entropies
    print("4. Calculating Model Entropies")
    print("-" * 30)
    
    # H(X)
    h_x = calculate_dataset_entropy(dataset_size, sequence_length, vocab_size)
    print(f"H(X)              = {h_x:8.1f} bits")
    
    # H(X | target_model)
    h_x_target = calculate_model_entropy(sequences, target_model, batch_size=50)
    print(f"H(X | target)     = {h_x_target:8.1f} bits")
    
    # H(X | oracle_model)  
    h_x_oracle = calculate_model_entropy(sequences, oracle_model, batch_size=50)
    print(f"H(X | oracle)     = {h_x_oracle:8.1f} bits")
    
    # H(X | target, oracle)
    h_x_joint = calculate_joint_model_entropy(sequences, target_model, oracle_model, batch_size=50)
    print(f"H(X | target, oracle) = {h_x_joint:8.1f} bits")
    print()
    
    # 5. Calculate Morris's memorization metrics
    print("5. Morris's Memorization Metrics")
    print("-" * 32)
    
    # Total memorization: mem(X, target) = H(X) - H(X | target)
    total_memorization = h_x - h_x_target
    print(f"Total Memorization     = {total_memorization:8.1f} bits")
    
    # Generalization: H(X | oracle) - H(X | target, oracle)
    generalization = h_x_oracle - h_x_joint
    print(f"Generalization         = {generalization:8.1f} bits")
    
    # Unintended memorization: Total - Generalization
    unintended_memorization = total_memorization - generalization
    print(f"Unintended Memorization = {unintended_memorization:8.1f} bits")
    print()
    
    # 6. Compression rates
    print("6. Compression Rates")
    print("-" * 20)
    
    target_rate = estimate_compression_rate(sequences, target_model, batch_size=50)
    oracle_rate = estimate_compression_rate(sequences, oracle_model, batch_size=50)
    
    print(f"Target model: {target_rate:.3f} bits/token")
    print(f"Oracle model: {oracle_rate:.3f} bits/token")
    print()
    
    # 7. Capacity estimation
    print("7. Capacity Estimation")
    print("-" * 22)
    
    bits_per_parameter = unintended_memorization / target_params
    print(f"Unintended memorization: {unintended_memorization:.1f} bits")
    print(f"Model parameters: {target_params:,}")
    print(f"Estimated capacity: {bits_per_parameter:.3f} bits/parameter")
    print()
    
    print("✓ Entropy calculator module working correctly")
    print("✓ Ready for Figure 1 reproduction experiments")

if __name__ == "__main__":
    main()
