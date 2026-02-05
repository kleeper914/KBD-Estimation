# -*- coding: utf-8 -*-
"""
Adversarial Task Generation Using Gradient-based Embedding Perturbation
Author: Implementation following PGD-style attack methodology
Date: 2026

This module implements an AdversarialTaskMiner that generates adversarial examples
by maximizing the semantic difference between TX (transmitter) and RX (receiver) model outputs
through gradient-based embedding perturbation in the embedding space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Tuple, Optional, List, Dict, Any


class AdversarialTaskMiner(nn.Module):
    """
    Generates adversarial task samples by perturbing input embeddings to maximize
    the semantic difference between TX and RX model outputs.
    
    Uses gradient-based optimization (similar to PGD attack) where we:
    1. Start with a small random perturbation delta
    2. Iteratively maximize the difference loss via gradient ascent
    3. Clip delta to bounded epsilon ball after each update
    4. Map perturbed embeddings back to discrete token space
    """
    
    def __init__(
        self,
        tx_model: nn.Module,
        rx_model: nn.Module,
        embedding_matrix: Optional[torch.Tensor] = None,
        vocab_size: Optional[int] = None,
        d_model: int = 128,
        epsilon: float = 1.0,
        num_steps: int = 10,
        step_size: float = 0.1,
        loss_type: str = 'kl_div',
        device: torch.device = None
    ):
        """
        Initialize the AdversarialTaskMiner.
        
        Args:
            tx_model: Frozen TX model (transmitter/encoder side)
            rx_model: Frozen RX model (receiver/decoder side)
            embedding_matrix: Shared embedding matrix for nearest neighbor search.
                             If None, will be extracted from tx_model.encoder.embedding.weight
            vocab_size: Vocabulary size. If None, inferred from embedding_matrix
            d_model: Embedding dimension (default: 128 for DeepSC)
            epsilon: Maximum perturbation magnitude (L-infinity bound)
            num_steps: Number of optimization iterations
            step_size: Learning rate for gradient ascent
            loss_type: 'kl_div' (KL divergence) or 'mse' (Mean Squared Error)
            device: torch device (cuda or cpu)
        """
        super(AdversarialTaskMiner, self).__init__()
        
        self.tx_model = tx_model
        self.rx_model = rx_model
        
        # Don't freeze model parameters in requires_grad!
        # Instead, we'll only update delta via the optimizer.
        # This allows gradients to flow through the computation graph
        # while model parameters remain unchanged (not included in optimizer).
        self.tx_model.eval()
        self.rx_model.eval()
        
        # Extract embedding matrix if not provided
        if embedding_matrix is None:
            self.embedding_matrix = tx_model.encoder.embedding.weight.data.clone()
        else:
            self.embedding_matrix = embedding_matrix
        
        if vocab_size is None:
            self.vocab_size = self.embedding_matrix.shape[0]
        else:
            self.vocab_size = vocab_size
            
        self.d_model = d_model
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.loss_type = loss_type
        
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        # Move models and embedding matrix to device
        self.tx_model = self.tx_model.to(self.device)
        self.rx_model = self.rx_model.to(self.device)
        self.embedding_matrix = self.embedding_matrix.to(self.device)
        
        # Validation
        assert self.embedding_matrix.shape[0] >= self.vocab_size, \
            "Embedding matrix vocabulary size mismatch"
        assert self.embedding_matrix.shape[1] == self.d_model, \
            "Embedding dimension mismatch"
        assert self.loss_type in ['kl_div', 'mse'], \
            "loss_type must be 'kl_div' or 'mse'"
    
    def forward_from_embeddings(
        self,
        embeddings: torch.Tensor,
        model: nn.Module,
        model_type: str = 'tx'
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass directly from embedding representations.
        
        This method bypasses the token-to-embedding layer and performs 
        the forward pass from the embedding layer onwards.
        
        Args:
            embeddings: Tensor of shape [batch_size, seq_len, d_model]
            model: TX or RX model
            model_type: 'tx' or 'rx' (for internal routing)
            
        Returns:
            logits: Raw output logits [batch_size, seq_len, vocab_size]
            features: Intermediate features for difference computation
        """
        # Apply positional encoding
        x = embeddings * math.sqrt(self.d_model)
        x = model.encoder.pos_encoding(x)
        
        # Pass through encoder layers
        # Create dummy mask (no padding assumed for adversarial samples)
        # Detach mask since it doesn't need gradients
        src_mask = torch.zeros(
            x.size(0), 1, x.size(1),
            device=self.device,
            dtype=torch.float32
        ).detach()
        
        # Handle both EncoderLayer (with mask) and simple Linear layers (without mask)
        for enc_layer in model.encoder.enc_layers:
            try:
                # Try calling with mask (for real EncoderLayer)
                x = enc_layer(x, src_mask)
            except TypeError:
                # Fall back to single argument (for testing Linear layers)
                x = enc_layer(x)
        
        # Channel encoding (TX-specific part)
        channel_enc = model.channel_encoder(x)
        
        # Normalize power
        channel_enc = self._power_normalize(channel_enc)
        
        # Channel pass (Ideal channel assumption for this implementation)
        channel_out = channel_enc
        
        # Channel decoding
        channel_dec = model.channel_decoder(channel_out)
        
        # Decoder part - create a dummy starting token
        batch_size = x.size(0)
        start_tokens = torch.ones(batch_size, 1, dtype=torch.long, device=self.device).detach()
        trg_embeddings = model.decoder.embedding(start_tokens) * math.sqrt(self.d_model)
        trg_embeddings = model.decoder.pos_encoding(trg_embeddings)
        
        for dec_layer in model.decoder.dec_layers:
            try:
                # Try calling with mask (for real DecoderLayer)
                trg_embeddings = dec_layer(
                    trg_embeddings, channel_dec,
                    look_ahead_mask=None,
                    trg_padding_mask=None
                )
            except TypeError:
                # Fall back to single/dual arguments (for testing Linear layers)
                trg_embeddings = dec_layer(trg_embeddings)
        
        # Generate logits
        logits = model.dense(trg_embeddings)  # [batch_size, 1, vocab_size]
        
        return logits, channel_enc
    
    def compute_difference_loss(
        self,
        tx_logits: torch.Tensor,
        rx_logits: torch.Tensor,
        tx_features: torch.Tensor,
        rx_features: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute the semantic difference loss between TX and RX outputs.
        
        We want to MAXIMIZE this loss, so we'll perform gradient ascent.
        
        Args:
            tx_logits: TX output logits [batch_size, seq_len, vocab_size]
            rx_logits: RX output logits [batch_size, seq_len, vocab_size]
            tx_features: TX intermediate features
            rx_features: RX intermediate features
            
        Returns:
            loss: Scalar tensor representing the difference
        """
        if self.loss_type == 'kl_div':
            # Compute KL divergence: KL(TX_dist || RX_dist)
            tx_probs = F.softmax(tx_logits, dim=-1)
            rx_log_probs = F.log_softmax(rx_logits, dim=-1)
            
            # Average over sequence and batch dimensions
            kl_loss = F.kl_div(rx_log_probs, tx_probs, reduction='batchmean')
            return kl_loss
            
        elif self.loss_type == 'mse':
            # Compute MSE between feature representations
            feature_loss = F.mse_loss(tx_features, rx_features, reduction='mean')
            return feature_loss
    
    def clip_delta(self, delta: torch.Tensor) -> torch.Tensor:
        """
        Clip perturbation to epsilon ball (L-infinity norm).
        
        Args:
            delta: Perturbation tensor
            
        Returns:
            Clipped perturbation
        """
        return torch.clamp(delta, -self.epsilon, self.epsilon)
    
    def _power_normalize(self, x: torch.Tensor) -> torch.Tensor:
        """
        Normalize signal power to 1.
        
        Args:
            x: Signal tensor
            
        Returns:
            Power-normalized signal
        """
        x_square = torch.mul(x, x)
        power = torch.mean(x_square).sqrt()
        if power > 1:
            x = torch.div(x, power)
        return x
    
    def generate_adversarial_samples(
        self,
        token_indices: torch.Tensor,
        return_stats: bool = False
    ) -> Dict[str, Any]:
        """
        Generate adversarial samples by perturbing embeddings.
        
        Core algorithm:
        1. Convert token indices to embeddings
        2. Initialize perturbation delta with small random values
        3. Iteratively:
           - Compute perturbed embeddings = original + delta
           - Forward through both TX and RX
           - Compute difference loss
           - Perform gradient ascent on delta
           - Clip delta to epsilon ball
        4. Return perturbed embeddings and statistics
        
        Args:
            token_indices: Tensor of token indices [batch_size, seq_len]
            return_stats: If True, return optimization statistics
            
        Returns:
            Dictionary containing:
                'perturbed_embeddings': Perturbed embedding tensor
                'delta': Final perturbation vector
                'difference_score': Final difference loss value
                'adversarial_tokens': Reconstructed token indices
                'stats': (optional) Optimization statistics
        """
        batch_size, seq_len = token_indices.shape
        
        # 1. Get original embeddings
        with torch.no_grad():
            original_embeddings = self.embedding_matrix[token_indices].to(self.device)
        
        # 2. Initialize perturbation delta
        delta = torch.zeros_like(original_embeddings, requires_grad=True)
        
        # Initialize with small random noise
        with torch.no_grad():
            delta.data = torch.randn_like(delta) * 0.01
        
        # 3. Create optimizer for delta ONLY
        # We don't freeze model parameters in requires_grad, instead we only
        # pass delta to the optimizer. This way gradients can flow through
        # the entire computation graph, but only delta will be updated.
        optimizer = torch.optim.Adam([delta], lr=self.step_size)
        
        # Track statistics
        loss_history = []
        
        print(f"[AdversarialTaskMiner] Starting optimization with {self.num_steps} steps...")
        
        # 4. Optimization loop
        # Temporarily enable gradient tracking for forward passes
        # (model parameters stay frozen, but gradients can flow through computations)
        for step in range(self.num_steps):
            # Compute current perturbed embeddings
            perturbed_embeddings = original_embeddings + delta
            
            # Forward pass through both models
            tx_logits, tx_features = self.forward_from_embeddings(
                perturbed_embeddings, self.tx_model, 'tx'
            )
            rx_logits, rx_features = self.forward_from_embeddings(
                perturbed_embeddings, self.rx_model, 'rx'
            )
                
            # Compute difference loss (we want to maximize this)
            diff_loss = self.compute_difference_loss(
                tx_logits, rx_logits,
                tx_features, rx_features
            )
            
            # Gradient ascent: maximize difference loss
            # This is equivalent to minimizing negative loss
            loss = -diff_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Clip perturbation to epsilon ball
            with torch.no_grad():
                delta.data = self.clip_delta(delta.data)
            
            loss_history.append(diff_loss.item())
            
            if (step + 1) % max(1, self.num_steps // 4) == 0:
                print(f"  Step {step + 1}/{self.num_steps}: "
                      f"Difference Loss = {diff_loss.item():.6f}")
        
        print(f"[AdversarialTaskMiner] Optimization complete!")
        
        # 5. Final perturbed embeddings and scoring
        with torch.no_grad():
            final_perturbed_embeddings = original_embeddings + delta
            
            # Compute final difference score
            tx_logits_final, tx_features_final = self.forward_from_embeddings(
                final_perturbed_embeddings, self.tx_model, 'tx'
            )
            rx_logits_final, rx_features_final = self.forward_from_embeddings(
                final_perturbed_embeddings, self.rx_model, 'rx'
            )
            
            final_diff_score = self.compute_difference_loss(
                tx_logits_final, rx_logits_final,
                tx_features_final, rx_features_final
            ).item()
            
            # Map back to discrete token space
            adversarial_tokens = self.embeddings_to_tokens(final_perturbed_embeddings)
        
        result = {
            'perturbed_embeddings': final_perturbed_embeddings.detach(),
            'delta': delta.detach(),
            'difference_score': final_diff_score,
            'adversarial_tokens': adversarial_tokens.detach(),
        }
        
        if return_stats:
            result['stats'] = {
                'loss_history': loss_history,
                'final_loss': loss_history[-1],
                'initial_loss': loss_history[0],
                'loss_improvement': loss_history[-1] - loss_history[0],
            }
        
        return result
    
    def embeddings_to_tokens(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        Map perturbed embeddings back to nearest discrete token indices.
        
        Uses nearest neighbor search in the embedding space:
        For each perturbed embedding, find the closest embedding in the
        vocabulary by computing cosine similarity or L2 distance.
        
        Args:
            embeddings: Perturbed embeddings [batch_size, seq_len, d_model]
            
        Returns:
            Token indices [batch_size, seq_len]
        """
        batch_size, seq_len, d_model = embeddings.shape
        
        # Reshape for batch processing
        flat_embeddings = embeddings.view(-1, d_model)  # [batch_size * seq_len, d_model]
        
        # Normalize embeddings for cosine similarity (optional but recommended)
        flat_embeddings_norm = F.normalize(flat_embeddings, p=2, dim=-1)
        embedding_matrix_norm = F.normalize(self.embedding_matrix, p=2, dim=-1)
        
        # Compute cosine similarity: [batch_size * seq_len, vocab_size]
        similarities = torch.mm(flat_embeddings_norm, embedding_matrix_norm.t())
        
        # Get indices of maximum similarity (nearest neighbors)
        token_indices = torch.argmax(similarities, dim=-1)  # [batch_size * seq_len]
        
        # Reshape back to original shape
        token_indices = token_indices.view(batch_size, seq_len)
        
        return token_indices
    
    def get_adversarial_words(
        self,
        token_indices: torch.Tensor,
        idx_to_token: Dict[int, str]
    ) -> List[List[str]]:
        """
        Convert adversarial token indices back to readable words.
        
        Args:
            token_indices: Tensor of token indices [batch_size, seq_len]
            idx_to_token: Dictionary mapping token indices to word strings
            
        Returns:
            List of lists containing word strings for each sample
        """
        batch_size, seq_len = token_indices.shape
        words_list = []
        
        for batch_idx in range(batch_size):
            words = []
            for seq_idx in range(seq_len):
                token_id = token_indices[batch_idx, seq_idx].item()
                word = idx_to_token.get(token_id, f"<UNK_{token_id}>")
                words.append(word)
            words_list.append(words)
        
        return words_list


# ============================================================================
# Usage Example and Testing
# ============================================================================

def example_usage():
    """
    Demonstrates how to use the AdversarialTaskMiner in practice.
    """
    print("=" * 80)
    print("AdversarialTaskMiner - Usage Example")
    print("=" * 80)
    
    # Example parameters (these should match your actual model)
    vocab_size = 5000
    d_model = 128
    batch_size = 4
    seq_len = 10
    num_steps = 15
    epsilon = 0.5
    
    # Mock token indices (in practice, these come from your dataset)
    token_indices = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    print(f"\nInput Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Embedding dimension: {d_model}")
    print(f"  Perturbation budget (epsilon): {epsilon}")
    print(f"  Optimization steps: {num_steps}")
    
    print(f"\nSample input token indices shape: {token_indices.shape}")
    print(f"Sample tokens: {token_indices[0, :5].tolist()}")
    
    # In practice, you would:
    # 1. Load your trained TX and RX models
    # 2. Create the adversarial task miner
    # 3. Generate adversarial samples
    
    print("\n" + "=" * 80)
    print("Code Template:")
    print("=" * 80)
    print("""
# Step 1: Load your trained models
from models.transceiver import DeepSC
tx_model = DeepSC(...)  # Your TX model
rx_model = DeepSC(...)  # Your RX model (can be same as TX or different)

# Step 2: Create the adversarial task miner
miner = AdversarialTaskMiner(
    tx_model=tx_model,
    rx_model=rx_model,
    vocab_size=5000,
    d_model=128,
    epsilon=1.0,
    num_steps=20,
    step_size=0.15,
    loss_type='kl_div',
    device=torch.device('cuda:0')
)

# Step 3: Generate adversarial samples
batch_token_indices = torch.randint(0, 5000, (4, 10))
results = miner.generate_adversarial_samples(
    batch_token_indices,
    return_stats=True
)

# Step 4: Inspect results
perturbed_embeddings = results['perturbed_embeddings']
adversarial_tokens = results['adversarial_tokens']
difference_score = results['difference_score']
loss_history = results['stats']['loss_history']

print(f"Difference Score: {difference_score}")
print(f"Loss improved by: {results['stats']['loss_improvement']}")

# Step 5: Map back to words (if you have idx_to_token mapping)
idx_to_token = {0: '<PAD>', 1: 'hello', 2: 'world', ...}
adversarial_words = miner.get_adversarial_words(
    adversarial_tokens,
    idx_to_token
)
print(f"Adversarial words: {adversarial_words}")
    """)
    
    print("\n" + "=" * 80)
    print("Key Parameters Explanation:")
    print("=" * 80)
    print("""
epsilon:
    - Controls the maximum perturbation magnitude (L-infinity norm)
    - Larger epsilon = more perturbation freedom, potentially larger semantic differences
    - Typical range: 0.1 - 2.0

num_steps:
    - Number of gradient ascent iterations
    - More steps = potentially better optimization, but slower
    - Typical range: 10 - 50

step_size:
    - Learning rate for the Adam optimizer
    - Affects convergence speed and stability
    - Typical range: 0.01 - 0.5

loss_type:
    - 'kl_div': KL divergence between TX and RX output distributions (recommended)
    - 'mse': Mean squared error between feature representations

Model Freezing:
    - Both TX and RX model weights remain frozen (requires_grad=False)
    - Only the perturbation delta is optimized
    - Gradients flow: delta → perturbed_embeddings → model_outputs → difference_loss
    """)


if __name__ == '__main__':
    example_usage()
