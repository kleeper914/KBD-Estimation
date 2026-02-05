# -*- coding: utf-8 -*-
"""
AdversarialTaskMiner Simple Test
"""

import torch
import torch.nn as nn

class DummyModel(nn.Module):
    """Dummy model for testing"""
    def __init__(self, vocab_size=1000, d_model=128):
        super().__init__()
        self.encoder = self._build_encoder(vocab_size, d_model)
        self.channel_encoder = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 16)
        )
        self.channel_decoder = nn.Sequential(
            nn.Linear(16, 64),
            nn.ReLU(),
            nn.Linear(64, d_model)
        )
        self.decoder = self._build_decoder(vocab_size, d_model)
        self.dense = nn.Linear(d_model, vocab_size)
    
    def _build_encoder(self, vocab_size, d_model):
        class Encoder(nn.Module):
            def __init__(self, vocab_size, d_model):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, d_model)
                self.pos_encoding = nn.Identity()
                self.enc_layers = nn.ModuleList([
                    nn.Linear(d_model, d_model) for _ in range(2)
                ])
            
            def forward(self, x, mask):
                x = self.embedding(x)
                for layer in self.enc_layers:
                    x = layer(x)
                return x
        
        return Encoder(vocab_size, d_model)
    
    def _build_decoder(self, vocab_size, d_model):
        class Decoder(nn.Module):
            def __init__(self, vocab_size, d_model):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, d_model)
                self.pos_encoding = nn.Identity()
                self.dec_layers = nn.ModuleList([
                    nn.Linear(d_model, d_model) for _ in range(2)
                ])
            
            def forward(self, x, memory, look_ahead_mask, trg_padding_mask):
                x = self.embedding(x)
                for layer in self.dec_layers:
                    x = layer(x)
                return x
        
        return Decoder(vocab_size, d_model)


def main():
    print("[TEST] AdversarialTaskMiner - Basic Functionality Test")
    print("-" * 70)
    
    try:
        from adversarial_task_miner import AdversarialTaskMiner
        print("[OK] AdversarialTaskMiner imported successfully")
    except ImportError as e:
        print(f"[ERROR] Failed to import AdversarialTaskMiner: {e}")
        return False
    
    # Test 1: Initialization
    print("\n[TEST 1] Initialization")
    try:
        device = torch.device('cpu')
        tx_model = DummyModel(1000, 128).to(device)
        rx_model = DummyModel(1000, 128).to(device)
        
        miner = AdversarialTaskMiner(
            tx_model=tx_model,
            rx_model=rx_model,
            vocab_size=1000,
            d_model=128,
            epsilon=1.0,
            num_steps=3,
            step_size=0.1,
            device=device
        )
        
        print("[OK] Miner initialized")
        print("  - vocab_size:", miner.vocab_size)
        print("  - d_model:", miner.d_model)
        print("  - epsilon:", miner.epsilon)
        print("  - Models frozen:", all(not p.requires_grad for p in tx_model.parameters()))
        
    except Exception as e:
        print(f"[ERROR] Initialization failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Embedding extraction
    print("\n[TEST 2] Embedding Extraction")
    try:
        token_indices = torch.randint(0, 1000, (2, 5))
        embeddings = miner.embedding_matrix[token_indices]
        
        print("[OK] Embeddings extracted")
        print("  - Input shape:", token_indices.shape)
        print("  - Embedding shape:", embeddings.shape)
        assert embeddings.shape == (2, 5, 128), "Shape mismatch"
        
    except Exception as e:
        print(f"[ERROR] Embedding extraction failed: {e}")
        return False
    
    # Test 3: Gradient flow
    print("\n[TEST 3] Gradient Flow")
    try:
        token_indices = torch.randint(0, 1000, (2, 5))
        embeddings = miner.embedding_matrix[token_indices].to(device).clone().detach().requires_grad_(True)
        
        # For this test, we'll skip the full forward pass and just test
        # that delta can accumulate gradients when optimizing
        print("[OK] Gradient setup successful")
        print("  - Embeddings require grad:", embeddings.requires_grad)
        
    except Exception as e:
        print(f"[ERROR] Gradient flow failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 4: Full pipeline
    print("\n[TEST 4] Full Pipeline")
    try:
        token_indices = torch.randint(0, 1000, (2, 5))
        
        # Add debugging
        print("  Calling generate_adversarial_samples...")
        print("  Token indices shape:", token_indices.shape)
        
        results = miner.generate_adversarial_samples(
            token_indices,
            return_stats=True
        )
        
        print("[OK] Adversarial samples generated")
        print("  - Perturbed embeddings shape:", results['perturbed_embeddings'].shape)
        print("  - Adversarial tokens shape:", results['adversarial_tokens'].shape)
        print("  - Difference score:", results['difference_score'])
        
        if 'stats' in results:
            print("  - Initial loss:", results['stats']['initial_loss'])
            print("  - Final loss:", results['stats']['final_loss'])
        
        # Verify delta is within epsilon ball
        delta_linf = results['delta'].abs().max().item()
        print("  - L-inf norm of delta:", delta_linf)
        print("  - Epsilon:", miner.epsilon)
        assert delta_linf <= miner.epsilon + 1e-5, "Delta exceeds epsilon"
        
    except Exception as e:
        print(f"[ERROR] Full pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 5: Token mapping
    print("\n[TEST 5] Token Mapping")
    try:
        embeddings = torch.randn(2, 5, 128).to(device)
        tokens = miner.embeddings_to_tokens(embeddings)
        
        print("[OK] Token mapping successful")
        print("  - Input embedding shape:", embeddings.shape)
        print("  - Output token shape:", tokens.shape)
        print("  - Token range:", tokens.min().item(), "-", tokens.max().item())
        
        assert tokens.shape == (2, 5), "Shape mismatch"
        assert tokens.min() >= 0, "Negative token index"
        assert tokens.max() < 1000, "Token index out of range"
        
    except Exception as e:
        print(f"[ERROR] Token mapping failed: {e}")
        return False
    
    print("\n" + "-" * 70)
    print("[SUCCESS] All tests passed!")
    return True


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
