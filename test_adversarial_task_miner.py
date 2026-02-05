# -*- coding: utf-8 -*-
"""
AdversarialTaskMiner 单元测试与验证脚本

测试内容：
1. 类初始化
2. 梯度流动
3. 优化过程
4. 离散化映射
5. 输出形状验证
"""

import torch
import torch.nn as nn
import sys
from typing import Tuple


class DummyModel(nn.Module):
    """用于测试的虚拟模型"""
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


def test_basic_initialization():
    """Test: Basic Initialization"""
    print("\n" + "="*80)
    print("TEST 1: Basic Initialization")
    print("="*80)
    
    try:
        from adversarial_task_miner import AdversarialTaskMiner
        
        vocab_size = 1000
        d_model = 128
        device = torch.device('cpu')
        
        tx_model = DummyModel(vocab_size, d_model).to(device)
        rx_model = DummyModel(vocab_size, d_model).to(device)
        
        miner = AdversarialTaskMiner(
            tx_model=tx_model,
            rx_model=rx_model,
            vocab_size=vocab_size,
            d_model=d_model,
            epsilon=1.0,
            num_steps=5,
            step_size=0.1,
            device=device
        )
        
        print("[OK] AdversarialTaskMiner initialized successfully")
        print(f"  Vocab size: {miner.vocab_size}")
        print(f"  Embedding dim: {miner.d_model}")
        print(f"  Epsilon: {miner.epsilon}")
        print(f"  Device: {miner.device}")
        
        # Check if models are frozen
        for param in miner.tx_model.parameters():
            assert not param.requires_grad, "TX model should be frozen"
        for param in miner.rx_model.parameters():
            assert not param.requires_grad, "RX model should be frozen"
        
        print("[OK] Models are properly frozen (requires_grad=False)")
        
        return True
    
    except Exception as e:
        print(f"[FAILED] {e}")
        import traceback
        traceback.print_exc()
        return False


def test_embedding_extraction():
    """测试：嵌入提取"""
    print("\n" + "="*80)
    print("TEST 2: Embedding Extraction and Shapes")
    print("="*80)
    
    try:
        from adversarial_task_miner import AdversarialTaskMiner
        
        vocab_size = 1000
        d_model = 128
        batch_size = 4
        seq_len = 10
        device = torch.device('cpu')
        
        tx_model = DummyModel(vocab_size, d_model).to(device)
        rx_model = DummyModel(vocab_size, d_model).to(device)
        
        miner = AdversarialTaskMiner(
            tx_model=tx_model,
            rx_model=rx_model,
            vocab_size=vocab_size,
            d_model=d_model,
            device=device
        )
        
        # 创建样本
        token_indices = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # 提取嵌入
        embeddings = miner.embedding_matrix[token_indices].to(device)
        
        print(f"✓ Embeddings extracted successfully")
        print(f"  Input shape: {token_indices.shape}")
        print(f"  Embedding shape: {embeddings.shape}")
        print(f"  Expected shape: ({batch_size}, {seq_len}, {d_model})")
        
        assert embeddings.shape == (batch_size, seq_len, d_model), \
            f"Shape mismatch: {embeddings.shape}"
        
        return True
    
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_gradient_flow():
    """测试：梯度流动"""
    print("\n" + "="*80)
    print("TEST 3: Gradient Flow Through Delta")
    print("="*80)
    
    try:
        from adversarial_task_miner import AdversarialTaskMiner
        
        vocab_size = 1000
        d_model = 128
        batch_size = 2
        seq_len = 5
        device = torch.device('cpu')
        
        tx_model = DummyModel(vocab_size, d_model).to(device)
        rx_model = DummyModel(vocab_size, d_model).to(device)
        
        miner = AdversarialTaskMiner(
            tx_model=tx_model,
            rx_model=rx_model,
            vocab_size=vocab_size,
            d_model=d_model,
            epsilon=1.0,
            num_steps=2,
            step_size=0.1,
            device=device
        )
        
        # 创建可训练的delta
        original_embeddings = miner.embedding_matrix[
            torch.randint(0, vocab_size, (batch_size, seq_len))
        ].to(device)
        
        delta = torch.zeros_like(original_embeddings, requires_grad=True)
        perturbed = original_embeddings + delta
        
        # 前向传播
        tx_logits, tx_features = miner.forward_from_embeddings(
            perturbed, miner.tx_model, 'tx'
        )
        rx_logits, rx_features = miner.forward_from_embeddings(
            perturbed, miner.rx_model, 'rx'
        )
        
        # 计算损失
        loss = miner.compute_difference_loss(
            tx_logits, rx_logits,
            tx_features, rx_features
        )
        
        # 反向传播
        loss.backward()
        
        print(f"✓ Gradient flow successful")
        print(f"  Loss value: {loss.item():.6f}")
        print(f"  Delta shape: {delta.shape}")
        print(f"  Delta grad is not None: {delta.grad is not None}")
        print(f"  Delta grad norm: {delta.grad.norm().item():.6f}")
        
        assert delta.grad is not None, "Delta should have gradients"
        assert delta.grad.norm() > 0, "Gradient norm should be positive"
        
        return True
    
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_clipping():
    """测试：epsilon约束"""
    print("\n" + "="*80)
    print("TEST 4: Epsilon Clipping (L-infinity Constraint)")
    print("="*80)
    
    try:
        from adversarial_task_miner import AdversarialTaskMiner
        
        device = torch.device('cpu')
        epsilon = 0.5
        
        tx_model = DummyModel().to(device)
        rx_model = DummyModel().to(device)
        
        miner = AdversarialTaskMiner(
            tx_model=tx_model,
            rx_model=rx_model,
            epsilon=epsilon,
            device=device
        )
        
        # 创建超出范围的delta
        delta = torch.randn(4, 10, 128) * 2.0  # 可能超出[-eps, eps]
        
        print(f"Before clipping:")
        print(f"  Delta max: {delta.max().item():.4f}")
        print(f"  Delta min: {delta.min().item():.4f}")
        print(f"  Epsilon: {epsilon}")
        
        clipped_delta = miner.clip_delta(delta)
        
        print(f"After clipping:")
        print(f"  Delta max: {clipped_delta.max().item():.4f}")
        print(f"  Delta min: {clipped_delta.min().item():.4f}")
        
        assert clipped_delta.max() <= epsilon, "Max should be <= epsilon"
        assert clipped_delta.min() >= -epsilon, "Min should be >= -epsilon"
        
        print(f"✓ Clipping successful - delta is within [-{epsilon}, {epsilon}]")
        
        return True
    
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_embeddings_to_tokens():
    """测试：嵌入到Token的映射"""
    print("\n" + "="*80)
    print("TEST 5: Embeddings to Tokens Mapping (Nearest Neighbor)")
    print("="*80)
    
    try:
        from adversarial_task_miner import AdversarialTaskMiner
        
        vocab_size = 500
        d_model = 128
        batch_size = 3
        seq_len = 8
        device = torch.device('cpu')
        
        tx_model = DummyModel(vocab_size, d_model).to(device)
        rx_model = DummyModel(vocab_size, d_model).to(device)
        
        miner = AdversarialTaskMiner(
            tx_model=tx_model,
            rx_model=rx_model,
            vocab_size=vocab_size,
            d_model=d_model,
            device=device
        )
        
        # 创建样本嵌入
        embeddings = torch.randn(batch_size, seq_len, d_model).to(device)
        
        # 映射到Token
        tokens = miner.embeddings_to_tokens(embeddings)
        
        print(f"✓ Embeddings to tokens mapping successful")
        print(f"  Input embeddings shape: {embeddings.shape}")
        print(f"  Output tokens shape: {tokens.shape}")
        print(f"  Expected shape: ({batch_size}, {seq_len})")
        
        assert tokens.shape == (batch_size, seq_len), "Shape mismatch"
        assert tokens.min() >= 0, "Token indices should be non-negative"
        assert tokens.max() < vocab_size, "Token indices should be < vocab_size"
        
        print(f"  Token range: [{tokens.min().item()}, {tokens.max().item()}]")
        print(f"  Vocab size: {vocab_size}")
        
        return True
    
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_pipeline():
    """测试：完整管道"""
    print("\n" + "="*80)
    print("TEST 6: Full Adversarial Sample Generation Pipeline")
    print("="*80)
    
    try:
        from adversarial_task_miner import AdversarialTaskMiner
        
        vocab_size = 500
        d_model = 128
        batch_size = 2
        seq_len = 5
        num_steps = 3
        device = torch.device('cpu')
        
        tx_model = DummyModel(vocab_size, d_model).to(device)
        rx_model = DummyModel(vocab_size, d_model).to(device)
        
        miner = AdversarialTaskMiner(
            tx_model=tx_model,
            rx_model=rx_model,
            vocab_size=vocab_size,
            d_model=d_model,
            epsilon=0.5,
            num_steps=num_steps,
            step_size=0.1,
            device=device
        )
        
        # 生成对抗样本
        token_indices = torch.randint(0, vocab_size, (batch_size, seq_len))
        print(f"Input tokens shape: {token_indices.shape}")
        
        results = miner.generate_adversarial_samples(
            token_indices,
            return_stats=True
        )
        
        print(f"\n✓ Full pipeline executed successfully")
        print(f"  Perturbed embeddings shape: {results['perturbed_embeddings'].shape}")
        print(f"  Delta shape: {results['delta'].shape}")
        print(f"  Adversarial tokens shape: {results['adversarial_tokens'].shape}")
        print(f"  Difference score: {results['difference_score']:.6f}")
        
        # 验证输出形状
        assert results['perturbed_embeddings'].shape == (batch_size, seq_len, d_model)
        assert results['delta'].shape == (batch_size, seq_len, d_model)
        assert results['adversarial_tokens'].shape == (batch_size, seq_len)
        
        # 验证统计信息
        if 'stats' in results:
            stats = results['stats']
            print(f"\nOptimization statistics:")
            print(f"  Initial loss: {stats['initial_loss']:.6f}")
            print(f"  Final loss: {stats['final_loss']:.6f}")
            print(f"  Loss improvement: {stats['loss_improvement']:.6f}")
            print(f"  Num steps: {len(stats['loss_history'])}")
        
        # 验证delta在epsilon球内
        delta_linf = results['delta'].abs().max().item()
        print(f"\nPerturbation constraint check:")
        print(f"  L∞ norm of delta: {delta_linf:.6f}")
        print(f"  Epsilon: {miner.epsilon}")
        
        assert delta_linf <= miner.epsilon + 1e-5, \
            "Delta should be within epsilon ball"
        
        return True
    
    except Exception as e:
        print(f"✗ FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """运行所有测试"""
    print("\n" + "="*80)
    print("AdversarialTaskMiner - Unit Tests")
    print("="*80)
    
    tests = [
        test_basic_initialization,
        test_embedding_extraction,
        test_gradient_flow,
        test_clipping,
        test_embeddings_to_tokens,
        test_full_pipeline,
    ]
    
    results = []
    for test_func in tests:
        try:
            result = test_func()
            results.append((test_func.__name__, result))
        except Exception as e:
            print(f"\nUnexpected error in {test_func.__name__}: {e}")
            results.append((test_func.__name__, False))
    
    # 打印总结
    print("\n" + "="*80)
    print("Test Summary")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！")
        return 0
    else:
        print(f"\n⚠️  {total - passed} 个测试失败")
        return 1


if __name__ == '__main__':
    exit_code = run_all_tests()
    sys.exit(exit_code)
