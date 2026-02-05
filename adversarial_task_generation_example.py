# -*- coding: utf-8 -*-
"""
Adversarial Task Generation - Integration Example
Integration with existing DeepSC model and dataset

This script demonstrates:
1. Loading a trained DeepSC model
2. Setting up the AdversarialTaskMiner
3. Generating adversarial samples
4. Analyzing and visualizing results
"""

import torch
import json
import numpy as np
from typing import Dict, List, Tuple
import sys

from adversarial_task_miner import AdversarialTaskMiner
from models.transceiver import DeepSC
from utils import SNR_to_noise
from dataset import EurDataset, collate_data
from torch.utils.data import DataLoader


class AdversarialTaskGenerator:
    """
    High-level interface for generating adversarial tasks.
    Integrates AdversarialTaskMiner with the DeepSC model ecosystem.
    """
    
    def __init__(
        self,
        model_path: str,
        vocab_file: str,
        device: torch.device = None,
        **miner_kwargs
    ):
        """
        Initialize the task generator.
        
        Args:
            model_path: Path to saved model checkpoint
            vocab_file: Path to vocabulary JSON file
            device: torch device
            **miner_kwargs: Additional arguments for AdversarialTaskMiner
        """
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        print(f"[AdversarialTaskGenerator] Using device: {self.device}")
        
        # Load vocabulary
        with open(vocab_file, 'rb') as f:
            self.vocab = json.load(f)
        
        self.token_to_idx = self.vocab['token_to_idx']
        self.idx_to_token = {v: k for k, v in self.token_to_idx.items()}
        self.vocab_size = len(self.token_to_idx)
        
        print(f"[AdversarialTaskGenerator] Loaded vocabulary with {self.vocab_size} tokens")
        
        # Build and load model
        self.model = DeepSC(
            num_layers=4,
            src_vocab_size=self.vocab_size,
            trg_vocab_size=self.vocab_size,
            src_max_len=30,
            trg_max_len=30,
            d_model=128,
            num_heads=8,
            dff=512,
            dropout=0.1
        ).to(self.device)
        
        if model_path:
            state_dict = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"[AdversarialTaskGenerator] Loaded model from {model_path}")
        
        # Create adversarial task miner
        self.miner = AdversarialTaskMiner(
            tx_model=self.model,
            rx_model=self.model,  # Can use different model if needed
            vocab_size=self.vocab_size,
            d_model=128,
            device=self.device,
            **miner_kwargs
        )
        
        print("[AdversarialTaskGenerator] Initialized AdversarialTaskMiner")
    
    def generate_from_batch(
        self,
        token_indices: torch.Tensor,
        return_stats: bool = True
    ) -> Dict:
        """
        Generate adversarial samples from a batch of token indices.
        
        Args:
            token_indices: Tensor of shape [batch_size, seq_len]
            return_stats: Whether to return optimization statistics
            
        Returns:
            Results dictionary with adversarial samples and metadata
        """
        results = self.miner.generate_adversarial_samples(
            token_indices,
            return_stats=return_stats
        )
        
        # Convert token indices to words
        adversarial_words = self.miner.get_adversarial_words(
            results['adversarial_tokens'],
            self.idx_to_token
        )
        
        results['adversarial_words'] = adversarial_words
        
        return results
    
    def generate_from_dataset(
        self,
        num_samples: int = 10,
        split: str = 'train',
        batch_size: int = 4
    ) -> List[Dict]:
        """
        Generate adversarial samples from the dataset.
        
        Args:
            num_samples: Number of adversarial samples to generate
            split: Dataset split ('train', 'val', 'test')
            batch_size: Batch size for processing
            
        Returns:
            List of results for each batch
        """
        dataset = EurDataset(split=split)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            collate_fn=collate_data
        )
        
        all_results = []
        samples_generated = 0
        
        for batch_idx, token_batch in enumerate(dataloader):
            if samples_generated >= num_samples:
                break
            
            token_batch = token_batch.to(self.device)
            
            print(f"\n[Batch {batch_idx + 1}] Processing {token_batch.shape[0]} samples...")
            results = self.generate_from_batch(token_batch, return_stats=True)
            
            # Add metadata
            results['batch_idx'] = batch_idx
            results['original_tokens'] = token_batch.cpu()
            
            all_results.append(results)
            samples_generated += token_batch.shape[0]
        
        return all_results
    
    def analyze_perturbations(self, results: Dict) -> Dict[str, float]:
        """
        Analyze the perturbation statistics.
        
        Args:
            results: Results dictionary from generate_from_batch
            
        Returns:
            Analysis dictionary with statistics
        """
        delta = results['delta'].cpu().numpy()
        
        stats = {
            'delta_mean': float(np.mean(np.abs(delta))),
            'delta_std': float(np.std(delta)),
            'delta_min': float(np.min(delta)),
            'delta_max': float(np.max(delta)),
            'delta_l2_norm': float(np.linalg.norm(delta)),
            'delta_linf_norm': float(np.max(np.abs(delta))),
        }
        
        return stats
    
    def print_results(self, results: Dict, sample_idx: int = 0):
        """
        Pretty-print the results for a single sample.
        
        Args:
            results: Results dictionary
            sample_idx: Index of sample to print
        """
        perturbed_words = results['adversarial_words'][sample_idx]
        difference_score = results['difference_score']
        
        print("\n" + "="*80)
        print(f"Adversarial Sample #{sample_idx}")
        print("="*80)
        print(f"Difference Score: {difference_score:.6f}")
        print(f"Adversarial Words: {' '.join(perturbed_words)}")
        
        if 'stats' in results:
            stats = results['stats']
            print(f"\nOptimization Statistics:")
            print(f"  Initial Loss:  {stats['initial_loss']:.6f}")
            print(f"  Final Loss:    {stats['final_loss']:.6f}")
            print(f"  Improvement:   {stats['loss_improvement']:.6f}")
        
        # Print perturbation statistics
        analysis = self.analyze_perturbations(results)
        print(f"\nPerturbation Statistics:")
        print(f"  L∞ Norm:  {analysis['delta_linf_norm']:.6f} (epsilon: {self.miner.epsilon})")
        print(f"  L2 Norm:  {analysis['delta_l2_norm']:.6f}")
        print(f"  Mean Abs: {analysis['delta_mean']:.6f}")
        print(f"  Std Dev:  {analysis['delta_std']:.6f}")
        print("="*80)


# ============================================================================
# Standalone Script Usage
# ============================================================================

def main():
    """
    Main execution function.
    Demonstrates full workflow of adversarial task generation.
    """
    print("\n" + "="*80)
    print("Adversarial Task Generation - Main Execution")
    print("="*80 + "\n")
    
    # Configuration
    config = {
        'vocab_file': '/import/antennas/Datasets/hx301/europarl/vocab.json',
        'model_path': None,  # Set to checkpoint path if available
        'device': torch.device("cuda:0" if torch.cuda.is_available() else "cpu"),
        'epsilon': 1.0,
        'num_steps': 15,
        'step_size': 0.15,
        'loss_type': 'kl_div',
    }
    
    print("Configuration:")
    for key, value in config.items():
        print(f"  {key}: {value}")
    
    # Initialize generator
    try:
        generator = AdversarialTaskGenerator(
            model_path=config['model_path'],
            vocab_file=config['vocab_file'],
            device=config['device'],
            epsilon=config['epsilon'],
            num_steps=config['num_steps'],
            step_size=config['step_size'],
            loss_type=config['loss_type'],
        )
    except Exception as e:
        print(f"\n[ERROR] Failed to initialize generator: {e}")
        print("\nNote: Make sure the vocab file path is correct and model files are available.")
        print("You can modify the paths in the config dictionary.")
        return
    
    # Generate adversarial samples
    print("\n" + "-"*80)
    print("Generating adversarial samples from dataset...")
    print("-"*80)
    
    try:
        results_list = generator.generate_from_dataset(
            num_samples=8,
            split='train',
            batch_size=4
        )
        
        # Print results
        print("\n" + "-"*80)
        print("Results Summary")
        print("-"*80)
        
        for batch_idx, results in enumerate(results_list):
            print(f"\nBatch {batch_idx}:")
            print(f"  Number of samples: {len(results['adversarial_words'])}")
            print(f"  Average difference score: {results['difference_score']:.6f}")
            
            if 'stats' in results:
                improvement = results['stats']['loss_improvement']
                print(f"  Average loss improvement: {improvement:.6f}")
            
            # Print first sample in detail
            if batch_idx == 0:
                generator.print_results(results, sample_idx=0)
    
    except Exception as e:
        print(f"\n[ERROR] Failed to generate samples: {e}")
        print("\nThis might be due to:")
        print("  - Dataset files not found")
        print("  - Model files not found")
        print("  - Incorrect paths")
        print("\nPlease check your configuration and file paths.")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
