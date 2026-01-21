# -*- coding: utf-8 -*-
"""
Semantic Knowledge Difference Analysis
语义知识差异分析脚本

用于分析不同蒸馏模型之间的语义知识差异
"""

import os
import json
import torch
import numpy as np
from utils import PowerNormalize, create_masks, Channels, loss_function
from dataset import EurDataset, collate_data
from models.transceiver import DeepSC
from torch.utils.data import DataLoader
import torch.nn.functional as F
from tqdm import tqdm

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_model(checkpoint_path, model_config):
    """加载模型"""
    model = DeepSC(
        model_config['num_layers'],
        model_config['vocab_size'],
        model_config['vocab_size'],
        model_config['max_len'],
        model_config['max_len'],
        model_config['d_model'],
        model_config['num_heads'],
        model_config['dff'],
        0.1
    )
    
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"✓ 加载模型: {checkpoint_path}")
    else:
        print(f"✗ 模型文件不存在: {checkpoint_path}")
        return None
    
    return model.to(device)


def compute_semantic_difference(model1, model2, dataloader, pad_idx, channel='Rayleigh', n_var=0.1):
    """
    计算两个模型之间的语义知识差异
    
    使用多个指标:
    1. 输出分布的KL散度
    2. 中间表示的欧式距离
    3. 预测差异性 (agreement rate)
    
    Returns:
        difference_metrics: 差异指标字典
    """
    model1.eval()
    model2.eval()
    
    kl_divergences = []
    representation_dists = []
    agreement_rates = []
    
    channels = Channels()
    
    print("计算语义差异...")
    pbar = tqdm(dataloader, desc='Processing')
    
    with torch.no_grad():
        for sents in pbar:
            sents = sents.to(device)
            trg_inp = sents[:, :-1]
            
            src_mask, look_ahead_mask = create_masks(sents, trg_inp, pad_idx)
            
            # Model 1 前向传播
            enc_output1 = model1.encoder(sents, src_mask)
            channel_enc_output1 = model1.channel_encoder(enc_output1)
            Tx_sig1 = PowerNormalize(channel_enc_output1)
            
            if channel == 'AWGN':
                Rx_sig1 = channels.AWGN(Tx_sig1, n_var)
            elif channel == 'Rayleigh':
                Rx_sig1 = channels.Rayleigh(Tx_sig1, n_var)
            else:
                Rx_sig1 = channels.Rician(Tx_sig1, n_var)
            
            channel_dec_output1 = model1.channel_decoder(Rx_sig1)
            dec_output1 = model1.decoder(trg_inp, channel_dec_output1, look_ahead_mask, src_mask)
            pred1 = model1.dense(dec_output1)  # [batch, seq_len, vocab_size]
            
            # Model 2 前向传播
            enc_output2 = model2.encoder(sents, src_mask)
            channel_enc_output2 = model2.channel_encoder(enc_output2)
            Tx_sig2 = PowerNormalize(channel_enc_output2)
            
            if channel == 'AWGN':
                Rx_sig2 = channels.AWGN(Tx_sig2, n_var)
            elif channel == 'Rayleigh':
                Rx_sig2 = channels.Rayleigh(Tx_sig2, n_var)
            else:
                Rx_sig2 = channels.Rician(Tx_sig2, n_var)
            
            channel_dec_output2 = model2.channel_decoder(Rx_sig2)
            dec_output2 = model2.decoder(trg_inp, channel_dec_output2, look_ahead_mask, src_mask)
            pred2 = model2.dense(dec_output2)
            
            # 计算KL散度
            pred1_soft = F.softmax(pred1, dim=-1)
            pred2_soft = F.softmax(pred2, dim=-1)
            kl_div = F.kl_div(F.log_softmax(pred2, dim=-1), pred1_soft, reduction='batchmean')
            kl_divergences.append(kl_div.item())
            
            # 计算中间表示差异
            repr_dist = F.mse_loss(dec_output1, dec_output2, reduction='mean')
            representation_dists.append(repr_dist.item())
            
            # 计算预测一致性
            pred1_top = torch.argmax(pred1, dim=-1)
            pred2_top = torch.argmax(pred2, dim=-1)
            agreement = (pred1_top == pred2_top).float().mean().item()
            agreement_rates.append(agreement)
    
    # 计算平均值
    avg_kl_div = np.mean(kl_divergences)
    avg_repr_dist = np.mean(representation_dists)
    avg_agreement = np.mean(agreement_rates)
    
    metrics = {
        'kl_divergence': avg_kl_div,
        'representation_distance': avg_repr_dist,
        'agreement_rate': avg_agreement,
        'semantic_difference_score': avg_kl_div * (1 - avg_agreement),
    }
    
    return metrics


def analyze_distillation_effects(model_paths, config_paths, model_config, pad_idx, channel='Rayleigh'):
    """
    分析蒸馏效果：比较不同温度参数的模型
    
    Args:
        model_paths: 模型文件路径列表
        config_paths: 配置文件路径列表
        model_config: 模型配置
        pad_idx: 填充索引
        channel: 信道类型
    """
    # 加载测试数据
    test_eur = EurDataset('test')
    test_iterator = DataLoader(test_eur, batch_size=64, num_workers=0,
                                pin_memory=True, collate_fn=collate_data)
    
    # 加载教师模型（参考）
    print("\n=== 加载参考模型 ===")
    teacher_model = load_model(model_paths[0], model_config)
    
    results = []
    
    print("\n=== 分析蒸馏效果 ===\n")
    
    for model_path, config_path in zip(model_paths[1:], config_paths[1:]):
        # 加载配置
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # 加载学生模型
        student_model = load_model(model_path, model_config)
        if student_model is None:
            continue
        
        # 计算语义差异
        print(f"\n分析模型: {os.path.basename(model_path)}")
        print(f"  配置: T={config['temperature']}, α={config['distill_weight']}, "
              f"部分={config.get('distill_part', 'full')}")
        
        metrics = compute_semantic_difference(
            teacher_model, student_model, test_iterator, pad_idx, channel
        )
        
        result = {
            'model': os.path.basename(model_path),
            'temperature': config['temperature'],
            'distill_weight': config['distill_weight'],
            'distill_part': config.get('distill_part', 'full'),
            'kl_divergence': metrics['kl_divergence'],
            'representation_distance': metrics['representation_distance'],
            'agreement_rate': metrics['agreement_rate'],
            'semantic_difference_score': metrics['semantic_difference_score'],
        }
        
        results.append(result)
        
        print(f"  ✓ KL散度: {metrics['kl_divergence']:.4f}")
        print(f"  ✓ 表示距离: {metrics['representation_distance']:.4f}")
        print(f"  ✓ 一致性: {metrics['agreement_rate']:.4f}")
        print(f"  ✓ 语义差异分数: {metrics['semantic_difference_score']:.4f}")
    
    return results


def print_analysis_report(results):
    """打印分析报告"""
    print("\n" + "="*80)
    print("语义知识差异分析报告")
    print("="*80)
    
    print(f"\n分析了 {len(results)} 个蒸馏模型\n")
    
    print("详细结果:")
    print("-" * 80)
    print(f"{'模型':<30} {'T值':<8} {'KL散度':<12} {'表示距离':<12} {'一致性':<12} {'差异分数':<12}")
    print("-" * 80)
    
    for result in results:
        print(f"{result['model']:<30} "
              f"{result['temperature']:<8.1f} "
              f"{result['kl_divergence']:<12.4f} "
              f"{result['representation_distance']:<12.4f} "
              f"{result['agreement_rate']:<12.4f} "
              f"{result['semantic_difference_score']:<12.4f}")
    
    print("-" * 80)
    
    # 统计
    temps = [r['temperature'] for r in results]
    kl_divs = [r['kl_divergence'] for r in results]
    sem_diffs = [r['semantic_difference_score'] for r in results]
    
    print("\n统计信息:")
    print(f"  温度范围: {min(temps):.1f} - {max(temps):.1f}")
    print(f"  KL散度范围: {min(kl_divs):.4f} - {max(kl_divs):.4f}")
    print(f"  语义差异范围: {min(sem_diffs):.4f} - {max(sem_diffs):.4f}")
    
    # 找出最大差异和最小差异的模型
    max_diff_idx = np.argmax(sem_diffs)
    min_diff_idx = np.argmin(sem_diffs)
    
    print(f"\n最大语义差异的模型:")
    print(f"  {results[max_diff_idx]['model']} "
          f"(T={results[max_diff_idx]['temperature']}, "
          f"差异分数={results[max_diff_idx]['semantic_difference_score']:.4f})")
    
    print(f"\n最小语义差异的模型:")
    print(f"  {results[min_diff_idx]['model']} "
          f"(T={results[min_diff_idx]['temperature']}, "
          f"差异分数={results[min_diff_idx]['semantic_difference_score']:.4f})")
    
    print("\n" + "="*80)


if __name__ == '__main__':
    import sys
    
    # 示例: 分析温度参数的效果
    print("="*80)
    print("语义知识差异分析")
    print("="*80)
    
    # 配置
    teacher_checkpoint = 'checkpoints/deepsc-Rayleigh/checkpoint_50.pth'
    model_config = {
        'num_layers': 4,
        'vocab_size': 50000,  # 需要从实际项目中获取
        'max_len': 30,
        'd_model': 128,
        'num_heads': 8,
        'dff': 512,
    }
    
    # 通常从词汇表获取
    import json
    with open('europarl/vocab.json', 'r') as f:
        vocab = json.load(f)
    model_config['vocab_size'] = len(vocab['token_to_idx'])
    pad_idx = vocab['token_to_idx']['<PAD>']
    
    # 模型路径
    model_paths = [
        teacher_checkpoint,
        'checkpoints/temp_exp/T2.0/final_student_model.pt',
        'checkpoints/temp_exp/T4.0/final_student_model.pt',
        'checkpoints/temp_exp/T8.0/final_student_model.pt',
    ]
    
    # 配置路径
    config_paths = [
        None,  # 教师模型没有配置
        'checkpoints/temp_exp/T2.0/distillation_config.json',
        'checkpoints/temp_exp/T4.0/distillation_config.json',
        'checkpoints/temp_exp/T8.0/distillation_config.json',
    ]
    
    # 检查文件存在
    print("检查文件...")
    for path in model_paths[1:]:
        if not os.path.exists(path):
            print(f"⚠ 文件不存在: {path}")
            print("  请先运行蒸馏脚本生成模型")
            sys.exit(1)
    
    print("✓ 所有文件存在\n")
    
    # 执行分析
    results = analyze_distillation_effects(
        model_paths, config_paths, model_config, pad_idx, channel='Rayleigh'
    )
    
    # 打印报告
    print_analysis_report(results)
    
    # 保存结果
    output_file = 'semantic_analysis_results.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"\n✓ 结果已保存到: {output_file}")
