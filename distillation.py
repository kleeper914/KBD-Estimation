# -*- coding: utf-8 -*-
"""
Created on 2026

Model Distillation Script with Temperature Control
能够通过控制温度参数T控制学生模型与教师模型的知识差异
支持Encoder-Decoder架构的收发端知识差异蒸馏

@author: Knowledge Distillation
"""
import os
import argparse
import time
import json
import torch
import random
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import SNR_to_noise, initNetParams, PowerNormalize, create_masks, Channels, loss_function
from dataset import EurDataset, collate_data
from models.transceiver import DeepSC
from torch.utils.data import DataLoader
from tqdm import tqdm
from copy import deepcopy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--vocab-file', default='europarl/vocab.json', type=str)
parser.add_argument('--teacher-checkpoint', default='checkpoints/deepsc-Rayleigh/model.pt', type=str, 
                    help='Path to teacher model checkpoint')
parser.add_argument('--student-checkpoint', default='', type=str, 
                    help='Path to student model checkpoint (for fine-tuning)')
parser.add_argument('--output-path', default='checkpoints/distillation', type=str,
                    help='Path to save distilled student model')
parser.add_argument('--channel', default='Rayleigh', type=str, 
                    help='Please choose Ideal, AWGN, Rayleigh, and Rician')
parser.add_argument('--MAX-LENGTH', default=30, type=int)
parser.add_argument('--MIN-LENGTH', default=4, type=int)
parser.add_argument('--d-model', default=128, type=int)
parser.add_argument('--dff', default=512, type=int)
parser.add_argument('--num-layers', default=4, type=int)
parser.add_argument('--num-heads', default=8, type=int)
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--epochs', default=80, type=int)
parser.add_argument('--temperature', default=4.0, type=float,
                    help='Temperature parameter for knowledge distillation (higher = softer distribution)')
parser.add_argument('--distill-weight', default=0.7, type=float,
                    help='Weight for distillation loss (alpha), task loss weight = 1-alpha')
parser.add_argument('--lr', default=1e-4, type=float, help='Learning rate')
parser.add_argument('--student-layers', default=2, type=int,
                    help='Number of layers in student model (should be <= teacher layers)')
parser.add_argument('--distill-part', default='full', type=str, 
                    choices=['full', 'encoder', 'decoder'],
                    help='Which part to distill: full (both), encoder only, or decoder only')
parser.add_argument('--create-hybrid', action='store_true',
                    help='Create hybrid model with distilled decoder weights')
parser.add_argument('--hybrid-output', default='checkpoints/hybrid_model.pt', type=str,
                    help='Path to save hybrid model (original encoder + distilled decoder)')

def setup_seed(seed):
    """设置随机种子, 确保模型训练的可重现性"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def calculate_distillation_loss(student_output, teacher_output, temperature):
    """
    计算蒸馏损失函数
    使用带温度参数T的KL散度损失函数
    
    Args:
        student_output: 学生模型的输出 [batch_size*seq_len, vocab_size]
        teacher_output: 教师模型的输出 [batch_size*seq_len, vocab_size]
        temperature: 温度参数T，控制softmax的平滑程度
                    - T越大，softmax分布越平滑，学生模型学到的知识越"软"
                    - T越小，分布越尖锐，更接近hard target
    
    Returns:
        蒸馏损失值
    """
    # 对输出进行softmax，使用温度参数进行缩放
    student_soft = F.softmax(student_output / temperature, dim=-1)
    teacher_soft = F.softmax(teacher_output / temperature, dim=-1)
    
    # 计算KL散度: KL(teacher || student)
    kl_loss = F.kl_div(
        F.log_softmax(student_output / temperature, dim=-1),
        teacher_soft,
        reduction='batchmean'
    )
    
    # 乘以temperature的平方，用于平衡梯度尺度
    distill_loss = kl_loss * (temperature ** 2)
    
    return distill_loss


def calculate_intermediate_distillation_loss(student_intermediate, teacher_intermediate, temperature):
    """
    计算中间层蒸馏损失（用于Encoder或Decoder的中间表示匹配）
    
    Args:
        student_intermediate: 学生模型中间表示 [batch_size, seq_len, d_model]
        teacher_intermediate: 教师模型中间表示 [batch_size, seq_len, d_model]
        temperature: 温度参数
    
    Returns:
        中间层蒸馏损失
    """
    # 使用MSE损失来匹配中间表示
    mse_loss = F.mse_loss(student_intermediate, teacher_intermediate, reduction='mean')
    return mse_loss


def forward_pass(model, src, trg, padding_idx, channel, n_var=0.1, return_intermediate=False):
    """
    执行前向传播，返回模型输出
    
    Args:
        model: Transformer模型
        src: 源序列输入
        trg: 目标序列输入
        padding_idx: 填充索引
        channel: 通道类型
        n_var: 噪声方差
        return_intermediate: 是否返回中间表示
    
    Returns:
        pred: 模型预测输出 [batch_size, seq_len-1, vocab_size]
        Tx_sig: 发送信号
        Rx_sig: 接收信号
        intermediate: 中间表示（如果return_intermediate=True）
    """
    channels = Channels()
    trg_inp = trg[:, :-1]
    
    src_mask, look_ahead_mask = create_masks(src, trg_inp, padding_idx)
    
    # Encoder
    enc_output = model.encoder(src, src_mask)
    channel_enc_output = model.channel_encoder(enc_output)
    Tx_sig = PowerNormalize(channel_enc_output)
    
    # 通道模拟
    if channel == 'Ideal':
        Rx_sig = channels.Ideal(Tx_sig, n_var)
    elif channel == 'AWGN':
        Rx_sig = channels.AWGN(Tx_sig, n_var)
    elif channel == 'Rayleigh':
        Rx_sig = channels.Rayleigh(Tx_sig, n_var)
    elif channel == 'Rician':
        Rx_sig = channels.Rician(Tx_sig, n_var)
    else:
        raise ValueError("Please choose from Ideal, AWGN, Rayleigh, and Rician")
    
    # Decoder
    channel_dec_output = model.channel_decoder(Rx_sig)
    dec_output = model.decoder(trg_inp, channel_dec_output, look_ahead_mask, src_mask)
    pred = model.dense(dec_output)
    
    if return_intermediate:
        return pred, Tx_sig, Rx_sig, {'enc_output': enc_output, 'dec_output': dec_output}
    
    return pred, Tx_sig, Rx_sig


def distillation_step(student_model, teacher_model, src, trg, n_var, pad_idx, 
                     student_optimizer, criterion, channel, temperature, distill_weight, 
                     distill_part='full'):
    """
    执行一步蒸馏训练
    
    Args:
        student_model: 学生模型
        teacher_model: 教师模型
        src: 源序列
        trg: 目标序列
        n_var: 噪声方差
        pad_idx: 填充索引
        student_optimizer: 学生模型优化器
        criterion: 原始损失函数
        channel: 通道类型
        temperature: 温度参数
        distill_weight: 蒸馏损失的权重(alpha)
        distill_part: 蒸馏部分 ('full', 'encoder', 'decoder')
    
    Returns:
        total_loss: 总损失
        task_loss: 任务损失
        distill_loss: 蒸馏损失
    """
    student_model.train()
    teacher_model.eval()
    
    trg_real = trg[:, 1:]
    student_optimizer.zero_grad()
    
    # 获取学生模型的输出
    student_result = forward_pass(student_model, src, trg, pad_idx, channel, n_var, return_intermediate=True)
    if len(student_result) == 4:
        student_pred, _, _, student_inter = student_result
    else:
        student_pred, _, _ = student_result
        student_inter = None
    
    # 获取教师模型的输出（不计算梯度）
    with torch.no_grad():
        teacher_result = forward_pass(teacher_model, src, trg, pad_idx, channel, n_var, return_intermediate=True)
        if len(teacher_result) == 4:
            teacher_pred, _, _, teacher_inter = teacher_result
        else:
            teacher_pred, _, _ = teacher_result
            teacher_inter = None
    
    ntokens = student_pred.size(-1)
    
    # 计算任务损失（标准交叉熵损失）
    task_loss = loss_function(student_pred.contiguous().view(-1, ntokens),
                              trg_real.contiguous().view(-1),
                              pad_idx, criterion)
    
    # 计算蒸馏损失（KL散度）
    distill_loss = calculate_distillation_loss(
        student_pred.contiguous().view(-1, ntokens),
        teacher_pred.contiguous().view(-1, ntokens),
        temperature
    )
    
    # 如果蒸馏特定部分，添加中间层损失
    if distill_part in ['encoder', 'decoder'] and student_inter is not None and teacher_inter is not None:
        if distill_part == 'encoder':
            inter_loss = calculate_intermediate_distillation_loss(
                student_inter['enc_output'], teacher_inter['enc_output'], temperature
            )
        else:  # decoder
            inter_loss = calculate_intermediate_distillation_loss(
                student_inter['dec_output'], teacher_inter['dec_output'], temperature
            )
        distill_loss = distill_loss + 0.5 * inter_loss
    
    # 组合损失：total_loss = (1-alpha)*task_loss + alpha*distill_loss
    total_loss = (1 - distill_weight) * task_loss + distill_weight * distill_loss
    
    # 反向传播
    total_loss.backward()
    student_optimizer.step()
    
    return total_loss.item(), task_loss.item(), distill_loss.item()


def validate_distillation(student_model, teacher_model, args, pad_idx, criterion, distill_part='full'):
    """
    验证蒸馏模型的性能
    """
    test_eur = EurDataset('test')
    test_iterator = DataLoader(test_eur, batch_size=args.batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_data)
    
    student_model.eval()
    teacher_model.eval()
    
    student_total_loss = 0.0
    teacher_total_loss = 0.0
    distill_total_loss = 0.0
    
    pbar = tqdm(test_iterator, desc='Validating')
    with torch.no_grad():
        for sents in pbar:
            sents = sents.to(device)
            trg_inp = sents[:, :-1]
            trg_real = sents[:, 1:]
            
            # 学生模型
            student_result = forward_pass(student_model, sents, sents, pad_idx, args.channel, 0.1, return_intermediate=True)
            if len(student_result) == 4:
                student_pred, _, _, student_inter = student_result
            else:
                student_pred, _, _ = student_result
                student_inter = None
            
            # 教师模型
            teacher_result = forward_pass(teacher_model, sents, sents, pad_idx, args.channel, 0.1, return_intermediate=True)
            if len(teacher_result) == 4:
                teacher_pred, _, _, teacher_inter = teacher_result
            else:
                teacher_pred, _, _ = teacher_result
                teacher_inter = None
            
            ntokens = student_pred.size(-1)
            
            student_loss = loss_function(student_pred.contiguous().view(-1, ntokens),
                                        trg_real.contiguous().view(-1),
                                        pad_idx, criterion)
            
            teacher_loss = loss_function(teacher_pred.contiguous().view(-1, ntokens),
                                        trg_real.contiguous().view(-1),
                                        pad_idx, criterion)
            
            distill_loss = calculate_distillation_loss(
                student_pred.contiguous().view(-1, ntokens),
                teacher_pred.contiguous().view(-1, ntokens),
                args.temperature
            )
            
            student_total_loss += student_loss
            teacher_total_loss += teacher_loss
            distill_total_loss += distill_loss
    
    return (student_total_loss / len(test_iterator),
            teacher_total_loss / len(test_iterator),
            distill_total_loss / len(test_iterator))


def create_hybrid_model(original_model, distilled_model, distill_part, vocab_size, 
                        d_model, dff, num_heads, num_layers, max_len, device, output_path):
    """
    创建混合模型：将蒸馏模型的特定部分（encoder或decoder）替换到原模型中
    
    Args:
        original_model: 原始教师模型
        distilled_model: 蒸馏后的学生模型
        distill_part: 蒸馏部分 ('full', 'encoder', 'decoder')
        vocab_size: 词汇表大小
        其他参数：模型配置
        output_path: 保存路径
    
    Returns:
        hybrid_model: 混合模型
    """
    print("\n=== Creating Hybrid Model ===")
    
    # 创建新的混合模型（基于原模型结构）
    hybrid_model = DeepSC(num_layers, vocab_size, vocab_size, vocab_size, vocab_size,
                          d_model, num_heads, dff, 0.1).to(device)
    
    if distill_part == 'decoder':
        # 使用原模型的encoder，蒸馏模型的decoder
        print("Creating hybrid model: Original Encoder + Distilled Decoder")
        
        # 复制原模型的encoder权重
        hybrid_model.encoder.load_state_dict(original_model.encoder.state_dict())
        hybrid_model.channel_encoder.load_state_dict(original_model.channel_encoder.state_dict())
        hybrid_model.channel_decoder.load_state_dict(original_model.channel_decoder.state_dict())
        
        # 使用蒸馏模型的decoder权重
        hybrid_model.decoder.load_state_dict(distilled_model.decoder.state_dict())
        hybrid_model.dense.load_state_dict(distilled_model.dense.state_dict())
        
        print("✓ Encoder: Original")
        print("✓ Decoder: Distilled")
        
    elif distill_part == 'encoder':
        # 使用蒸馏模型的encoder，原模型的decoder
        print("Creating hybrid model: Distilled Encoder + Original Decoder")
        
        # 使用蒸馏模型的encoder权重
        hybrid_model.encoder.load_state_dict(distilled_model.encoder.state_dict())
        hybrid_model.channel_encoder.load_state_dict(distilled_model.channel_encoder.state_dict())
        hybrid_model.channel_decoder.load_state_dict(distilled_model.channel_decoder.state_dict())
        
        # 复制原模型的decoder权重
        hybrid_model.decoder.load_state_dict(original_model.decoder.state_dict())
        hybrid_model.dense.load_state_dict(original_model.dense.state_dict())
        
        print("✓ Encoder: Distilled")
        print("✓ Decoder: Original")
        
    else:  # full
        # 完全使用蒸馏模型
        print("Creating hybrid model: Fully Distilled Model")
        hybrid_model.load_state_dict(distilled_model.state_dict())
        print("✓ All components: Distilled")
    
    # 保存混合模型
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    torch.save(hybrid_model.state_dict(), output_path)
    print(f"✓ Hybrid model saved to {output_path}\n")
    
    return hybrid_model


def train_distillation(args):
    """
    执行知识蒸馏训练
    """
    # 创建输出目录
    os.makedirs(args.output_path, exist_ok=True)
    
    # 设置随机种子
    setup_seed(42)
    
    # 加载词汇表
    with open(args.vocab_file, 'r') as f:
        vocab = json.load(f)
    vocab_size = len(vocab['token_to_idx'])
    pad_idx = vocab['token_to_idx']['<PAD>']
    
    print(f"Vocabulary size: {vocab_size}")
    print(f"Padding index: {pad_idx}")
    print(f"Temperature: {args.temperature}")
    print(f"Distillation weight (alpha): {args.distill_weight}")
    print(f"Channel: {args.channel}")
    
    print(f"Distillation part: {args.distill_part}")
    
    # 创建教师模型
    print("\n=== Loading Teacher Model ===")
    teacher_model = DeepSC(args.num_layers, vocab_size, vocab_size,
                           args.MAX_LENGTH, args.MAX_LENGTH, args.d_model, 
                           args.num_heads, args.dff, 0.1)
    
    if os.path.exists(args.teacher_checkpoint):
        print(f"Loading teacher checkpoint: {args.teacher_checkpoint}")
        teacher_model.load_state_dict(torch.load(args.teacher_checkpoint, map_location=device))
    else:
        print(f"Warning: Teacher checkpoint not found at {args.teacher_checkpoint}")
        print("Initializing teacher model with random weights")
        teacher_model = initNetParams(teacher_model)
    
    teacher_model = teacher_model.to(device)
    teacher_model.eval()  # 教师模型在蒸馏过程中不更新
    
    # 创建学生模型
    print("\n=== Creating Student Model ===")
    print(f"Student layers: {args.student_layers}")
    student_model = DeepSC(args.student_layers, vocab_size, vocab_size,
                           args.MAX_LENGTH, args.MAX_LENGTH, args.d_model,
                           args.num_heads, args.dff, 0.1)
    
    if args.student_checkpoint and os.path.exists(args.student_checkpoint):
        print(f"Loading student checkpoint: {args.student_checkpoint}")
        student_model.load_state_dict(torch.load(args.student_checkpoint, map_location=device))
    else:
        print("Initializing student model with random weights")
        student_model = initNetParams(student_model)
    
    student_model = student_model.to(device)
    
    # 计算参数数量
    teacher_params = sum(p.numel() for p in teacher_model.parameters())
    student_params = sum(p.numel() for p in student_model.parameters())
    print(f"\nTeacher model parameters: {teacher_params:,}")
    print(f"Student model parameters: {student_params:,}")
    print(f"Compression ratio: {teacher_params/student_params:.2f}x")
    
    # 优化器和损失函数
    student_optimizer = torch.optim.Adam(student_model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss(reduction='none')
    
    # 加载训练数据
    print("\n=== Loading Training Data ===")
    train_eur = EurDataset('distill')
    train_iterator = DataLoader(train_eur, batch_size=args.batch_size, num_workers=0,
                                pin_memory=True, collate_fn=collate_data)
    
    print(f"Training samples: {len(train_eur)}")
    
    # 记录最佳模型
    best_loss = float('inf')
    best_epoch = 0
    
    # 训练循环
    print("\n=== Starting Distillation Training ===\n")
    
    for epoch in range(args.epochs):
        epoch_start_time = time.time()
        
        pbar = tqdm(train_iterator, desc=f'Epoch {epoch+1}/{args.epochs}')
        
        total_loss = 0.0
        total_task_loss = 0.0
        total_distill_loss = 0.0
        batch_count = 0
        
        # 随机生成噪声标准差
        noise_std = np.random.uniform(SNR_to_noise(5), SNR_to_noise(10), size=(1))[0]
        
        for sents in pbar:
            sents = sents.to(device)
            
            # 执行蒸馏训练步骤
            loss, task_loss, distill_loss = distillation_step(
                student_model, teacher_model, sents, sents, noise_std,
                pad_idx, student_optimizer, criterion, args.channel,
                args.temperature, args.distill_weight, args.distill_part
            )
            
            total_loss += loss
            total_task_loss += task_loss
            total_distill_loss += distill_loss
            batch_count += 1
            
            pbar.set_description(
                f'Epoch: {epoch+1}; Loss: {loss:.5f}; '
                f'Task: {task_loss:.5f}; Distill: {distill_loss:.5f}'
            )
        
        avg_loss = total_loss / batch_count
        avg_task_loss = total_task_loss / batch_count
        avg_distill_loss = total_distill_loss / batch_count
        
        epoch_time = time.time() - epoch_start_time
        
        # 验证
        print(f"\nValidating...")
        val_student_loss, val_teacher_loss, val_distill_loss = validate_distillation(
            student_model, teacher_model, args, pad_idx, criterion, args.distill_part
        )
        
        print(f'Epoch {epoch+1} - Time: {epoch_time:.2f}s')
        print(f'  Train - Total Loss: {avg_loss:.5f}, Task Loss: {avg_task_loss:.5f}, '
              f'Distill Loss: {avg_distill_loss:.5f}')
        print(f'  Val   - Student Loss: {val_student_loss:.5f}, Teacher Loss: {val_teacher_loss:.5f}, '
              f'Distill Loss: {val_distill_loss:.5f}\n')
        
        # 保存最佳模型
        if val_student_loss < best_loss:
            best_loss = val_student_loss
            best_epoch = epoch
            best_model_path = os.path.join(args.output_path, 'best_student_model.pt')
            torch.save(student_model.state_dict(), best_model_path)
            print(f"✓ Best model saved to {best_model_path}\n")
        
        # 定期保存模型
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(args.output_path, f'student_model_epoch{epoch+1}.pt')
            torch.save(student_model.state_dict(), checkpoint_path)
            print(f"✓ Checkpoint saved to {checkpoint_path}\n")
    
    # 训练完成，保存最终模型
    final_model_path = os.path.join(args.output_path, 'final_student_model.pt')
    torch.save(student_model.state_dict(), final_model_path)
    print(f"\n✓ Final model saved to {final_model_path}")
    print(f"✓ Best model was at epoch {best_epoch+1} with loss {best_loss:.5f}")
    
    # 创建混合模型（如果指定）
    hybrid_model = None
    if args.create_hybrid:
        os.makedirs(os.path.dirname(args.hybrid_output), exist_ok=True)
        hybrid_model = create_hybrid_model(
            teacher_model, student_model, args.distill_part,
            vocab_size, args.d_model, args.dff, args.num_heads,
            args.num_layers, args.MAX_LENGTH, device, args.hybrid_output
        )
    
    # 保存配置信息
    config = {
        'temperature': args.temperature,
        'distill_weight': args.distill_weight,
        'distill_part': args.distill_part,
        'd_model': args.d_model,
        'dff': args.dff,
        'num_heads': args.num_heads,
        'student_layers': args.student_layers,
        'teacher_layers': args.num_layers,
        'channel': args.channel,
        'best_loss': best_loss,
        'best_epoch': best_epoch + 1,
        'teacher_params': teacher_params,
        'student_params': student_params,
        'compression_ratio': teacher_params / student_params,
        'create_hybrid': args.create_hybrid,
        'hybrid_output': args.hybrid_output if args.create_hybrid else None,
    }
    
    config_path = os.path.join(args.output_path, 'distillation_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"✓ Configuration saved to {config_path}\n")
    
    return student_model, hybrid_model, config


if __name__ == '__main__':
    args = parser.parse_args()
    
    print("="*70)
    print("Knowledge Distillation with Temperature Control for DeepSC")
    print("="*70)
    print(f"\nConfiguration:")
    print(f"  Temperature: {args.temperature}")
    print(f"  Distillation Weight: {args.distill_weight}")
    print(f"  Distillation Part: {args.distill_part}")
    print(f"  Create Hybrid Model: {args.create_hybrid}")
    print(f"  Channel: {args.channel}")
    print(f"  Student Layers: {args.student_layers}")
    print(f"  Learning Rate: {args.lr}")
    print(f"  Batch Size: {args.batch_size}")
    print(f"  Epochs: {args.epochs}\n")
    
    # 开始蒸馏训练
    student_model, hybrid_model, config = train_distillation(args)
    
    print("="*70)
    print("Training Complete!")
    print("="*70)
    print("\nSummary:")
    print(f"  Student model saved: {os.path.join(args.output_path, 'final_student_model.pt')}")
    if args.create_hybrid:
        print(f"  Hybrid model saved: {args.hybrid_output}")
    print(f"  Configuration saved: {os.path.join(args.output_path, 'distillation_config.json')}")
    print(f"\nKey Metrics:")
    print(f"  Distillation Part: {config['distill_part']}")
    print(f"  Temperature: {config['temperature']}")
    print(f"  Distillation Weight: {config['distill_weight']}")
    print(f"  Best Loss: {config['best_loss']:.5f}")
    print(f"  Best Epoch: {config['best_epoch']}")
    print(f"  Compression Ratio: {config['compression_ratio']:.2f}x")
    print("="*70)
