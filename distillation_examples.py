#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Knowledge Distillation Usage Examples
知识蒸馏脚本使用示例

该脚本展示了如何使用distillation.py进行各种蒸馏任务
"""

import subprocess
import sys


def run_command(cmd, description):
    """执行命令并显示输出"""
    print("\n" + "="*80)
    print(f"运行: {description}")
    print("="*80)
    print(f"命令: {cmd}\n")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=False)
        if result.returncode != 0:
            print(f"⚠ 命令执行出错，返回码: {result.returncode}")
        else:
            print(f"✓ {description} 完成")
    except Exception as e:
        print(f"✗ 执行失败: {e}")


# ============================================================================
# 示例 1: 基础蒸馏 - 创建压缩模型
# ============================================================================

example1_cmd = """python distillation.py \
    --teacher-checkpoint checkpoints/deepsc-Rayleigh/checkpoint_50.pth \
    --output-path checkpoints/distillation_basic \
    --temperature 4.0 \
    --distill-weight 0.7 \
    --student-layers 2 \
    --epochs 80 \
    --batch-size 128"""


# ============================================================================
# 示例 2: 收端蒸馏 + 创建混合模型
# 场景: 研究发端(原始) + 收端(蒸馏) 的语义通信系统
# ============================================================================

example2_cmd = """python distillation.py \
    --teacher-checkpoint checkpoints/deepsc-Rayleigh/checkpoint_50.pth \
    --output-path checkpoints/distillation_decoder \
    --distill-part decoder \
    --create-hybrid \
    --hybrid-output checkpoints/hybrid_decoder_distilled.pt \
    --temperature 4.0 \
    --distill-weight 0.7 \
    --student-layers 4 \
    --epochs 80"""


# ============================================================================
# 示例 3: 发端蒸馏 + 创建混合模型
# 场景: 研究发端(蒸馏) + 收端(原始) 的语义通信系统
# ============================================================================

example3_cmd = """python distillation.py \
    --teacher-checkpoint checkpoints/deepsc-Rayleigh/checkpoint_50.pth \
    --output-path checkpoints/distillation_encoder \
    --distill-part encoder \
    --create-hybrid \
    --hybrid-output checkpoints/hybrid_encoder_distilled.pt \
    --temperature 4.0 \
    --distill-weight 0.7 \
    --student-layers 4 \
    --epochs 80"""


# ============================================================================
# 示例 4: 软知识蒸馏 (高温度)
# 场景: 发收端语义知识差异小 - 接近的语义理解
# 温度越高，两个模型学到的知识越接近
# ============================================================================

example4_cmd = """python distillation.py \
    --teacher-checkpoint checkpoints/deepsc-Rayleigh/checkpoint_50.pth \
    --output-path checkpoints/distillation_soft \
    --distill-part decoder \
    --create-hybrid \
    --hybrid-output checkpoints/hybrid_soft_knowledge.pt \
    --temperature 8.0 \
    --distill-weight 0.8 \
    --student-layers 4 \
    --epochs 80"""


# ============================================================================
# 示例 5: 硬知识蒸馏 (低温度)
# 场景: 发收端语义知识差异大 - 不同的语义理解
# 温度越低，两个模型学到的知识差异越大
# ============================================================================

example5_cmd = """python distillation.py \
    --teacher-checkpoint checkpoints/deepsc-Rayleigh/checkpoint_50.pth \
    --output-path checkpoints/distillation_hard \
    --distill-part decoder \
    --create-hybrid \
    --hybrid-output checkpoints/hybrid_hard_knowledge.pt \
    --temperature 2.0 \
    --distill-weight 0.5 \
    --student-layers 4 \
    --epochs 80"""


# ============================================================================
# 示例 6: 完整蒸馏实验套件
# 场景: 对比不同温度参数的效果
# ============================================================================

example6_cmds = [
    ("""python distillation.py \
        --teacher-checkpoint checkpoints/deepsc-Rayleigh/checkpoint_50.pth \
        --output-path checkpoints/temp_exp/T2.0 \
        --distill-part decoder \
        --create-hybrid \
        --hybrid-output checkpoints/temp_exp/hybrid_T2.0.pt \
        --temperature 2.0 \
        --distill-weight 0.5 \
        --student-layers 4 \
        --epochs 40""", "温度 T=2.0 (硬知识)"),
    
    ("""python distillation.py \
        --teacher-checkpoint checkpoints/deepsc-Rayleigh/checkpoint_50.pth \
        --output-path checkpoints/temp_exp/T4.0 \
        --distill-part decoder \
        --create-hybrid \
        --hybrid-output checkpoints/temp_exp/hybrid_T4.0.pt \
        --temperature 4.0 \
        --distill-weight 0.7 \
        --student-layers 4 \
        --epochs 40""", "温度 T=4.0 (平衡)"),
    
    ("""python distillation.py \
        --teacher-checkpoint checkpoints/deepsc-Rayleigh/checkpoint_50.pth \
        --output-path checkpoints/temp_exp/T8.0 \
        --distill-part decoder \
        --create-hybrid \
        --hybrid-output checkpoints/temp_exp/hybrid_T8.0.pt \
        --temperature 8.0 \
        --distill-weight 0.8 \
        --student-layers 4 \
        --epochs 40""", "温度 T=8.0 (软知识)"),
]


# ============================================================================
# 示例 7: 针对不同信道的蒸馏
# ============================================================================

example7_cmds = [
    ("""python distillation.py \
        --teacher-checkpoint checkpoints/deepsc-AWGN/checkpoint_50.pth \
        --output-path checkpoints/channel_exp/AWGN \
        --channel AWGN \
        --distill-part decoder \
        --temperature 4.0 \
        --epochs 40""", "AWGN信道蒸馏"),
    
    ("""python distillation.py \
        --teacher-checkpoint checkpoints/deepsc-Rician/checkpoint_50.pth \
        --output-path checkpoints/channel_exp/Rician \
        --channel Rician \
        --distill-part decoder \
        --temperature 4.0 \
        --epochs 40""", "Rician信道蒸馏"),
]


if __name__ == '__main__':
    print("\n" + "="*80)
    print("知识蒸馏脚本 - 使用示例")
    print("="*80)
    
    print("\n可用的示例:")
    print("  1. 基础蒸馏 - 创建压缩模型")
    print("  2. 收端蒸馏 + 混合模型 (发端原始 + 收端蒸馏)")
    print("  3. 发端蒸馏 + 混合模型 (发端蒸馏 + 收端原始)")
    print("  4. 软知识蒸馏 (T=8.0, 知识差异小)")
    print("  5. 硬知识蒸馏 (T=2.0, 知识差异大)")
    print("  6. 完整温度实验套件 (T=2.0/4.0/8.0对比)")
    print("  7. 不同信道蒸馏实验")
    print("  all. 运行所有示例")
    
    # 如果有命令行参数，使用该示例
    if len(sys.argv) > 1:
        choice = sys.argv[1]
    else:
        choice = input("\n请选择要运行的示例 (1-7 或 all): ").strip()
    
    if choice == '1':
        run_command(example1_cmd, "基础蒸馏")
    
    elif choice == '2':
        run_command(example2_cmd, "收端蒸馏 + 混合模型")
    
    elif choice == '3':
        run_command(example3_cmd, "发端蒸馏 + 混合模型")
    
    elif choice == '4':
        run_command(example4_cmd, "软知识蒸馏 (T=8.0)")
    
    elif choice == '5':
        run_command(example5_cmd, "硬知识蒸馏 (T=2.0)")
    
    elif choice == '6':
        print("\n运行温度实验套件...")
        for cmd, desc in example6_cmds:
            run_command(cmd, desc)
    
    elif choice == '7':
        print("\n运行信道实验...")
        for cmd, desc in example7_cmds:
            run_command(cmd, desc)
    
    elif choice.lower() == 'all':
        print("\n运行所有示例...")
        run_command(example1_cmd, "示例 1: 基础蒸馏")
        run_command(example2_cmd, "示例 2: 收端蒸馏 + 混合模型")
        run_command(example3_cmd, "示例 3: 发端蒸馏 + 混合模型")
        run_command(example4_cmd, "示例 4: 软知识蒸馏")
        run_command(example5_cmd, "示例 5: 硬知识蒸馏")
        
        print("\n运行温度实验套件...")
        for cmd, desc in example6_cmds:
            run_command(cmd, desc)
        
        print("\n运行信道实验...")
        for cmd, desc in example7_cmds:
            run_command(cmd, desc)
    
    else:
        print("无效的选择")
        sys.exit(1)
    
    print("\n" + "="*80)
    print("所有示例执行完成！")
    print("="*80)
    print("\n提示:")
    print("  - 检查 checkpoints/distillation_*/distillation_config.json 查看蒸馏配置")
    print("  - 检查混合模型文件 checkpoints/hybrid_*.pt")
    print("  - 在main.py中集成这些模型进行性能评测")
    print("="*80)
