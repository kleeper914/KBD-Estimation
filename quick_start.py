#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
📌 欢迎使用 DeepSC 知识蒸馏系统
🎯 快速入门指南

本文件提供交互式的快速入门体验
"""

def print_header():
    print("\n" + "="*80)
    print("🚀 DeepSC 知识蒸馏系统 - 快速入门")
    print("="*80)

def print_menu():
    print("\n请选择您的身份：\n")
    print("  [1] 我是新手，想快速了解这个系统")
    print("  [2] 我是研究人员，想进行对比实验")
    print("  [3] 我是开发者，想深入理解代码")
    print("  [4] 我想运行一个完整的例子")
    print("  [5] 显示所有可用文件说明")
    print("  [0] 退出")

def guide_beginner():
    print("\n" + "-"*80)
    print("👶 新手入门指南 (推荐时间: 15分钟)")
    print("-"*80)
    
    print("""
1️⃣  阅读快速参考卡 (5分钟)
    📄 文件: QUICK_REFERENCE.md
    📝 内容: 快速命令、参数速查、常见问题
    💡 命令: cat QUICK_REFERENCE.md

2️⃣  运行第一个蒸馏任务 (5分钟)
    🔧 命令:
    python distillation.py \\
        --epochs 10 \\
        --batch-size 64
    
    ⏱️  预计时间: 根据GPU性能，通常5-10分钟

3️⃣  查看结果 (2分钟)
    📊 检查输出文件:
    cat checkpoints/distillation/distillation_config.json
    
    📈 关键指标:
    - best_loss: 最佳验证损失
    - compression_ratio: 模型压缩比
    - best_epoch: 最佳轮数

4️⃣  创建混合模型 (3分钟)
    🔀 命令:
    python distillation.py \\
        --create-hybrid \\
        --hybrid-output checkpoints/my_hybrid_model.pt \\
        --epochs 10
    
    ✨ 结果: 
    - checkpoints/distillation/final_student_model.pt (学生模型)
    - checkpoints/my_hybrid_model.pt (混合模型)

下一步: 阅读 README_DISTILLATION.md 了解更多功能
""")

def guide_researcher():
    print("\n" + "-"*80)
    print("🔬 研究人员实验指南 (推荐时间: 1小时)")
    print("-"*80)
    
    print("""
实验设计框架：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

阶段1: 温度参数扫描 (30分钟)
    🎯 目标: 研究不同温度对知识差异的影响
    
    📝 命令:
    python distillation_examples.py 6
    
    📊 生成模型:
    - checkpoints/temp_exp/T2.0/   (硬蒸馏)
    - checkpoints/temp_exp/T4.0/   (平衡)
    - checkpoints/temp_exp/T8.0/   (软蒸馏)

阶段2: 生成混合模型 (15分钟)
    🎯 目标: 创建发收端知识差异的系统
    
    📝 命令:
    python distillation.py \\
        --temperature 2.0 \\
        --distill-part decoder \\
        --create-hybrid \\
        --hybrid-output checkpoints/hybrid_T2.0.pt \\
        --output-path checkpoints/exp_hard
    
    # 为每个温度重复上述命令，替换 T=2.0/4.0/8.0

阶段3: 语义差异分析 (15分钟)
    🎯 目标: 定量评估模型之间的差异
    
    📝 命令:
    python semantic_analysis.py
    
    📊 输出指标:
    - KL散度: 输出分布差异
    - 表示距离: 中间层差异
    - 一致性: 预测相似度
    - 综合分数: 总体差异评分

实验结果分析：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

预期结果:
✓ T低(2.0)  → KL散度大  → 知识差异大
✓ T中(4.0)  → KL散度中  → 知识差异中
✓ T高(8.0)  → KL散度小  → 知识差异小

论文素材生成:
📄 文件: semantic_analysis_results.json
   包含所有关键指标，可直接用于论文

建议论证角度:
1. 温度参数的有效性
2. 发收端知识差异的量化
3. 不同蒸馏策略的对比
4. 混合模型的性能特性
""")

def guide_developer():
    print("\n" + "-"*80)
    print("👨‍💻 开发者深入指南 (推荐时间: 2小时)")
    print("-"*80)
    
    print("""
代码理解路线：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

第1步: 理解核心类和函数 (30分钟)
    📄 文件: distillation.py
    
    关键类/函数:
    • calculate_distillation_loss()     - KL散度蒸馏损失
    • calculate_intermediate_distillation_loss() - 中间层损失
    • forward_pass()                   - 前向传播
    • distillation_step()              - 蒸馏训练步骤
    • create_hybrid_model()            - 混合模型生成 ⭐
    • validate_distillation()          - 验证函数
    • train_distillation()             - 主训练循环
    
    💡 提示: 每个函数都有详细的中文注释

第2步: 理解蒸馏算法 (30分钟)
    📄 文件: DISTILLATION_GUIDE.md
    
    章节: "技术细节"
    
    关键公式:
    ├─ 总损失公式: L = (1-α)·L_task + α·L_distill
    ├─ 蒸馏损失: L_distill = KL(soft_teacher || soft_student) × T²
    └─ 中间层损失: L_inter = MSE(h_teacher, h_student)

第3步: 追踪执行流程 (30分钟)
    🔄 流程图在 DISTILLATION_GUIDE.md 中
    
    主流程:
    1. 加载教师模型 (冻结)
    2. 创建学生模型 (可训练)
    3. FOR 每个epoch:
       a) 前向传播两个模型
       b) 计算蒸馏损失和任务损失
       c) 反向传播更新学生
       d) 验证并保存最佳模型
    4. 创建混合模型 (可选)

第4步: 扩展功能 (30分钟)
    🛠️ 可能的改进方向:
    
    ├─ 多教师蒸馏
    │  修改: forward_pass() 支持多个教师
    │
    ├─ 自适应温度
    │  修改: 在训练过程中动态调整 T
    │
    ├─ 对抗性蒸馏
    │  新增: 判别器网络
    │
    └─ 特征匹配
       新增: 多层中间特征对齐

代码快速导航：
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

文件结构:
distillation.py
├─ 导入和设备配置 (第1-30行)
├─ 命令行参数定义 (第31-60行)
├─ 工具函数 (第61-120行)
│  ├─ setup_seed()
│  ├─ calculate_distillation_loss()
│  └─ calculate_intermediate_distillation_loss()
├─ 前向传播 (第121-160行)
│  └─ forward_pass()
├─ 蒸馏训练 (第161-260行)
│  ├─ distillation_step()
│  └─ validate_distillation()
├─ 混合模型 (第261-320行)
│  └─ create_hybrid_model() ⭐
├─ 主训练循环 (第321-500行)
│  └─ train_distillation()
└─ 主程序 (第501-550行)
   └─ if __name__ == '__main__':
""")

def guide_example():
    print("\n" + "-"*80)
    print("🔧 完整示例运行指南 (推荐时间: 30分钟)")
    print("-"*80)
    
    print("""
快速示例 1: 基础蒸馏 (5分钟)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

命令:
python distillation.py \\
    --epochs 5 \\
    --batch-size 32

期望输出:
✓ 开始蒸馏训练
✓ 每个epoch显示: Loss, Task Loss, Distill Loss
✓ 保存: checkpoints/distillation/final_student_model.pt

检查结果:
cat checkpoints/distillation/distillation_config.json

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

快速示例 2: 创建混合模型 (10分钟)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

命令:
python distillation.py \\
    --distill-part decoder \\
    --create-hybrid \\
    --hybrid-output checkpoints/my_hybrid.pt \\
    --temperature 4.0 \\
    --epochs 5

期望输出:
✓ 开始蒸馏 (仅decoder)
✓ 保存最终模型
✓ 创建混合模型 (原始encoder + 蒸馏decoder)
✓ 混合模型保存: checkpoints/my_hybrid.pt

检查结果:
ls -lh checkpoints/my_hybrid.pt
cat checkpoints/distillation_decoder/distillation_config.json

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

快速示例 3: 完整实验套件 (15分钟)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

命令:
python distillation_examples.py 6

期望:
✓ 运行三个温度的蒸馏: T=2.0, 4.0, 8.0
✓ 生成模型: checkpoints/temp_exp/T*
✓ 每个模型都保存配置文件

验证:
ls -R checkpoints/temp_exp/

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

快速示例 4: 分析结果 (5分钟)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

命令:
python semantic_analysis.py

期望输出:
✓ 加载模型
✓ 计算语义差异
✓ 显示详细报告
✓ 保存结果: semantic_analysis_results.json

查看结果:
cat semantic_analysis_results.json
""")

def print_files():
    print("\n" + "-"*80)
    print("📁 项目文件完整说明")
    print("-"*80)
    
    files = [
        ("distillation.py", "核心蒸馏脚本", "~600行"),
        ("distillation_examples.py", "实验示例脚本", "~300行"),
        ("semantic_analysis.py", "差异分析工具", "~250行"),
        ("", "", ""),
        ("DISTILLATION_GUIDE.md", "详细使用指南", "~500行"),
        ("README_DISTILLATION.md", "项目概览文档", "~400行"),
        ("QUICK_REFERENCE.md", "快速参考卡片", "~300行"),
        ("COMPLETION_SUMMARY.md", "修改完成总结", "~200行"),
        ("FILE_STRUCTURE.md", "文件结构说明", "~300行"),
        ("", "", ""),
        ("utils.py", "[原有] 工具函数", ""),
        ("main.py", "[原有] 主程序", ""),
        ("dataset.py", "[原有] 数据集", ""),
        ("models/transceiver.py", "[原有] DeepSC模型", ""),
    ]
    
    print(f"\n{'文件名':<30} {'说明':<20} {'大小':<15}")
    print("-" * 65)
    
    for filename, desc, size in files:
        if filename:
            print(f"{filename:<30} {desc:<20} {size:<15}")
        else:
            print()
    
    print("\n💡 提示:")
    print("  ✓ 核心代码: distillation.py (完全重写)")
    print("  ✓ 详细指南: DISTILLATION_GUIDE.md")
    print("  ✓ 快速参考: QUICK_REFERENCE.md")
    print("  ✓ 实验脚本: distillation_examples.py")
    print("  ✓ 分析工具: semantic_analysis.py")
    print("  ✓ 原有文件: 完全兼容，无修改")

def print_tips():
    print("\n" + "="*80)
    print("💡 使用提示")
    print("="*80)
    print("""
✓ 快速命令 (复制粘贴运行):
  python distillation.py --epochs 20 --temperature 4.0 --create-hybrid

✓ 查看帮助:
  python distillation.py --help

✓ 查看文档:
  cat QUICK_REFERENCE.md          # 快速参考
  cat DISTILLATION_GUIDE.md       # 详细指南
  cat README_DISTILLATION.md      # 项目概览

✓ 重要参数:
  --temperature (2.0-10.0)        # 温度参数
  --distill-weight (0.3-0.9)      # 蒸馏强度
  --distill-part (full/encoder/decoder)  # 蒸馏部分
  --create-hybrid                 # 生成混合模型

✓ 查看结果:
  cat checkpoints/distillation/distillation_config.json

✓ 常见问题:
  1. CUDA内存不足 → 降低 --batch-size
  2. 损失不下降 → 增加 --temperature
  3. 模型加载失败 → 检查 --teacher-checkpoint 路径
  4. 混合模型不生成 → 添加 --create-hybrid 参数
""")

def main():
    while True:
        print_header()
        print_menu()
        
        choice = input("\n请选择 (0-5): ").strip()
        
        if choice == '1':
            guide_beginner()
        elif choice == '2':
            guide_researcher()
        elif choice == '3':
            guide_developer()
        elif choice == '4':
            guide_example()
        elif choice == '5':
            print_files()
        elif choice == '0':
            print("\n👋 感谢使用！再见！\n")
            break
        else:
            print("❌ 无效选择，请重试")
            continue
        
        input("\n按 Enter 继续...")

if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️  程序被中断")
    except Exception as e:
        print(f"\n❌ 出错: {e}")
