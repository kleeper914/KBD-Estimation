# 知识蒸馏脚本使用指南

## 概述

该脚本实现了针对DeepSC语义通信系统的**参数可控的知识蒸馏**，通过控制温度参数T来精细控制学生模型与教师模型的知识差异程度。支持创建具有收发端知识库差异的混合模型。

## 系统架构

```
DeepSC (Encoder-Decoder)
├── Encoder (发端)
│   └── 语义编码
├── Channel (信道)
│   ├── Channel Encoder (信道编码)
│   └── Channel Decoder (信道解码)
└── Decoder (收端)
    └── 语义解码
```

## 核心功能

### 1. 温度参数控制 (Temperature Control)

**温度参数T** 控制蒸馏知识的"软硬"程度：

- **T 越大** (如 8.0)：输出分布越平滑，学生模型学到"软"知识，收发端语义差异小
- **T 越小** (如 2.0)：分布越尖锐，学生模型学到"硬"知识，收发端语义差异大
- **T 默认值** (4.0)：平衡的蒸馏策略

**数学公式：**

$$P_{\text{soft}} = \text{softmax}\left(\frac{z}{T}\right)$$

蒸馏损失使用KL散度：
$$L_{\text{distill}} = \text{KL}(P_{\text{teacher}} || P_{\text{student}}) \times T^2$$

### 2. 部分蒸馏 (Partial Distillation)

支持三种蒸馏模式：

| 模式 | 参数值 | 说明 |
|------|------|------|
| **完全蒸馏** | `--distill-part full` | 蒸馏整个模型 |
| **发端蒸馏** | `--distill-part encoder` | 仅蒸馏Encoder（发端） |
| **收端蒸馏** | `--distill-part decoder` | 仅蒸馏Decoder（收端） |

### 3. 混合模型创建 (Hybrid Model)

通过分离蒸馏，创建具有知识差异的发收端系统：

```
混合模型构成方式：
- 原始Encoder + 蒸馏Decoder → 发收端知识库不同的系统
- 蒸馏Encoder + 原始Decoder → 另一种知识差异组合
```

## 使用示例

### 基础蒸馏

```bash
python distillation.py \
    --teacher-checkpoint checkpoints/deepsc-Rayleigh/model.pt \
    --temperature 4.0 \
    --distill-weight 0.7 \
    --student-layers 2 \
    --epochs 80
```

### 创建发收端知识差异系统 - 收端蒸馏

```bash
python distillation.py \
    --teacher-checkpoint checkpoints/deepsc-Rayleigh/model.pt \
    --temperature 4.0 \
    --distill-weight 0.7 \
    --student-layers 4 \
    --distill-part decoder \
    --create-hybrid \
    --hybrid-output checkpoints/hybrid_decoder_distilled.pt \
    --output-path checkpoints/distillation_decoder \
    --epochs 80
```

### 创建发收端知识差异系统 - 发端蒸馏

```bash
python distillation.py \
    --teacher-checkpoint checkpoints/deepsc-Rayleigh/model.pt \
    --temperature 4.0 \
    --distill-weight 0.7 \
    --student-layers 4 \
    --distill-part encoder \
    --create-hybrid \
    --hybrid-output checkpoints/hybrid_encoder_distilled.pt \
    --output-path checkpoints/distillation_encoder \
    --epochs 80
```

### 软知识蒸馏（收发端知识库差异小）

```bash
python distillation.py \
    --teacher-checkpoint checkpoints/deepsc-Rayleigh/model.pt \
    --temperature 8.0 \
    --distill-weight 0.8 \
    --distill-part decoder \
    --create-hybrid \
    --hybrid-output checkpoints/hybrid_soft_distilled.pt \
    --output-path checkpoints/distillation_soft \
    --epochs 80
```

### 硬知识蒸馏（收发端知识库差异大）

```bash
python distillation.py \
    --teacher-checkpoint checkpoints/deepsc-Rayleigh/model.pt \
    --temperature 2.0 \
    --distill-weight 0.5 \
    --distill-part decoder \
    --create-hybrid \
    --hybrid-output checkpoints/hybrid_hard_distilled.pt \
    --output-path checkpoints/distillation_hard \
    --epochs 80
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|------|------|
| `--vocab-file` | `europarl/vocab.json` | 词汇表文件路径 |
| `--teacher-checkpoint` | `checkpoints/deepsc-Rayleigh/model.pt` | 教师模型检查点 |
| `--student-checkpoint` | `''` | 学生模型初始检查点（可选） |
| `--output-path` | `checkpoints/distillation` | 学生模型保存路径 |
| `--channel` | `Rayleigh` | 信道类型（AWGN/Rayleigh/Rician） |
| `--d-model` | `128` | 模型维度 |
| `--dff` | `512` | 前馈层维度 |
| `--num-layers` | `4` | 教师模型层数 |
| `--num-heads` | `8` | 注意力头数 |
| `--batch-size` | `128` | 批次大小 |
| `--epochs` | `80` | 训练轮数 |
| **`--temperature`** | **4.0** | **温度参数（控制知识软硬）** |
| **`--distill-weight`** | **0.7** | **蒸馏损失权重(α)** |
| **`--lr`** | **1e-4** | **学习率** |
| **`--student-layers`** | **2** | **学生模型层数** |
| **`--distill-part`** | **full** | **蒸馏部分(full/encoder/decoder)** |
| **`--create-hybrid`** | - | **是否创建混合模型** |
| **`--hybrid-output`** | `checkpoints/hybrid_model.pt` | **混合模型保存路径** |

## 输出文件说明

```
checkpoints/distillation/
├── final_student_model.pt          # 最终学生模型
├── best_student_model.pt           # 最佳学生模型
├── student_model_epoch10.pt        # 第10轮检查点
├── student_model_epoch20.pt        # 第20轮检查点
└── distillation_config.json        # 蒸馏配置和性能指标

checkpoints/
└── hybrid_model.pt                 # 混合模型（如果创建）
```

### distillation_config.json 说明

```json
{
    "temperature": 4.0,
    "distill_weight": 0.7,
    "distill_part": "decoder",
    "d_model": 128,
    "num_heads": 8,
    "student_layers": 2,
    "teacher_layers": 4,
    "best_loss": 3.2156,
    "best_epoch": 45,
    "teacher_params": 12345678,
    "student_params": 6172839,
    "compression_ratio": 2.0,
    "create_hybrid": true,
    "hybrid_output": "checkpoints/hybrid_model.pt"
}
```

## 损失函数设计

### 总损失公式

$$L_{\text{total}} = (1-\alpha) \cdot L_{\text{task}} + \alpha \cdot L_{\text{distill}}$$

其中：
- $\alpha$ = `--distill-weight` （蒸馏权重）
- $L_{\text{task}}$ = 原始交叉熵损失
- $L_{\text{distill}}$ = KL散度蒸馏损失

### 蒸馏损失

$$L_{\text{distill}} = \text{KL}\left(\text{softmax}\left(\frac{z_{\text{teacher}}}{T}\right) \parallel \text{softmax}\left(\frac{z_{\text{student}}}{T}\right)\right) \times T^2$$

## 工作流程

```
┌─────────────────────────────────┐
│  加载教师模型（原始DeepSC）      │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  创建学生模型（可选更少层数）    │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  前向传播获取两个模型输出         │
│  - 教师输出（固定）              │
│  - 学生输出（可训练）            │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  计算蒸馏损失                     │
│  温度参数T控制知识软硬程度       │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  计算组合损失                     │
│  L = (1-α)·L_task + α·L_distill  │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  反向传播更新学生模型            │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  验证与保存最佳模型              │
└────────────┬────────────────────┘
             │
             ▼
┌─────────────────────────────────┐
│  创建混合模型（可选）            │
│  原始Encoder + 蒸馏Decoder      │
│  或                             │
│  蒸馏Encoder + 原始Decoder      │
└─────────────────────────────────┘
```

## 实验建议

### 研究知识差异的不同温度参数

```bash
# 实验1: T=2.0（硬知识，差异大）
python distillation.py --temperature 2.0 --distill-weight 0.5 --output-path checkpoints/T2.0

# 实验2: T=4.0（平衡）
python distillation.py --temperature 4.0 --distill-weight 0.7 --output-path checkpoints/T4.0

# 实验3: T=8.0（软知识，差异小）
python distillation.py --temperature 8.0 --distill-weight 0.8 --output-path checkpoints/T8.0
```

### 对比不同蒸馏部分的效果

```bash
# 全模型蒸馏
python distillation.py --distill-part full --output-path checkpoints/full

# 发端蒸馏
python distillation.py --distill-part encoder --output-path checkpoints/encoder_only

# 收端蒸馏
python distillation.py --distill-part decoder --output-path checkpoints/decoder_only
```

## 常见问题

### Q: 温度参数应该怎么选择？
**A:** 根据需要的知识差异程度：
- 差异小（软知识）：T ∈ [6.0, 10.0]
- 差异中等（平衡）：T ∈ [3.0, 5.0]
- 差异大（硬知识）：T ∈ [1.0, 2.5]

### Q: 学生模型层数应该设多少？
**A:** 通常设为教师模型层数的50%-70%，以实现有效压缩同时保留性能。

### Q: 蒸馏权重α怎么选？
**A:** 
- α 较大 (0.7-0.9)：更强的蒸馏效果，学生更接近教师
- α 较小 (0.3-0.5)：保留更多任务特定知识

### Q: 如何使用混合模型进行推理？
**A:** 混合模型可以直接用于DeepSC的训练或评估，与普通模型相同：
```python
from models.transceiver import DeepSC
import torch

model = DeepSC(...)
model.load_state_dict(torch.load('checkpoints/hybrid_model.pt'))
# 进行推理...
```

## 引文和参考

关键技术参考：
- 知识蒸馏: Hinton et al., "Distilling the Knowledge in a Neural Network"
- 温度参数: 软化softmax输出用于蒸馏
- 语义通信: DeepSC系统设计

## 许可证

遵循原项目许可证

---

**版本**: 1.0  
**最后更新**: 2026年1月
