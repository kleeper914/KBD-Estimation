# 模型蒸馏系统 - 收发端知识库差异控制

## 项目概述

本项目实现了针对DeepSC语义通信系统的**参数可控的知识蒸馏框架**，通过精细控制温度参数T和蒸馏部分，能够创建具有不同语义知识差异的发收端系统。

## 核心创新

### 1. 温度参数可控的蒸馏机制
- **软知识蒸馏** (T高): 发收端学到相近的语义知识，系统可靠性高
- **硬知识蒸馏** (T低): 发收端学到差异化的语义知识，系统多样性强

### 2. 灵活的部分蒸馏
- **完全蒸馏**: 整个模型压缩和知识迁移
- **发端蒸馏**: 仅蒸馏Encoder，研究发端的语义差异
- **收端蒸馏**: 仅蒸馏Decoder，研究收端的语义差异

### 3. 混合模型创建
自动将蒸馏模型的特定部分与原模型组合，创建发收端知识库差异的系统：

```
┌─────────────────────────────────────────────┐
│  原始教师模型（Baseline）                    │
├─────────────────────────────────────────────┤
│  Encoder (发端) ──┬─→ Channel ──┬─→ Decoder (收端)│
└────────────────┼──────────────┼───────────┘
               ▼              ▼
         蒸馏对象1        蒸馏对象2
                │              │
         [混合策略]────────────┘
                │
                ▼
┌─────────────────────────────────────────────┐
│  混合模型（发收端知识差异系统）              │
├─────────────────────────────────────────────┤
│  Encoder (原始) ──┬─→ Channel ──┬─→ Decoder (蒸馏)│
│  或              │              │  或           │
│  Encoder (蒸馏) ──┴─────────────┴─→ Decoder (原始)│
└─────────────────────────────────────────────┘
```

## 文件说明

### 新增/修改文件

| 文件 | 说明 |
|------|------|
| **distillation.py** | ⭐ 主蒸馏脚本（完全改写） |
| **DISTILLATION_GUIDE.md** | 📖 详细使用指南 |
| **distillation_examples.py** | 🔧 使用示例和实验脚本 |
| **semantic_analysis.py** | 📊 语义差异分析工具 |
| **README_DISTILLATION.md** | 📝 本文件 |

### 修改的原始文件
- 无任何原始文件被修改，完全向后兼容

## 快速开始

### 基础蒸馏

```bash
python distillation.py \
    --teacher-checkpoint checkpoints/deepsc-Rayleigh/checkpoint_50.pth \
    --output-path checkpoints/distillation_basic \
    --temperature 4.0 \
    --student-layers 2 \
    --epochs 80
```

### 创建发收端知识差异系统（收端蒸馏）

```bash
python distillation.py \
    --teacher-checkpoint checkpoints/deepsc-Rayleigh/checkpoint_50.pth \
    --output-path checkpoints/distillation_decoder \
    --distill-part decoder \
    --create-hybrid \
    --hybrid-output checkpoints/hybrid_model.pt \
    --temperature 4.0
```

### 分析语义差异

```bash
python semantic_analysis.py
```

## 主要参数

### 关键参数说明

#### 温度参数 (`--temperature`)
```
T 值       效果            应用场景
────────────────────────────────────
1.0-2.0   硬知识，大差异    探索发收端差异化
3.0-5.0   平衡             通用蒸馏
6.0-10.0  软知识，小差异    保持一致性
```

#### 蒸馏部分 (`--distill-part`)
```
full     → 完整蒸馏，创建压缩模型
encoder  → 仅蒸馏发端 Encoder
decoder  → 仅蒸馏收端 Decoder
```

#### 蒸馏权重 (`--distill-weight`)
- `0.3-0.5`: 更多任务特定知识，蒸馏效果较弱
- `0.6-0.8`: 强蒸馏，学生跟随教师
- `0.9+`: 极强蒸馏，可能过拟合

## 输出结果

### 目录结构
```
checkpoints/
├── distillation_decoder/
│   ├── final_student_model.pt          # 蒸馏后的学生模型
│   ├── best_student_model.pt           # 最佳性能模型
│   ├── student_model_epoch10.pt        # 检查点
│   └── distillation_config.json        # 蒸馏配置记录
└── hybrid_model.pt                     # 混合模型
```

### 配置文件 (distillation_config.json)
```json
{
  "temperature": 4.0,
  "distill_weight": 0.7,
  "distill_part": "decoder",
  "compression_ratio": 2.0,
  "best_loss": 3.2156,
  "best_epoch": 45,
  "teacher_params": 12345678,
  "student_params": 6172839
}
```

## 实验流程

### 第一步：温度参数扫描
研究不同温度参数对发收端知识差异的影响：

```bash
# T=2.0（硬知识）
python distillation.py --temperature 2.0 --output-path exp/T2.0

# T=4.0（平衡）
python distillation.py --temperature 4.0 --output-path exp/T4.0

# T=8.0（软知识）
python distillation.py --temperature 8.0 --output-path exp/T8.0
```

### 第二步：创建混合模型
生成具有特定知识差异的发收端系统：

```bash
python distillation.py \
    --temperature 4.0 \
    --distill-part decoder \
    --create-hybrid \
    --hybrid-output checkpoints/hybrid_T4.0_decoder.pt
```

### 第三步：分析差异
使用分析工具评估模型之间的语义知识差异：

```bash
python semantic_analysis.py
```

## 技术细节

### 蒸馏损失函数

**总损失：**
$$L_{\text{total}} = (1-\alpha) \cdot L_{\text{task}} + \alpha \cdot L_{\text{distill}}$$

**蒸馏损失（KL散度）：**
$$L_{\text{distill}} = KL\left(\text{softmax}\left(\frac{z_t}{T}\right) \parallel \text{softmax}\left(\frac{z_s}{T}\right)\right) \times T^2$$

其中：
- $\alpha$ = 蒸馏权重
- $T$ = 温度参数
- $z_t$ = 教师模型输出 logits
- $z_s$ = 学生模型输出 logits

### 中间层蒸馏

对于部分蒸馏（encoder或decoder），还额外使用中间表示损失：

$$L_{\text{inter}} = \text{MSE}(h_t, h_s)$$

最终蒸馏损失为：
$$L_{\text{distill}} = KL(\cdot) + 0.5 \cdot \text{MSE}(\cdot)$$

## 应用案例

### 案例1：创建多语义版本的通信系统
```
发端知识库：通用编码  (原始模型)
收端知识库：专一解码  (蒸馏模型)
→ 发送端学习通用语义，接收端专注于解码任务
```

### 案例2：研究知识转移的影响
```
对比不同温度的蒸馏模型：
- T=2.0: 大知识差异
- T=4.0: 中等差异
- T=8.0: 小知识差异
→ 分析温度对通信性能的影响
```

### 案例3：模型压缩与知识保留的平衡
```
低温度蒸馏 → 保留发收端差异，模型轻量化
高温度蒸馏 → 保留发收端一致性，提升可靠性
```

## 评估指标

### 语义差异指标 (semantic_analysis.py)

| 指标 | 说明 | 范围 |
|------|------|------|
| **KL散度** | 两个模型输出分布的差异 | [0, ∞) |
| **表示距离** | 中间层表示的欧式距离 | [0, ∞) |
| **一致性** | 预测结果的一致率 | [0, 1] |
| **语义差异分数** | 综合评估指标 | [0, 1] |

## 常见问题

### Q: 如何选择温度参数？
**A:** 根据需求：
- 硬知识（差异大）：T ∈ [1.0, 2.5]
- 平衡：T ∈ [3.0, 5.0]  
- 软知识（差异小）：T ∈ [6.0, 10.0]

### Q: 混合模型怎么使用？
**A:** 与正常模型相同，直接加载权重进行训练或推理。

### Q: 可以同时蒸馏encoder和decoder吗？
**A:** 可以，使用 `--distill-part full` 实现完全蒸馏。

### Q: 如何集成到main.py中？
**A:** 将混合模型路径作为检查点加载到DeepSC中，与原始训练流程相同。

## 引用

如果使用本项目，请引用：

```bibtex
@article{distillation_deepsc,
  title={Temperature-Controlled Knowledge Distillation for Semantic Communication},
  year={2026}
}
```

## 许可证

遵循原项目许可证

---

**版本**: 1.0  
**最后更新**: 2026年1月21日  
**作者**: Knowledge Distillation Module
