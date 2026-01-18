# 训练运行指南

## 所有改进已实施

该项目现已实施全部4个优先级的改进：

### ✓ 优先级1：OccupancyGrid 观察
- 使用 `OccupancyGrid` 代替 `Kinematics`，让智能体能看到周围的网格化交通情景
- 自动生成 27.5m × 27.5m 的栅格图，分辨率 1.75m

### ✓ 优先级2：课程学习
- **Stage 1 (50k步)**：在简单直道 `highway-fast-v0` 上预训练，快速学习基础驾驶
- **Stage 2 (150k步)**：在复杂弯道环境上微调，学习复杂场景应对

### ✓ 优先级3：改进奖励函数
- 碰撞惩罚：`-8.0`（从 `-5.0` 提升）
- 变道奖励：`+0.3` (Stage 1) 和 `+0.25` (Stage 2)（从 `+0.05` 提升）
- 探索熵奖励：`ent_coef=0.01`（鼓励更多探索）

### ✓ 优先级4：优化超参数
- 网络架构：`[512, 256]`（从 `[256, 256]` 加深）
- CNN 策略：使用 `CnnPolicy` 处理网格观察
- 学习率：`1e-3`（从 `5e-4` 提升）
- n_steps：`2048`（从 `512` 增加）
- batch_size：`256`（从 `64` 增加）
- n_epochs：`25`（从 `10` 增加）
- gamma：`0.98`（从 `0.9` 增加，关注长期）

---

## 快速运行

### 1️⃣ 训练（完整两阶段）
```bash
cd highway_ppo
python train.py
```

**预计耗时：**
- GPU：15-25 分钟
- CPU (8核)：45-90 分钟

**输出文件：**
- `model_stage1.zip` - 第一阶段模型
- `model_complex.zip` - 最终模型（用于测试）

### 2️⃣ 测试
```bash
python test.py
```

**输出示例：**
```
Model loaded from model_complex

Episode 1 started.
Episode 1 finished.
  Steps: 350
  Distance: 8500.45 m
  Total Reward: 425.32
  Crashed: False

...
```

---

## 分阶段训练（如需调试）

### 仅训练第一阶段
```python
from train import train
train(use_gpu=True, stage="stage1")
```

### 仅训练第二阶段（需先完成stage1）
```python
from train import train
train(use_gpu=True, stage="stage2")
```

---

## 期望改进

| 指标 | 之前 | 预期现在 |
|------|------|---------|
| 单 episode 行驶距离 | < 2000m | > 5000m |
| 碰撞概率 | 80% | < 20% |
| 变道次数 | 很少 | 频繁主动变道 |
| 平均奖励 | < 100 | > 300 |

---

## 可调整参数

如果训练效果不理想，可以修改 `train.py` 中的这些参数：

```python
# Stage 1 调整
env_config = {
    "collision_reward": -8.0,    # ↑ 增加撞车惩罚
    "lane_change_reward": 0.3,   # ↑ 增加变道奖励
}

# Stage 2 调整
total_timesteps = int(15e4)      # ↑ 增加训练步数

# 超参数调整（train_stage_1 中）
learning_rate = 1e-3,            # ↑ 提高学习率
gamma = 0.98,                    # ↑ 关注更长期
ent_coef = 0.01,                 # ↑ 增加探索
```

---

## 常见问题

**Q: 训练太慢了？**
A: 课程学习方案总共200k步（比之前多），但质量更高。可以试试：
- 减少 `n_epochs` 从 25 到 15
- 减少 `batch_size` 从 256 到 128

**Q: 仍然碰撞很多？**
A: 
- 增加 `collision_reward` 到 `-10.0`
- 增加 `lane_change_reward` 到 `0.5`
- 延长 Stage 1 训练步数

**Q: 智能体变道太频繁？**
A: 减少 `lane_change_reward` 值

---

## 监测训练进度

训练会输出类似的日志：
```
======================================================================
STAGE 1: 课程学习第一阶段 - 简单直道 (highway-fast-v0)
======================================================================
Stage 1: 在 6 CPUs 和 CUDA 上训练 50000 步...
...
✓ Stage 1 模型已保存到 model_stage1.zip

======================================================================
STAGE 2: 课程学习第二阶段 - 复杂环境微调 (弯道和合并)
======================================================================
Stage 2: 在复杂环境上继续训练 150000 步...
...
✓ 最终模型已保存到 model_complex.zip
```

祝训练顺利！🚗✨
