# Highway-PPO 训练指南

## 项目概述

这个项目使用 **Stable Baselines3 (PPO)** 算法在 **Highway-Env** 环境中训练强化学习智能体，使其学会自动驾驶。环境包含直道、弯道和复杂的道路场景，智能体需要学会：
- 保持高速行驶
- 避免碰撞
- 主动变道超车

---

## 当前训练问题分析

### 问题现象
1. **智能体不愿意变道**：倾向于保持当前车道，导致频繁追尾
2. **碰撞率高**：即使增加了碰撞惩罚，仍然常常撞车
3. **训练效果差**：即使训练20万步，性能仍不理想

### 根本原因

#### 1. **观察空间信息不足**
- Highway-Env 默认使用 Kinematics 观察（相对位置和速度）
- 智能体看不到完整的交通图景，难以做出前瞻性决策
- **改进**：应使用 `Occupancy Grid` 或 CNN-based 观察，提供更丰富的空间信息

#### 2. **环境难度陡增**
- 之前是简单直道，现在加入了弯道和随机路网
- 弯道本身增加了追尾风险，但没有给智能体足够的"学习缓冲"
- **改进**：应该分阶段训练（课程学习），先学直道，再学弯道

#### 3. **奖励函数设计不当**
- 当前奖励函数：碰撞 `-5`，高速 `+0.5`，变道 `+0.05`
- 问题：变道奖励太小，**智能体可能认为稳定直行比冒险变道更划算**
- 改进思路：
  - 增加变道的**主动奖励**（不仅仅是鼓励，而是有必要时强制引导）
  - 根据**车前方距离**动态调整奖励（前方有车则变道更有价值）

#### 4. **训练超参数不优**
- `n_steps=512` 对于多环境并行训练可能不够
- `learning_rate=5e-4` 可能不适合复杂场景
- **改进**：需要更精细的超参数搜索

#### 5. **环境设置问题**
- `custom_env.py` 中的弯道生成可能不够平滑
- 其他车辆的行为可能不够智能，容易卡住或堵车
- **改进**：优化环境参数，降低环境本身的"噪声"

---

## 改进策略（按优先级）

### 优先级 1：改用更好的观察空间
```python
# 将观察从 Kinematics 改为 Occupancy Grid
config = {
    "observation": {
        "type": "OccupancyGrid",
        "vehicles_count": 15,
        "features": ["presence", "on_road"],
        "grid_size": [[-27.5, 27.5], [-27.5, 27.5]],
        "grid_step": 1.75,
    }
}
```
这样智能体能"看到"周围的交通态势，做出更聪明的变道决策。

### 优先级 2：实现课程学习
```python
# 第一阶段：只训练直道（highway-fast-v0）
# 第二阶段：引入简单弯道
# 第三阶段：复杂环境

# 可以通过修改 custom_env.py 中的 segment_type 概率来实现
```

### 优先级 3：改进奖励函数
```python
# 基于前方距离的动态奖励
def _reward(self, action):
    base_reward = standard_reward
    
    # 检测前方碰撞风险
    distance_to_front = check_distance_ahead()
    if distance_to_front < 20:  # 前方太近
        if action == LANE_LEFT or action == LANE_RIGHT:
            base_reward += 2.0  # 大幅奖励变道
        else:
            base_reward -= 1.0  # 惩罚不变道
    
    return base_reward
```

### 优先级 4：调整超参数
```python
# 更激进的学习设置
learning_rate = 1e-3          # 提高学习率
n_steps = 2048                # 更多样本再更新
batch_size = 128              # 增加批量大小
n_epochs = 20                 # 更多迭代
gamma = 0.95                  # 更关注远期

# 增加总训练步数
total_timesteps = 1e6         # 100万步
```

### 优先级 5：增加网络容量
```python
policy_kwargs = dict(
    net_arch=dict(
        pi=[512, 512, 256],   # 更深的策略网络
        vf=[512, 512, 256]    # 更深的价值网络
    )
)
```

---

## 推荐训练流程

### 阶段 1：基础直道训练（推荐）
```bash
# 先用原始的 highway-fast-v0 训练
# 修改 train.py 改用 "highway-fast-v0"
python train.py

# 这会更快地让智能体学会基本驾驶
```

### 阶段 2：迁移到复杂环境
```bash
# 加载已训练的模型，继续在复杂环境上微调
model = PPO.load("model_complex")
model.set_env(complex_env)
model.learn(additional_timesteps=5e5)
```

### 阶段 3：微调和评估
```bash
# 运行 test.py 观察实际表现
python test.py
```

---

## 文件说明

| 文件 | 说明 |
|------|------|
| `train.py` | PPO 训练脚本，使用多进程并行环境 |
| `test.py` | 测试脚本，加载模型并可视化智能体行为 |
| `custom_env.py` | 自定义复杂环境（包含弯道和随机路段） |
| `model_complex.zip` | 训练好的 PPO 模型参数 |

---

## 快速开始

### 1. 训练
```bash
cd highway_ppo
python train.py
```
预计耗时：
- CPU 仅：20-30 分钟（取决于CPU核心数）
- GPU：5-10 分钟

### 2. 测试
```bash
python test.py
```
会运行 5 个 episode，显示每个 episode 的步数、行驶距离、奖励和是否碰撞。

---

## 性能指标说明

### 理想的训练目标
- **距离**：每个 episode 应该能行驶 **3000+ 米** 不碰撞
- **步数**：至少 **150+ 步** （环境每秒 15 步）
- **成功率**：至少 **80%** 的 episode 能行驶到终点不碰撞

### 当前性能差的迹象
- 距离 < 1000 m
- 步数 < 100
- 频繁碰撞

---

## 下一步改进方向

如果以上基础优化后效果仍不理想，可以考虑：

1. **更换算法**：尝试 SAC、TD3 等连续控制友好的算法
2. **增加观察维度**：使用 CNN 处理图像观察
3. **离线强化学习**：收集一些人类驾驶数据，先做行为克隆
4. **多目标学习**：同时优化速度、安全性和燃油效率
5. **分布式训练**：使用 ray 框架并行加速训练

---

## 常见问题

**Q: 为什么训练这么慢？**
A: 强化学习需要大量与环境的交互。PPO 是样本效率较低的算法，但更稳定。可以用 GPU 加速或增加 CPU 核心数。

**Q: 为什么智能体还是不变道？**
A: 最可能的原因是观察空间不足。Kinematics 观察无法充分表示"需要变道"的情景。改用 OccupancyGrid 会大幅改善。

**Q: 能否用预训练模型？**
A: 可以考虑用在简单 highway-v0 上预训练的模型，然后迁移到复杂环境。

---

## 参考资源

- [Stable Baselines3 文档](https://stable-baselines3.readthedocs.io/)
- [Highway-Env 文档](https://highway-env.farama.org/)
- [PPO 论文](https://arxiv.org/abs/1707.06347)
