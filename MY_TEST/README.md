# Highway-Env Multi-Scenario Training Framework

这是一个基于 Stable Baselines3 的强化学习框架，用于在 highway-env 环境中训练和测试 PPO (Proximal Policy Optimization) 智能体，支持多个场景的独立训练与评估。

## 文件结构

```
MY_TEST/
├── train_highway_ppo.py          # 多场景训练脚本
├── test_highway_ppo.py           # 多场景测试脚本
├── highway_ppo/                  # 训练结果存储目录（自动创建）
│   ├── highway_merge/            # 高速公路+汇入场景
│   │   └── run_N/
│   │       ├── model.zip         # Stable Baselines3 模型
│   │       ├── model.pt          # PyTorch 策略参数
│   │       └── events.out...     # TensorBoard 日志
│   ├── roundabout/               # 环岛场景
│   │   └── run_N/
│   └── parking/                  # 停车场景
│       └── run_N/
├── video/                        # 测试时录制的视频（自动创建）
├── evaluation_results.png        # 评估结果可视化
└── README.md                     # 本文件
```

## 支持的场景

### Scenario 1: Highway + Merge (高速公路与汇入)

- **环境**: `highway-fast-v0` 和 `merge-v0` 混合训练
- **动作**: 离散动作 (5 种)
  - 0: 向左变道
  - 1: 保持不变
  - 2: 向右变道
  - 3: 加速
  - 4: 减速
- **评价指标**: 最大行驶距离 (m)
- **失败条件**: 发生碰撞

### Scenario 2: Roundabout (环岛通过)

- **环境**: `roundabout-v0`
- **动作**: 离散动作 (5 种)
- **评价指标**: 最短通过时间 (秒)
- **成功条件**: 安全通过环岛（Y坐标 < -20）
- **失败条件**: 发生碰撞或超时未通过

### Scenario 3: Parking (自动停车)

- **环境**: `parking-v0`
- **动作**: 连续动作 (2 维)
  - 方向盘转角 (-45° ~ +45°)
  - 加速度 (-1.0 ~ 1.0)
- **评价指标**: 最短停车时间 (秒)
- **成功条件**: 倒车入库成功（停到蓝色车位）
- **失败条件**: 发生碰撞或超时未停好

## 使用方法

### 环境依赖

```bash
pip install gymnasium stable-baselines3 highway-env tensorboard torch
```

### 1. 训练

在 `MY_TEST` 目录下执行训练脚本，支持三种场景：

#### 训练场景 1 (高速+汇入) - 建议首选

```bash
cd MY_TEST
python train_highway_ppo.py --scenario 1
```

#### 训练场景 2 (环岛)

```bash
python train_highway_ppo.py --scenario 2
```

#### 训练场景 3 (停车) - 耗时最长

```bash
python train_highway_ppo.py --scenario 3
```

**训练参数说明**:

- 总训练步数: 300,000 步
- 并行环境: 6 个
- 批次大小: 64
- 神经网络结构: [256, 256] (策略) + [256, 256] (值函数)
- 学习率: 5e-4
- 折扣因子 (gamma): 0.8

**输出**:

```
Output directory for this run: /home/tsui/src/HighwayEnv/MY_TEST/highway_ppo/scenario_name/run_N
Training finished.
Model saved to .../run_N/model.zip
Policy saved to .../run_N/model.pt
```

### 2. 测试

#### 测试最新训练的模型 (自动加载该场景最新的 run_N)

```bash
python test_highway_ppo.py --scenario 1
```

#### 测试指定编号的模型

```bash
python test_highway_ppo.py --scenario 2 --run_id 1
```

#### 完整用法示例

```bash
# 测试场景 1，使用 run_3
python test_highway_ppo.py --scenario 1 --run_id 3

# 测试场景 3，使用最新模型
python test_highway_ppo.py --scenario 3
```

**测试参数说明**:

- `--scenario`: 选择测试场景 (1, 2, 3) - 默认为 1
- `--run_id`: 指定要加载的运行编号 (可选，默认加载最新)

**输出样例**:

场景 1 (高速公路):

```
Episode 1: Distance: 125.34 m
Episode 2: FAILED (Crash)
Episode 3: Distance: 145.67 m
...
```

场景 2 (环岛):

```
Episode 1: SUCCESS - Time: 3.45 s
Episode 2: FAILED (Crash)
Episode 3: SUCCESS - Time: 2.98 s
...
```

场景 3 (停车):

```
Episode 1: SUCCESS - Time: 8.23 s
Episode 2: FAILED (Crash)
Episode 3: SUCCESS - Time: 7.15 s
...
```

## 目录说明

### highway_ppo/ (自动创建)

每个场景独立存储训练结果，按编号自增管理：

- **highway_merge/run_1, run_2, ...**: 高速+汇入场景的各次训练
- **roundabout/run_1, run_2, ...**: 环岛场景的各次训练
- **parking/run_1, run_2, ...**: 停车场景的各次训练

每个 `run_N/` 包含:

- `model.zip`: Stable Baselines3 模型文件
- `model.pt`: PyTorch 神经网络权重
- `events.out.tfevents.*`: TensorBoard 日志

### video/ (自动创建)

测试时录制的环境播放视频（MP4 格式）

### 其他文件

- `evaluation_results.png`: 评估统计图表
- `train_highway_ppo.py`: 训练脚本
- `test_highway_ppo.py`: 测试脚本

## 核心特性

✅ **多场景独立管理**: 不同场景的训练结果完全隔离
✅ **自动版本编号**: 每次训练自动生成递增的 run_N 编号
✅ **灵活的模型加载**: 支持加载指定编号或最新的模型
✅ **详细的评价指标**: 每个场景有针对性的评价标准
✅ **视频录制**: 自动保存测试过程的视频
✅ **TensorBoard 支持**: 实时监控训练进度

## 高级用法

### 查看 TensorBoard 训练曲线

```bash
cd MY_TEST
tensorboard --logdir=highway_ppo/highway_merge/run_1
```

然后在浏览器中打开 `http://localhost:6006`

### 对比不同场景的训练效果

```bash
tensorboard --logdir=highway_ppo
```

### 加载模型进行自定义推理

```python
from stable_baselines3 import PPO

model = PPO.load("MY_TEST/highway_ppo/highway_merge/run_1/model")
obs = env.reset()
while True:
    action, _ = model.predict(obs)
    obs, reward, done, truncated, info = env.step(action)
```

## 注意事项

1. **GPU vs CPU**: PPO 在 CPU 上训练效果更好。如需强制使用 CPU：

   ```bash
   export CUDA_VISIBLE_DEVICES=
   python train_highway_ppo.py --scenario 1
   ```
2. **训练时间**:

   - 场景 1 & 2: ~10-20 分钟 (GPU), ~30-60 分钟 (CPU)
   - 场景 3: ~20-40 分钟 (GPU), ~60-120 分钟 (CPU)
3. **内存需求**:

   - 最少: 4GB RAM
   - 推荐: 8GB+ RAM (用于 6 个并行环境)
4. **碰撞判定**:

   - 场景 1&2: 任何碰撞立即停止评估
   - 场景 3: 碰撞后无法获得成功分数

## 常见问题

**Q: 训练失败显示"找不到模型"**
A: 确保已完成训练。路径应为 `MY_TEST/highway_ppo/scenario_name/run_N/`

**Q: 如何只训练一个特定场景？**
A: 使用 `--scenario` 参数指定，其他场景的结果不会被清除。

**Q: 测试时总是超时（Timeout）**
A: 模型未收敛，需要增加训练步数或调整超参数。

**Q: 停车场景一直失败**
A: 停车是最难的任务。确保训练步数足够（建议 > 500K），或增加训练时长。

## 可调整参数详解

本框架的所有关键超参数都设计为易于修改。以下是完整的参数说明及推荐值。

### 训练参数 (train_highway_ppo.py)

#### 1. 训练步数 (train_timesteps)

**位置**: `train_highway_ppo.py` 第 138 行

```python
train_timesteps = 300000  # 修改这个值
```

| 参数值            | 训练时间             | 效果           | 适用场景           |
| ----------------- | -------------------- | -------------- | ------------------ |
| 10,000            | ~1-2 分钟            | 快速测试       | 调试参数           |
| 50,000            | ~5-10 分钟           | 初步学习       | 场景 1, 2          |
| **300,000** | **30-60 分钟** | **推荐** | **所有场景** |
| 500,000           | ~60-120 分钟         | 高精度         | 场景 3 (停车)      |
| 1,000,000         | 2+ 小时              | 极高精度       | 对精度要求高       |

**推荐修改**:

- 快速实验: 改为 `50000`
- 生产环境: 改为 `500000` ~ `1000000`

#### 2. 并行环境数 (n_cpu)

**位置**: `train_highway_ppo.py` 第 73 行

```python
n_cpu = 6  # 修改这个值
```

| 参数值      | 内存占用       | 训练速度           | 建议             |
| ----------- | -------------- | ------------------ | ---------------- |
| 2           | ~2GB           | 较慢               | 内存紧张         |
| 4           | ~3GB           | 正常               | 标准配置         |
| **6** | **~4GB** | **推荐快速** | **默认值** |
| 8           | ~5-6GB         | 很快               | 高配置机器       |
| 12+         | 6GB+           | 最快               | 服务器级         |

**推荐修改**:

- 低配机器 (4GB RAM): 改为 `2` 或 `4`
- 高配机器 (16GB+ RAM): 改为 `8` 或 `12`

#### 3. 批次大小 (batch_size)

**位置**: `train_highway_ppo.py` 第 74 行

```python
batch_size = 64  # 修改这个值
```

| 参数值       | 更新频率       | 梯度稳定性     | 建议               |
| ------------ | -------------- | -------------- | ------------------ |
| 32           | 更频繁         | 不稳定         | 小数据集           |
| **64** | **适中** | **稳定** | **默认推荐** |
| 128          | 不频繁         | 很稳定         | 数据多             |
| 256          | 很不频繁       | 过度稳定       | 大数据集           |

**推荐修改**:

- 默认保持 `64` 不动
- 如果训练不稳定: 改为 `128`
- 快速实验: 改为 `32`

#### 4. 学习率 (learning_rate)

**位置**: `train_highway_ppo.py` 第 126 行

```python
learning_rate=5e-4,  # 修改这个值，即 0.0005
```

| 参数值         | 学习速度       | 收敛性         | 建议             |
| -------------- | -------------- | -------------- | ---------------- |
| 1e-5           | 极慢           | 极稳定         | 微调             |
| 1e-4           | 慢             | 很稳定         | 保守训练         |
| **5e-4** | **适中** | **推荐** | **默认值** |
| 1e-3           | 快             | 可能不稳定     | 激进训练         |
| 5e-3           | 很快           | 容易发散       | 不推荐           |

**推荐修改**:

- 如果训练不收敛: 改为 `1e-4` 或 `2e-4`
- 如果训练太慢: 改为 `1e-3`
- 默认保持 `5e-4` 效果最好

#### 5. 折扣因子 (gamma)

**位置**: `train_highway_ppo.py` 第 127 行

```python
gamma=0.8,  # 修改这个值
```

| 参数值        | 远期奖励         | 收敛速度       | 建议             |
| ------------- | ---------------- | -------------- | ---------------- |
| 0.9           | 看得很远         | 慢收敛         | 长期任务         |
| **0.8** | **看得远** | **推荐** | **默认值** |
| 0.95          | 看得很远         | 很慢收敛       | 非常长期         |
| 0.99          | 看得非常远       | 极慢收敛       | 极端长期         |

**推荐修改**:

- 通常不需要修改
- 对于长期规划任务: 改为 `0.9` 或 `0.95`

#### 6. PPO 的 Epoch 数 (n_epochs)

**位置**: `train_highway_ppo.py` 第 125 行

```python
n_epochs=10,  # 修改这个值
```

| 参数值       | 每步学习次数   | 样本利用率     | 建议             |
| ------------ | -------------- | -------------- | ---------------- |
| 5            | 1 倍           | 低             | 快速实验         |
| **10** | **2 倍** | **推荐** | **默认值** |
| 20           | 4 倍           | 高             | 样本稀缺         |
| 30+          | 6+ 倍          | 过度拟合       | 不推荐           |

**推荐修改**:

- 通常不需要修改
- 如果样本利用率低: 改为 `20`

#### 7. 神经网络架构 (policy_kwargs)

**位置**: `train_highway_ppo.py` 第 121 行

```python
policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
```

**含义**:

- `pi=[256, 256]`: 策略网络有 2 层隐藏层，每层 256 个神经元
- `vf=[256, 256]`: 值函数网络有 2 层隐藏层，每层 256 个神经元

| 配置                 | 模型大小     | 学习能力       | 建议             |
| -------------------- | ------------ | -------------- | ---------------- |
| `[128, 128]`       | 小           | 低             | 快速实验         |
| **[256, 256]** | **中** | **推荐** | **默认值** |
| `[512, 512]`       | 大           | 高             | 复杂任务         |
| `[64, 64, 64]`     | 大           | 很高           | 停车场景         |

**推荐修改** (停车场景优化):

```python
policy_kwargs=dict(net_arch=dict(pi=[512, 512], vf=[512, 512])),
```

#### 8. 环境持续时间 (duration)

**位置**: `train_highway_ppo.py` 第 61 行

```python
common_config = {
    ...
    "duration": 40,  # 修改这个值
    ...
}
```

| 参数值       | 每轮长度          | 学习难度       | 建议             |
| ------------ | ----------------- | -------------- | ---------------- |
| 20           | ~1.3 秒           | 简单           | 快速学习         |
| **40** | **~2.6 秒** | **推荐** | **默认值** |
| 60           | ~4 秒             | 困难           | 长序列           |
| 100+         | 6+ 秒             | 很困难         | 极长序列         |

**推荐修改**:

- 通常保持默认
- 高速场景: 改为 `60` (更多学习机会)

### 测试参数 (test_highway_ppo.py)

#### 1. 测试轮数 (episodes)

**位置**: `test_highway_ppo.py` 第 138 行

```python
episodes = 5  # 修改这个值
```

| 参数值      | 评估时间            | 统计准确性     | 建议             |
| ----------- | ------------------- | -------------- | ---------------- |
| 1           | ~10 秒              | 极低           | 快速检查         |
| **5** | **~1-2 分钟** | **推荐** | **默认值** |
| 10          | ~2-4 分钟           | 中高           | 对标评估         |
| 20+         | 4+ 分钟             | 很高           | 精确评估         |

**推荐修改**:

- 快速检查: 改为 `2` 或 `3`
- 详细评估: 改为 `10` 或 `20`

#### 2. 测试环境时长 (duration)

**位置**: `test_highway_ppo.py` 第 54 行

```python
common_config = {
    ...
    "duration": 60,  # 修改这个值
    ...
}
```

| 参数值       | 环节长度       | 建议           |
| ------------ | -------------- | -------------- |
| 40           | 同训练时长     | 保持一致       |
| **60** | 给更多时间完成 | **推荐** |
| 100+         | 非常宽松       | 检查最高能力   |

**推荐修改**:

- 保持 `60` (给模型更多机会完成任务)

### 场景特定优化建议

#### 场景 1 (高速+汇入)

推荐配置:

```python
train_timesteps = 200000
n_cpu = 6
batch_size = 64
learning_rate = 5e-4
gamma = 0.85
```

#### 场景 2 (环岛)

推荐配置:

```python
train_timesteps = 300000
n_cpu = 6
batch_size = 64
learning_rate = 3e-4    # 相对保守
gamma = 0.9
```

#### 场景 3 (停车 - 最难)

推荐配置:

```python
train_timesteps = 500000  # 更多步数
n_cpu = 4                 # 降低并行以增加样本多样性
batch_size = 32           # 更小批次
learning_rate = 2e-4      # 更低学习率
gamma = 0.95
policy_kwargs=dict(net_arch=dict(pi=[512, 512], vf=[512, 512]))  # 更大网络
```

## 修改建议

如需调整超参数，请按以下步骤操作：

### 步骤 1: 编辑配置文件

**train_highway_ppo.py** (完整参数对照):

```python
# 第 73-74 行：基础配置
n_cpu = 6
batch_size = 64

# 第 61 行：环境配置
"duration": 40,

# 第 121-127 行：模型和训练配置
policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
n_steps=batch_size * 12 // n_cpu,
n_epochs=10,
learning_rate=5e-4,
gamma=0.8,

# 第 138 行：总训练步数
train_timesteps = 300000
```

**test_highway_ppo.py** (完整参数对照):

```python
# 第 54 行：环境持续时间
"duration": 60,

# 第 138 行：测试轮数
episodes = 5
```

### 步骤 2: 保存并运行

```bash
# 编辑完成后直接运行
python train_highway_ppo.py --scenario 1
```

### 步骤 3: 监控效果

- **进度**: 在终端看实时输出
- **学习曲线**: 使用 TensorBoard
  ```bash
  tensorboard --logdir=highway_ppo/highway_merge/run_1
  ```

### 参数调整建议流程

1. **快速测试**: 设置 `train_timesteps = 50000`, `n_cpu = 2`
2. **初步优化**: 根据测试结果调整 `learning_rate` 和 `n_epochs`
3. **微调收敛**: 调整 `gamma` 和网络架构
4. **最终训练**: 设置 `train_timesteps = 500000+`, 使用最优超参数

---

## 常见问题

**Q: 训练失败显示"找不到模型"**
A: 确保已完成训练。路径应为 `MY_TEST/highway_ppo/scenario_name/run_N/`

**Q: 如何只训练一个特定场景？**
A: 使用 `--scenario` 参数指定，其他场景的结果不会被清除。

**Q: 测试时总是超时（Timeout）**
A: 模型未收敛，需要增加训练步数或调整超参数。

**Q: 停车场景一直失败**
A: 停车是最难的任务。确保训练步数足够（建议 > 500K），或使用"场景 3 优化建议"的配置。

**Q: 修改了参数后效果变差**
A: 恢复默认值重新尝试。参数调整需要多次实验。建议一次只改一个参数。

**Q: 如何快速测试新想法？**
A: 用小的 `train_timesteps` (10K-50K) 和 `n_cpu=2` 进行快速实验，验证可行性后再用大参数训练。

## 联系与反馈

如有问题，请检查：

1. 依赖库版本是否正确
2. 训练结果文件是否已生成
3. 命令参数是否正确

---

更新于: 2026-01-18
Highway-Env 多场景 PPO 训练框架 v1.0
