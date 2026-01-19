# 平稳驾驶优化改进 - Smooth Driving Improvements

## 概述 (Overview)

本文档详细描述了对MY_TEST中的训练和测试脚本所做的优化，旨在使智能体学习更加**平稳的驾驶行为**。

---

## 主要改进 (Key Improvements)

### 1. 优化奖励函数 (Optimized Reward Function)

#### 文件: `train_highway_ppo.py` 和 `test_highway_ppo.py` 中的 `CustomRewardWrapper` 类

**改进前的问题:**
- 奖励FASTER动作 +0.5，过于激励急加速
- 行驶距离奖励 = speed/15.0，没有考虑速度稳定性
- 缺乏对动作平稳性的激励

**改进后的策略:**

```python
# (a) 碰撞惩罚增强
if info.get('crashed', False):
    reward -= 10.0  # 从 -5.0 增加到 -10.0，更强的避碰动机

# (b) 鼓励平稳驾驶而非急速
if action == 1:  # IDLE (保持速度)
    reward += 0.3  # 新增奖励，鼓励平稳维持速度
elif action == 3:  # FASTER
    reward += 0.1  # 从 0.5 降低到 0.1，减少急加速
elif action == 4:  # SLOWER
    reward += 0.1  # 从 -0.05 改为 +0.1，平稳减速也是好的

# (c) 惩罚急剧的动作变化
if self.last_action is not None:
    if (self.last_action == 3 and action == 4) or \
       (self.last_action == 4 and action == 3):
        reward -= 0.2  # 新增：避免加速->减速的急剧转换

# (d) 理想速度奖励 (新增)
ideal_speed = 20.0  # m/s
speed_diff = abs(speed - ideal_speed)
speed_reward = max(0, (ideal_speed - speed_diff) / ideal_speed) * 0.5
reward += speed_reward  # 鼓励保持接近20 m/s的平稳速度
```

**效果:**
- ✅ 减少频繁的加速/减速切换
- ✅ 鼓励IDLE操作，维持稳定速度
- ✅ 促进速度在合理范围内变化

---

### 2. PPO超参数优化 (Optimized PPO Hyperparameters)

#### 文件: `train_highway_ppo.py`

| 参数 | 改进前 | 改进后 | 说明 |
|------|-------|-------|------|
| `n_steps` | 128 | 170 | 增加每次更新的经验量，更稳定 |
| `n_epochs` | 10 | 20 | 更充分地学习每批数据 |
| `learning_rate` | 5e-4 | 3e-4 | 降低学习率，更平稳的策略更新 |
| `gamma` | 0.8 | 0.95 | 增加折扣因子，重视长期回报 |
| `gae_lambda` | 默认 | 0.9 | 新增：控制GAE平衡，提高学习稳定性 |
| `ent_coef` | 默认 | 0.01 | 新增：适度的熵系数，鼓励探索 |
| `vf_coef` | 默认 | 0.5 | 新增：价值函数的权重 |

**改进原理:**
- **n_steps增加**: 利用更多样化的经验，减少噪声影响
- **n_epochs增加**: 从相同的批数据中学习更充分
- **learning_rate降低**: 避免过大的参数跳跃，策略更新更平稳
- **gamma增加**: 0.95 vs 0.8，更重视未来回报，鼓励长期最优决策
- **gae_lambda**: 0.9是标准值，平衡偏差和方差

**效果:**
- ✅ 更稳定的策略收敛
- ✅ 更好的长期规划能力
- ✅ 减少学习过程中的波动

---

### 3. 动作平稳化机制 (Action Smoothing Mechanism)

#### 位置: `CustomRewardWrapper.__init__()` 和 `step()` 方法

```python
def __init__(self, env):
    super().__init__(env)
    self.last_action = None  # 记录前一步动作
    
def step(self, action):
    # ... 计算奖励 ...
    
    # 检测并惩罚快速的动作切换
    if self.last_action is not None:
        if (self.last_action == 3 and action == 4) or \  # 加速->减速
           (self.last_action == 4 and action == 3):      # 减速->加速
            reward -= 0.2  # 惩罚
    
    self.last_action = action  # 更新历史
    return obs, reward, done, truncated, info
```

**效果:**
- ✅ 防止"加速->减速"的震荡模式
- ✅ 创建平稳的速度曲线
- ✅ 模拟真实驾驶的惯性感

---

## 预期效果 (Expected Results)

使用这些改进后，可以预期：

1. **驾驶平稳性 ↑**
   - 更少的急加速/减速
   - 速度变化更渐进
   - 乘坐舒适度提高

2. **安全性 ↑**
   - 更少的碰撞风险
   - 更好的距离保持
   - 更可预测的行为

3. **训练稳定性 ↑**
   - 更平稳的学习曲线
   - 更少的奖励波动
   - 更可靠的模型收敛

4. **泛化能力 ↑**
   - 学习到的策略更通用
   - 在未见场景中表现更好

---

## 使用方法 (Usage)

无需修改使用方式，直接运行：

```bash
# 训练
python train_highway_ppo.py --scenario 1

# 测试
python test_highway_ppo.py --scenario 1
```

改进会自动应用。

---

## 进一步优化建议 (Further Optimization Tips)

如果仍想进一步改进，可以尝试：

1. **增加训练时间**: 将 `train_timesteps` 从 100000 增加到 200000-500000
2. **网络结构调整**: 将 `net_arch` 改为 `[512, 512]` 以增加表达能力
3. **环境配置**: 增加 `duration` 参数给更长的episode来学习长期规划
4. **动作缓冲**: 实现真正的动作平滑化（过去N步的滑动平均）
5. **速度范围微调**: 调整 `ideal_speed` 参数（当前20.0 m/s）根据具体场景

---

## 文件修改清单 (Modified Files)

- ✅ `/home/tsui/src/HighwayEnv/MY_TEST/train_highway_ppo.py`
  - CustomRewardWrapper (第 12-57 行)
  - PPO超参数定义 (第 86-101 行)
  - 模型创建调用 (第 130-148 行)

- ✅ `/home/tsui/src/HighwayEnv/MY_TEST/test_highway_ppo.py`
  - CustomRewardWrapper (第 14-57 行)

---

## 版本信息 (Version Info)

- 改进日期: 2026-01-19
- 优化版本: v2.0 (Smooth Driving)
- 兼容性: Stable Baselines3, gymnasium, highway-env
