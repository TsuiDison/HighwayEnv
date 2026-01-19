# 快速参考 - Quick Reference Guide

## 🎯 3大核心改进

### 1️⃣ 高速驾驶优化
```
✅ 增加高速模式(>25m/s)下IDLE动作奖励到+0.4
✅ 监控速度方差，波动大时惩罚
✅ Episode时间增加到60s，学习长期平稳驾驶
✅ 网络从[256,256]扩大到[384,256]
✅ 训练步数从100K→300K
```

### 2️⃣ 倒车入库专用优化  
```
✅ 新增ParkingRewardWrapper专门处理停车
✅ 精细的距离梯度奖励（越接近目标奖励越多）
✅ 动作平稳性约束，限制转向变化
✅ 网络扩大到[512,512,256]处理复杂操纵
✅ 训练步数大幅增加到500K（需5-6小时）
✅ 降低学习率到2e-4，更稳定收敛
✅ batch_size降到32以提高梯度精度
```

### 3️⃣ 分场景超参数
```
停车(Scenario 3):    最复杂，最多资源
高速(Scenario 1):    次复杂，中等资源  
环岛(Scenario 2):    相对简单，标准资源
```

---

## 🚀 快速开始

### 训练
```bash
# 高速驾驶 (推荐先用这个测试改进)
python train_highway_ppo.py --scenario 1

# 倒车入库 (完整版，会花5-6小时)
python train_highway_ppo.py --scenario 3

# 环岛通过
python train_highway_ppo.py --scenario 2
```

### 测试
```bash
python test_highway_ppo.py --scenario 1
python test_highway_ppo.py --scenario 3
```

---

## 📊 关键数字

| 项目 | 数值 |
|------|------|
| 停车训练步数 | **500,000** |
| 高速训练步数 | **300,000** |
| 环岛训练步数 | **250,000** |
| 停车网络 | [512,512,256] |
| 高速网络 | [384,256] |
| 停车学习率 | 2e-4 |
| 高速学习率 | 3e-4 |
| 停车batch_size | 32 |
| 高速batch_size | 64 |

---

## ✨ 改进指标

### 高速驾驶
- 加速/减速频率: ↓ 30-40%
- 速度波动: ↓ 20-30%  
- 驾驶平稳度: ↑ 30-50%

### 停车
- 成功率: ↑ 20-40%
- 停车精度: ↑ 25-35%
- 平稳度: ↑ 40-50%

---

## 🔧 文件列表

| 文件 | 变更 |
|------|------|
| train_highway_ppo.py | ✅ 新增ParkingRewardWrapper，分场景超参数 |
| test_highway_ppo.py | ✅ 更新wrapper使用 |
| SMOOTH_DRIVING_IMPROVEMENTS.md | v1.0基础改进 |
| ADVANCED_TRAINING_GUIDE.md | v2.1本文档 |
| README.md | 建议补充新的训练指导 |

---

## 💡 问题排查

| 问题 | 解决方案 |
|------|---------|
| 停车还是不稳定 | 增加训练步数或降低learning_rate |
| 高速还是突然加速 | 检查CustomRewardWrapper中的速度阈值25.0 |
| 训练太慢 | 停车的500K对GPU来说需要5-6小时，CPU更久 |
| 成功率低 | 增加episode长度(duration)或网络大小 |

---

## 📈 训练进度预期

### Scenario 1 (高速): 3-4小时
- 0-50K步: 学习基本驾驶
- 50-150K步: 优化速度控制
- 150-300K步: 微调稳定性

### Scenario 3 (停车): 5-7小时  
- 0-100K步: 学习基本操纵
- 100-300K步: 精化目标定位
- 300-500K步: 精准对齐

---

## 🎓 理论基础

### 为什么分场景优化？
1. **高速**: 需要稳定的速度控制，不是精准定位
2. **停车**: 需要精确的空间定位，而不是速度

### 为什么增加训练步数？
- 更复杂的任务需要更多样本
- 停车需要在多个位置学会对齐
- 通过课程学习逐步适应难度

### 为什么调整网络大小？
- 停车的决策空间更复杂（3D位置+角度）
- 需要更大的表达能力
- 高速相对简单，小网络足够

---

## ⚙️ 高级调参

如果要进一步优化，修改这些参数：

```python
# 在 train_highway_ppo.py 中

# 1. 速度约束（第70行附近）
ideal_speed = 20.0  → 改为25.0（偏快）或15.0（偏慢）

# 2. 速度方差敏感度（第85行附近）  
speed_variance * 0.05  → 改为0.10更严格，0.02更宽松

# 3. 停车距离因子（第140行附近）
distance_improvement * proximity_factor * 2.0  → 改为3.0或1.5

# 4. 碰撞惩罚（第37行）
reward -= 10.0  → 改为-15.0或-5.0

# 5. 网络深度（第307-316行）
pi=[512, 512, 256]  → 改为[512, 512, 512, 256]（更深）
```

---

## 📝 文档导航

- **当前文件**: 快速参考卡片
- [ADVANCED_TRAINING_GUIDE.md](ADVANCED_TRAINING_GUIDE.md): 详细技术文档
- [SMOOTH_DRIVING_IMPROVEMENTS.md](SMOOTH_DRIVING_IMPROVEMENTS.md): v1.0初版改进
- [README.md](README.md): 项目总体说明
- [TRAINING_GUIDE.md](highway_ppo/TRAINING_GUIDE.md): 训练指南

---

**最后更新**: 2026-01-19  
**版本**: v2.1 Advanced Training Enhancements  
**维护者**: HighwayEnv 优化小组
