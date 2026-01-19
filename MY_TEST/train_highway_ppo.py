import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import highway_env  # noqa: F401
import os
import glob
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from scipy.ndimage import uniform_filter1d

# ==================================
#        Custom Wrapper
# ==================================
class CustomRewardWrapper(gym.Wrapper):
    """改进的离散动作环境奖励包装器 (Highway, Merge, Roundabout)"""
    def __init__(self, env):
        super().__init__(env)
        self.last_action = None
        self.last_speed = 0.0
        self.speed_history = []
        self.max_history = 5
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Only apply custom reward modifications for discrete actions environments
        # (Highway, Merge, Roundabout)
        # 0: LANE_LEFT, 1: IDLE, 2: LANE_RIGHT, 3: FASTER, 4: SLOWER
        
        # 1. Collision Penalty - Strong penalty to discourage crashes
        if info.get('crashed', False):
            reward -= 10.0
        
        # 2. Get current speed for analysis
        try:
            speed = self.env.unwrapped.vehicle.speed
            self.speed_history.append(speed)
            if len(self.speed_history) > self.max_history:
                self.speed_history.pop(0)
            speed_variance = np.var(self.speed_history) if len(self.speed_history) > 1 else 0
        except AttributeError:
            speed = 0.0
            speed_variance = 0.0
        
        # 3. Smooth Action Encouragement with speed constraints
        if isinstance(action, (int, np.integer)):
            # 对于高速驾驶: 鼓励IDLE，更强烈地惩罚极端加速
            if speed > 25.0:  # 高速模式 (> 25 m/s)
                if action == 1:  # IDLE
                    reward += 0.4  # 高速下保持稳定是最好的
                elif action == 3:  # FASTER
                    reward += 0.05  # 更小的加速奖励
                    # 如果已经很快，更强烈地惩罚继续加速
                    if speed > 28.0:
                        reward -= 0.3
                elif action == 4:  # SLOWER
                    reward += 0.15  # 鼓励平缓减速
            else:  # 正常速度 (15-25 m/s)
                if action == 1:  # IDLE
                    reward += 0.3
                elif action == 3:  # FASTER
                    reward += 0.1
                elif action == 4:  # SLOWER
                    reward += 0.1
            
            # Penalize abrupt action changes
            if self.last_action is not None:
                if (self.last_action == 3 and action == 4) or \
                   (self.last_action == 4 and action == 3):
                    reward -= 0.2  # 急剧加速->减速的转变
        
        # 4. Speed variance penalty - 鼓励平稳的速度曲线
        if speed_variance > 0.5:
            reward -= speed_variance * 0.05
        
        # 5. Ideal speed reward with wider range for highway
        ideal_speeds = {
            "highway": (25.0, 30.0),  # 高速公路的理想速度范围
            "normal": (15.0, 25.0),
        }
        
        ideal_low, ideal_high = ideal_speeds["highway"] if speed > 22 else ideal_speeds["normal"]
        
        if ideal_low <= speed <= ideal_high:
            reward += 0.2
        else:
            speed_diff = min(abs(speed - ideal_low), abs(speed - ideal_high))
            reward -= speed_diff * 0.02
        
        self.last_action = action
        self.last_speed = speed
        return obs, reward, done, truncated, info


class ParkingRewardWrapper(gym.Wrapper):
    """针对停车场景的连续动作环境奖励优化"""
    def __init__(self, env):
        super().__init__(env)
        self.prev_distance_to_goal = None
        self.episode_step = 0
        self.max_episode_steps = 200
    
    def reset(self, **kwargs):
        self.prev_distance_to_goal = None
        self.episode_step = 0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.episode_step += 1
        
        # 1. 基础碰撞惩罚
        if info.get('crashed', False):
            reward -= 10.0
        
        # 2. 目标距离的更好奖励
        try:
            # obs 通常是 [achieved_goal, desired_goal, observation]
            # 对于停车: achieved_goal是当前位置, desired_goal是停车位置
            if isinstance(obs, dict):
                achieved = obs.get('achieved_goal', np.array([0, 0, 0, 0, 0, 0]))
                desired = obs.get('desired_goal', np.array([0, 0, 0, 0, 0, 0]))
            else:
                achieved = obs[:6] if len(obs) >= 6 else np.array([0, 0, 0, 0, 0, 0])
                desired = obs[6:12] if len(obs) >= 12 else np.array([0, 0, 0, 0, 0, 0])
            
            # 计算距离 (位置和方向的组合距离)
            pos_diff = np.sqrt((achieved[0] - desired[0])**2 + (achieved[1] - desired[1])**2)
            angle_diff = abs(achieved[4] - desired[4]) + abs(achieved[5] - desired[5])
            distance = pos_diff + angle_diff * 0.5
            
            # 奖励逐步接近目标
            if self.prev_distance_to_goal is not None:
                distance_improvement = self.prev_distance_to_goal - distance
                if distance_improvement > 0:
                    # 接近目标时奖励更多
                    proximity_factor = max(0.5, 1.0 - distance / 10.0)
                    reward += distance_improvement * proximity_factor * 2.0
                else:
                    # 远离目标时轻微惩罚
                    reward -= abs(distance_improvement) * 0.5
            
            self.prev_distance_to_goal = distance
            
            # 3. 时间效率奖励 (完成得快更好，但不能太着急)
            if self.episode_step > 1:
                steps_taken = self.episode_step
                # 奖励快速但平稳的完成
                if steps_taken < 100:
                    reward += 0.1  # 快速完成奖励
                    
        except (IndexError, ValueError):
            pass
        
        # 4. 成功停车的大奖励
        if info.get('is_success', False):
            reward += 10.0  # 成功停车大奖励
        
        # 5. 平稳动作约束 - 连续动作空间中限制方向变化
        if hasattr(self, 'last_action') and self.last_action is not None:
            action_diff = np.sqrt(np.sum((action - self.last_action)**2))
            if action_diff > 0.5:  # 动作变化过大
                reward -= action_diff * 0.2
        
        self.last_action = action.copy() if isinstance(action, np.ndarray) else action
        
        return obs, reward, done, truncated, info

# ==================================
#    绘制训练曲线函数
# ==================================
def plot_training_curves(run_dir, scenario_name, script_dir):
    """从TensorBoard日志绘制训练曲线"""
    try:
        # 创建输出目录
        plot_dir = script_dir
        os.makedirs(plot_dir, exist_ok=True)
        
        # 使用EventAccumulator读取TensorBoard日志
        event_acc = EventAccumulator(run_dir)
        event_acc.Reload()
        
        print(f"\n📊 Reading TensorBoard logs from: {run_dir}")
        
        # 获取所有标签
        tags = event_acc.Tags()
        print(f"Available tags: {tags}")
        
        # 获取标量数据
        if 'scalars' not in tags or not tags['scalars']:
            print(f"⚠️  Warning: No scalar data found in {run_dir}")
            return
            
        scalars = tags['scalars']
        print(f"Scalar keys found: {scalars}")
        
        # 提取所有数据
        data_dict = {}
        for scalar_name in scalars:
            try:
                events = event_acc.Scalars(scalar_name)
                if events:
                    steps = np.array([e.step for e in events])
                    values = np.array([e.value for e in events])
                    data_dict[scalar_name] = (steps, values)
                    print(f"  ✓ {scalar_name}: {len(events)} data points")
            except Exception as e:
                print(f"  ✗ Failed to read {scalar_name}: {e}")
        
        if not data_dict:
            print(f"⚠️  Warning: No scalar data could be extracted from {run_dir}")
            return
        
        # 创建子图
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(f'Training Progress - {scenario_name.upper()}', fontsize=16, fontweight='bold')
        
        plot_count = 0
        
        # 1. 策略梯度损失 (Policy Gradient Loss)
        policy_loss_keys = [k for k in data_dict.keys() if 'policy_gradient_loss' in k.lower()]
        if policy_loss_keys:
            key = policy_loss_keys[0]
            steps, values = data_dict[key]
            if len(steps) > 0:
                axes[0, 0].plot(steps, values, linewidth=2, color='#FF6B6B')
                axes[0, 0].set_title('Policy Gradient Loss', fontweight='bold')
                axes[0, 0].set_xlabel('Training Steps')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].grid(True, alpha=0.3)
                plot_count += 1
                print(f"  📈 Plotted: Policy Gradient Loss")
        else:
            axes[0, 0].text(0.5, 0.5, 'No Policy Loss Data', ha='center', va='center', transform=axes[0, 0].transAxes)
        
        # 2. 价值函数损失 (Value Loss)
        value_loss_keys = [k for k in data_dict.keys() if 'train/value_loss' in k]
        if value_loss_keys:
            key = value_loss_keys[0]
            steps, values = data_dict[key]
            if len(steps) > 0:
                axes[0, 1].plot(steps, values, linewidth=2, color='#4ECDC4')
                axes[0, 1].set_title('Value Loss', fontweight='bold')
                axes[0, 1].set_xlabel('Training Steps')
                axes[0, 1].set_ylabel('Loss')
                axes[0, 1].grid(True, alpha=0.3)
                plot_count += 1
                print(f"  📈 Plotted: Value Loss")
        else:
            axes[0, 1].text(0.5, 0.5, 'No Value Loss Data', ha='center', va='center', transform=axes[0, 1].transAxes)
        
        # 3. 总损失 (Total Loss)
        loss_keys = [k for k in data_dict.keys() if k == 'train/loss']
        entropy_keys = [k for k in data_dict.keys() if 'train/entropy_loss' in k]
        
        if loss_keys:
            key = loss_keys[0]
            steps, values = data_dict[key]
            if len(steps) > 0:
                axes[1, 0].plot(steps, values, linewidth=2, color='#95E1D3')
                axes[1, 0].set_title('Total Loss', fontweight='bold')
                axes[1, 0].set_xlabel('Training Steps')
                axes[1, 0].set_ylabel('Loss')
                axes[1, 0].grid(True, alpha=0.3)
                plot_count += 1
                print(f"  📈 Plotted: Total Loss")
        elif entropy_keys:
            key = entropy_keys[0]
            steps, values = data_dict[key]
            if len(steps) > 0:
                axes[1, 0].plot(steps, values, linewidth=2, color='#95E1D3')
                axes[1, 0].set_title('Entropy Loss', fontweight='bold')
                axes[1, 0].set_xlabel('Training Steps')
                axes[1, 0].set_ylabel('Loss')
                axes[1, 0].grid(True, alpha=0.3)
                plot_count += 1
                print(f"  📈 Plotted: Entropy Loss")
        else:
            axes[1, 0].text(0.5, 0.5, 'No Loss Data', ha='center', va='center', transform=axes[1, 0].transAxes)
        
        # 4. 平均奖励 (Mean Episode Reward)
        reward_keys = [k for k in data_dict.keys() if k == 'rollout/ep_rew_mean']
        if reward_keys:
            key = reward_keys[0]
            steps, values = data_dict[key]
            if len(steps) > 0:
                axes[1, 1].plot(steps, values, linewidth=2, color='#F7DC6F')
                axes[1, 1].fill_between(steps, values, alpha=0.3, color='#F7DC6F')
                axes[1, 1].set_title('Mean Episode Reward', fontweight='bold')
                axes[1, 1].set_xlabel('Training Steps')
                axes[1, 1].set_ylabel('Reward')
                axes[1, 1].grid(True, alpha=0.3)
                plot_count += 1
                print(f"  📈 Plotted: Mean Episode Reward")
        else:
            axes[1, 1].text(0.5, 0.5, 'No Reward Data', ha='center', va='center', transform=axes[1, 1].transAxes)
        
        plt.tight_layout()
        
        # 保存图表到当前目录
        plot_path = os.path.join(plot_dir, f'training_curves_{scenario_name}_run_{os.path.basename(run_dir)}.png')
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Training curves saved to: {plot_path}")
        
        plt.close()
        
        # 绘制详细的奖励曲线
        reward_keys = [k for k in data_dict.keys() if k == 'rollout/ep_rew_mean']
        if reward_keys:
            fig, ax = plt.subplots(figsize=(12, 6))
            key = reward_keys[0]
            steps, values = data_dict[key]
            
            if len(steps) > 0:
                # 平滑曲线
                if len(values) > 20:
                    smoothed = uniform_filter1d(values, size=20)
                elif len(values) > 5:
                    smoothed = uniform_filter1d(values, size=5)
                else:
                    smoothed = values
                
                ax.plot(steps, values, alpha=0.3, color='#4ECDC4', label='Raw Reward')
                ax.plot(steps, smoothed, linewidth=2.5, color='#FF6B6B', label='Smoothed Reward')
                ax.fill_between(steps, values, alpha=0.1, color='#4ECDC4')
                ax.set_title(f'Episode Reward Over Training - {scenario_name.upper()}', fontsize=14, fontweight='bold')
                ax.set_xlabel('Training Steps', fontsize=12)
                ax.set_ylabel('Mean Episode Reward', fontsize=12)
                ax.legend(fontsize=11)
                ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                reward_plot_path = os.path.join(plot_dir, f'reward_curve_{scenario_name}_run_{os.path.basename(run_dir)}.png')
                plt.savefig(reward_plot_path, dpi=300, bbox_inches='tight')
                print(f"✅ Reward curve saved to: {reward_plot_path}")
                plt.close()
        
        print(f"📊 Total plots generated: {plot_count + 1 if reward_keys else plot_count}")
        
    except Exception as e:
        print(f"❌ Error plotting training curves: {e}")
        import traceback
        traceback.print_exc()
        print("Make sure tensorboard is installed: pip install tensorboard")

# ==================================
#        Main script
# ==================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Highway Agents")
    parser.add_argument("--scenario", type=int, default=1, choices=[1, 2, 3], 
                        help="1: Highway+Merge (Mixed), 2: Roundabout, 3: Parking")
    args = parser.parse_args()

    # Define the custom configuration - 根据场景调整
    if args.scenario == 1:
        # 高速驾驶场景 - 更长的episode来学习平稳驾驶
        common_config = {
            "action": {
                "type": "DiscreteMetaAction",
                "longitudinal": True,
                "lateral": True,
            },
            "duration": 60,  # 增加到60s，更长的连续驾驶
            "collision_reward": -5.0,
            "vehicles_count": 50,
        }
    elif args.scenario == 2:
        # 环岛场景
        common_config = {
            "action": {
                "type": "DiscreteMetaAction",
                "longitudinal": True,
                "lateral": True,
            },
            "duration": 50,
            "collision_reward": -5.0,
        }
    else:
        # 停车场景 - 将在make_env中单独处理
        common_config = {}

    n_cpu = 6
    batch_size = 64
    policy_type = "MlpPolicy"
    
    # 针对不同场景的超参数优化
    if args.scenario == 3:  # 停车场景需要更精细的控制
        batch_size = 32
        n_cpu = 4
    
    # 优化的PPO超参数，促进平稳驾驶和精确控制
    ppo_params = {
        "n_steps": batch_size * 20 // n_cpu,  # 增加步长，提高稳定性
        "batch_size": batch_size,
        "n_epochs": 25 if args.scenario == 3 else 20,  # 停车需要更多epoch
        "learning_rate": 2e-4 if args.scenario == 3 else 3e-4,  # 停车用更低的学习率
        "gamma": 0.99 if args.scenario == 3 else 0.95,  # 停车重视长期规划
        "gae_lambda": 0.95 if args.scenario == 3 else 0.9,  # 停车需要更好的优势估计
        "clip_range": 0.2,
        "ent_coef": 0.005 if args.scenario == 3 else 0.01,  # 停车减少随机性
        "vf_coef": 0.5,
    }
    
    # Save directory setup
    scenario_names = {1: "highway_merge", 2: "roundabout", 3: "parking"}
    scenario_name = scenario_names[args.scenario]
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "highway_ppo", scenario_name)
    
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)

    # Find the next run ID
    existing_runs = glob.glob(os.path.join(base_dir, "run_*"))
    run_ids = []
    for run_path in existing_runs:
        try:
            run_name = os.path.basename(run_path)
            run_ids.append(int(run_name.split("_")[-1]))
        except ValueError:
            continue
    
    next_run_id = max(run_ids) + 1 if run_ids else 1
    run_dir = os.path.join(base_dir, f"run_{next_run_id}")
    os.makedirs(run_dir, exist_ok=True)
    print(f"Scenario: {scenario_name} (Mode {args.scenario})")
    print(f"Output directory for this run: {run_dir}")

    # Create wrapped environment for training
    def make_env():
        if args.scenario == 1:
            # Mix Highway and Merge
            # SubprocVecEnv will call this N times, we can randomize here
            env_id = np.random.choice(["highway-fast-v0", "merge-v0"])
            env = gym.make(env_id, render_mode="rgb_array", config=common_config)
            env = CustomRewardWrapper(env)
        elif args.scenario == 2:
            # Roundabout
            env = gym.make("roundabout-v0", render_mode="rgb_array", config=common_config)
            env = CustomRewardWrapper(env)
        elif args.scenario == 3:
            # Parking - 使用专门的停车奖励包装器
            env = gym.make("parking-v0", render_mode="rgb_array")
            env = ParkingRewardWrapper(env)
        return env

    if args.scenario == 3:
        policy_type = "MultiInputPolicy"

    # Use SubprocVecEnv for parallel training
    env = make_vec_env(make_env, n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
    
    # 根据场景调整网络大小
    if args.scenario == 3:  # 停车需要更大的网络
        net_arch = dict(pi=[512, 512, 256], vf=[512, 512, 256])
        train_timesteps = 500000  # 停车需要更多训练步数
    elif args.scenario == 1:  # 高速驾驶
        net_arch = dict(pi=[384, 256], vf=[384, 256])
        train_timesteps = 300000  # 增加到300K
    else:  # 环岛
        net_arch = dict(pi=[256, 256], vf=[256, 256])
        train_timesteps = 50000
    
    model = PPO(
        policy_type,
        env,
        policy_kwargs=dict(net_arch=net_arch),
        n_steps=ppo_params["n_steps"],
        batch_size=ppo_params["batch_size"],
        n_epochs=ppo_params["n_epochs"],
        learning_rate=ppo_params["learning_rate"],
        gamma=ppo_params["gamma"],
        gae_lambda=ppo_params["gae_lambda"],
        clip_range=ppo_params["clip_range"],
        ent_coef=ppo_params["ent_coef"],
        vf_coef=ppo_params["vf_coef"],
        verbose=2,
        tensorboard_log=run_dir,
    )
    
    print(f"Starting training for {train_timesteps} steps...")
    print(f"Configuration: Scenario {args.scenario}, Network {net_arch}")
    
    # Train the agent
    model.learn(total_timesteps=train_timesteps)
    
    # Save the agent (SB3 zip format)
    save_path_zip = os.path.join(run_dir, "model")
    model.save(save_path_zip)
    
    # Save the policy (PT format)
    save_path_pt = os.path.join(run_dir, "model.pt")
    torch.save(model.policy.state_dict(), save_path_pt)
    
    print(f"Training finished.")
    print(f"Model saved to {save_path_zip}.zip")
    print(f"Policy saved to {save_path_pt}")
    
    # ==================================
    # 绘制训练曲线
    # ==================================
    print("\nGenerating training curves...")
    plot_training_curves(run_dir, scenario_name, script_dir)

