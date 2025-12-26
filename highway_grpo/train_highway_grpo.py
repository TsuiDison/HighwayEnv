import os

# Limit BLAS threads to avoid forkserver/OpenBLAS errors in constrained envs.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")

import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv
import highway_env  # noqa: F401
from torch.utils.tensorboard import SummaryWriter

# ==================================
#        Custom Wrapper (优化奖励)
# ==================================
class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        
        # 1. 强力碰撞惩罚
        if info.get('crashed', False):
             reward -= 10.0 
        
        # 2. 鼓励高速行驶 (这是达到600米的关键)
        speed = info.get('speed', 0)
        # 基础分：只要活着就有分，速度越快分越高
        reward += 0.1 * (speed / 20.0) 
        
        # 3. 动作平滑惩罚（防止疯狂变道导致的失控）
        if action in [0, 2]: # LANE_LEFT or LANE_RIGHT
            reward -= 0.05

        return obs, reward, done, truncated, info

# ==================================
#       GRPO Implementation
# ==================================
class GRPOTrainer:
    def __init__(self, policy_net, env, learning_rate=3e-4, eps_clip=0.2, beta=0.01):
        self.env = env
        self.policy = policy_net
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.eps_clip = eps_clip
        self.beta = beta
        
    def select_action(self, obs):
        obs = torch.FloatTensor(obs)
        with torch.no_grad():
            probs = self.policy(obs)
        dist = Categorical(probs)
        action = dist.sample()
        return action.cpu().numpy(), dist.log_prob(action).cpu().numpy()

    def train(self, total_timesteps, rollout_steps, n_epochs):
        curr_steps = 0
        save_dir = "highway_grpo"
        os.makedirs(save_dir, exist_ok=True)
        writer = SummaryWriter(log_dir=os.path.join(save_dir, "logs_v2"))
        global_step = 0
        last_print_step = 0
        rollout_log_interval = 512
        
        while curr_steps < total_timesteps:
            obs = self.env.reset()
            all_obs, all_actions, all_log_probs, all_rewards = [], [], [], []
            
            # 记录业务指标
            episode_speeds = []
            crash_count = 0
            rollout_reward_sum = 0.0
            rollout_reward_count = 0
            dt = 1 / 5 # policy_frequency 是 5Hz，所以每步是 0.2 秒
            
            # --- Rollout Phase ---
            episode_distances = {env_id: 0.0 for env_id in range(self.env.num_envs)}
            for rollout_idx in range(rollout_steps):
                action, log_prob = self.select_action(obs)
                next_obs, reward, done, info = self.env.step(action)
                
                all_obs.append(obs)
                all_actions.append(action)
                all_log_probs.append(log_prob)
                all_rewards.append(reward)
                rollout_reward_sum += float(np.mean(reward))
                rollout_reward_count += 1
                # 统计数据
                for env_id, env_info in enumerate(info):
                    episode_speeds.append(env_info.get("speed", 0))
                    episode_distances[env_id] += env_info.get("speed", 0) * dt
                    if env_info.get("crashed", False):
                        crash_count += 1
                
                obs = next_obs
                curr_steps += self.env.num_envs
                if (rollout_idx + 1) % rollout_log_interval == 0:
                    mean_step_reward = (
                        rollout_reward_sum / rollout_reward_count
                        if rollout_reward_count > 0
                        else 0.0
                    )
                    print(
                        f"[rollout] step {rollout_idx + 1}/{rollout_steps} | "
                        f"total_steps {curr_steps}/{total_timesteps} | "
                        f"mean_step_reward {mean_step_reward:.2f}"
                    )
            
            # --- GRPO Logic: Group Relative Advantage ---
            # rewards shape: [rollout_steps, n_envs]
            rewards_np = np.array(all_rewards)
            # 计算每个环境在这一轮内的累计奖励 (Trajectory Return)
            group_returns = np.sum(rewards_np, axis=0) # shape: [n_envs]
            
            # 计算群体的均值和标准差
            mean_ret = np.mean(group_returns)
            std_ret = np.std(group_returns) + 1e-8
            
            # 计算相对优势：表现好的轨迹优势为正，表现差的为负
            norm_advantages = (group_returns - mean_ret) / std_ret
            # 将优势广播到整个轨迹的每一步
            advantages_tensor = torch.FloatTensor(norm_advantages).repeat(rollout_steps, 1).view(-1)

            # 数据转换
            obs_tensor = torch.FloatTensor(np.array(all_obs)).view(-1, *all_obs[0].shape[1:])
            action_tensor = torch.LongTensor(np.array(all_actions)).view(-1)
            old_log_probs = torch.FloatTensor(np.array(all_log_probs)).view(-1)

            # --- Update Phase ---
            for _ in range(n_epochs):
                probs = self.policy(obs_tensor)
                dist = Categorical(probs)
                new_log_probs = dist.log_prob(action_tensor)
                entropy = dist.entropy().mean()
                
                ratio = torch.exp(new_log_probs - old_log_probs)
                surr1 = ratio * advantages_tensor
                surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages_tensor
                
                loss = -torch.min(surr1, surr2).mean() - self.beta * entropy
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # --- Logging ---
            writer.add_scalar("env/mean_return", mean_ret, global_step)
            writer.add_scalar("env/avg_speed", np.mean(episode_speeds), global_step)
            writer.add_scalar("env/crash_rate", crash_count / (rollout_steps * self.env.num_envs), global_step)
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/entropy", entropy.item(), global_step)
            mean_distance = float(np.mean(list(episode_distances.values())))
            writer.add_scalar("env/mean_distance", mean_distance, global_step)
            # 记录所有并行环境里跑得最远的那辆车是多少米
            writer.add_scalar("env/max_distance", float(np.max(list(episode_distances.values()))), global_step)
            
            # 记录动作分布情况
            for a in range(5):
                act_perc = (action_tensor == a).float().mean().item()
                writer.add_scalar(f"actions/act_{a}", act_perc, global_step)

            if curr_steps - last_print_step >= 1000:
                print(
                    f"Steps: {curr_steps}/{total_timesteps} | "
                    f"Ret: {mean_ret:.2f} | Speed: {np.mean(episode_speeds):.2f} | "
                    f"Dist: {mean_distance:.2f} | CrashRate: {crash_count / (rollout_steps * self.env.num_envs):.3f}"
                )
                last_print_step = curr_steps
            global_step += 1

        writer.close()

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

# 策略网络保持不变...
class PolicyNet(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_dim[0] * input_dim[1], 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, output_dim),
            nn.Softmax(dim=-1)
        )
    def forward(self, x):
        return self.net(x)

if __name__ == "__main__":
    env_config = {
        "action": {"type": "DiscreteMetaAction"},
        "duration": 300,  # 增加时长到300s，确保有物理空间跑够600米
        "observation": {"type": "Kinematics"},
        "simulation_frequency": 15,
        "policy_frequency": 5,
    }

    n_cpu = 128 
    rollout_steps = 1024 # 增加采样长度，以便计算更完整的轨迹奖励
    
    def make_env():
        env = gym.make("highway-fast-v0", render_mode="rgb_array", config=env_config)
        env = CustomRewardWrapper(env)
        return env

    env = make_vec_env(make_env, n_envs=n_cpu, vec_env_cls=DummyVecEnv)
    
    policy = PolicyNet(env.observation_space.shape, env.action_space.n)
    model = GRPOTrainer(policy, env, learning_rate=3e-4)
    
    print("Training GRPO v2 (Trajectory-based)...")
    model.train(total_timesteps=int(2e6), rollout_steps=rollout_steps, n_epochs=10)
    model.save("highway_grpo/model_grpo_v2.pth")
