import argparse
import os

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

import highway_env  # noqa: F401


class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)

        # 对齐 PPO 版本的 reward shaping
        if info.get("crashed", False):
            reward -= 5.0

        if action == 3:  # FASTER
            reward += 0.5
        elif action == 4:  # SLOWER
            reward -= 0.05

        try:
            speed = self.env.unwrapped.vehicle.speed
            reward += speed / 15.0
        except AttributeError:
            pass

        return obs, reward, done, truncated, info


class PolicyNet(nn.Module):
    def __init__(self, obs_shape, act_dim):
        super().__init__()
        flat_dim = int(np.prod(obs_shape))
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
        )
        self.logits = nn.Linear(256, act_dim)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        x = self.net(obs)
        return self.logits(x)


def run_episode(env, policy, device, deterministic=False):
    obs, _ = env.reset()
    done = truncated = False
    total_reward = 0.0
    steps = 0
    distance = 0.0
    crashed = False
    while not (done or truncated):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            logits = policy(obs_tensor)
        dist = Categorical(logits=logits)
        action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
        obs, reward, done, truncated, info = env.step(action.item())
        total_reward += float(reward)
        steps += 1
        try:
            current_speed = env.unwrapped.vehicle.speed
            distance += current_speed / 15.0
        except AttributeError:
            pass
        if info.get("crashed", False):
            crashed = True
    return total_reward, steps, distance, crashed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="highway_grpo/model_grpo_v2.pth")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--video-dir", default="highway_grpo/videos")
    parser.add_argument("--video-episodes", type=int, default=1)
    args = parser.parse_args()

    env_config = {
        "action": {"type": "DiscreteMetaAction"},
        "duration": 300,
        "observation": {"type": "Kinematics"},
        "simulation_frequency": 15,
        "policy_frequency": 5,
    }

    render_mode = "rgb_array" if args.video else None
    env = gym.make("highway-fast-v0", config=env_config, render_mode=render_mode)
    env = CustomRewardWrapper(env)
    if args.video:
        os.makedirs(args.video_dir, exist_ok=True)
        env = gym.wrappers.RecordVideo(
            env,
            video_folder=args.video_dir,
            episode_trigger=lambda ep_id: ep_id < args.video_episodes,
        )

    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    policy = PolicyNet(obs_shape, n_actions).to(device)
    policy.load_state_dict(torch.load(args.model, map_location=device))
    policy.eval()

    returns = []
    lengths = []
    distances = []
    crashes = 0
    for idx in range(args.episodes):
        ep_return, ep_len, ep_dist, ep_crashed = run_episode(
            env, policy, device, args.deterministic
        )
        returns.append(ep_return)
        lengths.append(ep_len)
        distances.append(ep_dist)
        if ep_crashed:
            crashes += 1
        print(
            f"Episode {idx + 1:03d} | "
            f"return {ep_return:.2f} | length {ep_len} | "
            f"distance {ep_dist:.2f} | crashed {ep_crashed}"
        )

    print(f"Episodes: {args.episodes}")
    print(f"Mean return: {np.mean(returns):.2f} | Std: {np.std(returns):.2f}")
    print(f"Mean length: {np.mean(lengths):.2f} | Std: {np.std(lengths):.2f}")
    print(f"Mean distance: {np.mean(distances):.2f} | Std: {np.std(distances):.2f}")
    print(f"Crash rate: {crashes / max(1, args.episodes):.2f}")


if __name__ == "__main__":
    main()
