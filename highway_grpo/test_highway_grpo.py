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

        if info.get("crashed", False):
            reward -= 10.0

        speed = info.get("speed", 0)
        reward += 0.1 * (speed / 20.0)

        if action in [0, 2]:  # LANE_LEFT or LANE_RIGHT
            reward -= 0.05

        return obs, reward, done, truncated, info


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
            nn.Softmax(dim=-1),
        )

    def forward(self, x):
        return self.net(x)


def run_episode(env, policy, device, deterministic=False):
    obs, _ = env.reset()
    done = truncated = False
    total_reward = 0.0
    steps = 0
    speeds = []
    distance = 0.0
    dt = 1 / 5
    crashed = False
    while not (done or truncated):
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        with torch.no_grad():
            probs = policy(obs_tensor)
        dist = Categorical(probs)
        action = dist.probs.argmax(dim=-1) if deterministic else dist.sample()
        obs, reward, done, truncated, info = env.step(action.item())
        total_reward += float(reward)
        steps += 1
        speeds.append(info.get("speed", 0))
        distance += info.get("speed", 0) * dt
        if info.get("crashed", False):
            crashed = True
    avg_speed = float(np.mean(speeds)) if speeds else 0.0
    return total_reward, steps, avg_speed, distance, crashed


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
    speeds = []
    distances = []
    crashes = 0
    for idx in range(args.episodes):
        ep_return, ep_len, ep_speed, ep_dist, ep_crashed = run_episode(
            env, policy, device, args.deterministic
        )
        returns.append(ep_return)
        lengths.append(ep_len)
        speeds.append(ep_speed)
        distances.append(ep_dist)
        if ep_crashed:
            crashes += 1
        print(
            f"Episode {idx + 1:03d} | "
            f"return {ep_return:.2f} | length {ep_len} | "
            f"distance {ep_dist:.2f} | speed {ep_speed:.2f} | crashed {ep_crashed}"
        )

    print(f"Episodes: {args.episodes}")
    print(f"Mean return: {np.mean(returns):.2f} | Std: {np.std(returns):.2f}")
    print(f"Mean length: {np.mean(lengths):.2f} | Std: {np.std(lengths):.2f}")
    print(f"Mean speed: {np.mean(speeds):.2f} | Std: {np.std(speeds):.2f}")
    print(f"Mean distance: {np.mean(distances):.2f} | Std: {np.std(distances):.2f}")
    print(f"Crash rate: {crashes / max(1, args.episodes):.2f}")


if __name__ == "__main__":
    main()
