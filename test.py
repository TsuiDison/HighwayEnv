import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

import highway_env  # noqa: F401

if __name__ == '__main__':
    # 直接加载已训练的 PPO 模型（不继续训练）
    model = PPO.load("highway_ppo/model")

    # 创建测试环境
    test_env = gym.make("highway-fast-v0")

    print("Testing the trained PPO agent (no additional training)...")

    for episode in range(5):  # 测试更多 episodes
        obs, info = test_env.reset()
        done = truncated = False
        total_reward = 0
        total_distance = 0  # 跟踪总距离
        step = 0

        while not (done or truncated):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = test_env.step(action)
            total_reward += reward
            # 距离 ≈ 前向速度 * 时间步 (假设每步 0.1s)
            forward_speed = test_env.unwrapped.vehicle.speed * np.cos(test_env.unwrapped.vehicle.heading)
            total_distance += forward_speed * 0.1
            step += 1

        print(f"Episode {episode + 1}: Steps {step}, Total Reward {total_reward:.2f}, Distance {total_distance:.2f}")

    test_env.close()

    # 如果想继续训练，取消下面的注释：
    # print("Continuing training...")
    # n_cpu = 6
    # env = make_vec_env("highway-fast-v0", n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
    # model = PPO.load("highway_ppo/model", env=env)
    # model.learn(total_timesteps=5000, reset_num_timesteps=False)
    # model.save("highway_ppo/model_v2")