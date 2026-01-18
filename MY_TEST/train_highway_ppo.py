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

# ==================================
#        Custom Wrapper
# ==================================
class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Only apply custom reward modifications for discrete actions environments
        # (Highway, Merge, Roundabout)
        # 0: LANE_LEFT, 1: IDLE, 2: LANE_RIGHT, 3: FASTER, 4: SLOWER
        
        # 1. Increase Collision Penalty
        if info.get('crashed', False):
             reward -= 5.0 
        
        # 2. Acceleration Reward / Deceleration Penalty (Discrete approximation)
        # Check if action is scalar/discrete
        if isinstance(action, (int, np.integer)):
            if action == 3: # FASTER
                reward += 0.5 
            elif action == 4: # SLOWER
                reward -= 0.05
            
        # 3. Distance Traveled Reward
        try:
            # For Parking, this might differ, but we handle logic in main
            speed = self.env.unwrapped.vehicle.speed
            distance_reward = speed / 15.0
            reward += distance_reward
        except AttributeError:
            pass

        return obs, reward, done, truncated, info

# ==================================
#        Main script
# ==================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Highway Agents")
    parser.add_argument("--scenario", type=int, default=1, choices=[1, 2, 3], 
                        help="1: Highway+Merge (Mixed), 2: Roundabout, 3: Parking")
    args = parser.parse_args()

    # Define the custom configuration
    # Note: Parking config differs, will handle below
    common_config = {
        "action": {
            "type": "DiscreteMetaAction",
            "longitudinal": True,
            "lateral": True,
        },
        "duration": 40,
        "collision_reward": -5.0,
    }

    n_cpu = 6
    batch_size = 64
    policy_type = "MlpPolicy"
    
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
            # Parking
            # Parking matches are continuous, use MultiInputPolicy
            env = gym.make("parking-v0", render_mode="rgb_array")
            # We do NOT use CustomRewardWrapper for Parking (continuous, different reward structure)
            # Default parking reward is dense enough (-distance)
        return env

    if args.scenario == 3:
        policy_type = "MultiInputPolicy"

    # Use SubprocVecEnv for parallel training
    env = make_vec_env(make_env, n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
    
    model = PPO(
        policy_type,
        env,
        policy_kwargs=dict(net_arch=dict(pi=[256, 256], vf=[256, 256])),
        n_steps=batch_size * 12 // n_cpu,
        batch_size=batch_size,
        n_epochs=10,
        learning_rate=5e-4,
        gamma=0.8,
        verbose=2,
        tensorboard_log=run_dir,
    )
    
    train_timesteps = 100000 
    print(f"Starting training for {train_timesteps} steps...")
    
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
