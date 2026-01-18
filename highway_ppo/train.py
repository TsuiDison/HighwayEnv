import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import highway_env  # noqa: F401
import os
import argparse
from stable_baselines3.common.callbacks import BaseCallback

# ==================================
#        Custom Wrapper (From MY_TEST)
# ==================================
class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Action is discrete: 0: LANE_LEFT, 1: IDLE, 2: LANE_RIGHT, 3: FASTER, 4: SLOWER
        
        # 1. Increase Collision Penalty (MY_TEST style)
        if info.get('crashed', False):
             reward -= 5.0 
        
        # 2. Acceleration Reward
        if action == 3: # FASTER
            reward += 0.5 
        elif action == 4: # SLOWER
            reward -= 0.05
            
        # 3. Distance Traveled Reward
        try:
            speed = self.env.unwrapped.vehicle.speed
            # Distance = speed * dt (1/15)
            distance_reward = speed / 15.0
            reward += distance_reward
        except AttributeError:
            pass

        return obs, reward, done, truncated, info

class SimpleProgressCallback(BaseCallback):
    """简单的进度回调 - 每完成10%打印一次."""
    
    def __init__(self, total_timesteps):
        super().__init__()
        self.total_timesteps = total_timesteps
        self.last_progress = 0
    
    def _on_step(self):
        current = min(self.num_timesteps, self.total_timesteps)
        progress = int(100 * current / self.total_timesteps)
        
        if progress >= self.last_progress + 10:
            print(f"  Progress: {progress}% ({current}/{self.total_timesteps} steps)")
            self.last_progress = progress - (progress % 10)
        return True

def get_next_exp_folder(base_dir="highway_ppo_results"):
    """
    Check for existing folders like exp_1, exp_2 and return the next available one.
    """
    if not os.path.exists(base_dir):
        os.makedirs(base_dir)
    
    existing_exps = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("exp_")]
    
    if not existing_exps:
        next_id = 1
    else:
        # Extract numbers, default to 0 if format is wrong
        ids = []
        for d in existing_exps:
            try:
                ids.append(int(d.split("_")[1]))
            except ValueError:
                pass
        next_id = max(ids) + 1 if ids else 1
    
    new_exp_dir = os.path.join(base_dir, f"exp_{next_id}")
    os.makedirs(new_exp_dir)
    return new_exp_dir, next_id

def train():
    # 1. Define Environment Config (From MY_TEST)
    env_config = {
        "action": {
            "type": "DiscreteMetaAction",
            "longitudinal": True,
            "lateral": True,
        },
        "duration": 40,
        "collision_reward": -5.0,
    }

    # 2. Setup paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(base_dir, "runs")
    exp_dir, exp_id = get_next_exp_folder(results_dir)
    print(f"\n[INFO] Starting Experiment {exp_id}")
    print(f"[INFO] Saving results to: {exp_dir}")

    # 3. Training Settings
    n_cpu = 6  # From MY_TEST
    batch_size = 64
    total_timesteps = int(2e4) # 20k steps per MY_TEST

    # Update for quick debug if needed (not active now)
    # total_timesteps = 2000 

    # 4. Create Environment
    # We use a helper to wrap the env
    def make_env():
        env = gym.make("highway-fast-v0", render_mode="rgb_array", config=env_config)
        env = CustomRewardWrapper(env)
        return env

    env = make_vec_env(make_env, n_envs=n_cpu, vec_env_cls=SubprocVecEnv)

    # 5. Create Model
    print(f"[INFO] Training on {n_cpu} CPUs...")
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
        n_steps=batch_size * 12 // n_cpu, # Adaptive n_steps
        batch_size=batch_size,
        n_epochs=10,
        learning_rate=5e-4,
        gamma=0.8,
        verbose=1,
        tensorboard_log=os.path.join(exp_dir, "tensorboard"), # Log inside exp folder
        device="cuda"
    )

    # 6. Train
    callback = SimpleProgressCallback(total_timesteps)
    model.learn(total_timesteps=total_timesteps, callback=callback)
    
    # 7. Save
    save_path = os.path.join(exp_dir, "model")
    model.save(save_path)
    print(f"\n[SUCCESS] Model saved to {save_path}.zip")
    print(f"To test this model, run: python highway_ppo/test.py --exp {exp_id}")

if __name__ == "__main__":
    train()

