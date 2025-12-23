import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
import highway_env  # noqa: F401
import os

# ==================================
#        Custom Wrapper
# ==================================
class CustomRewardWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Action is discrete: 0: LANE_LEFT, 1: IDLE, 2: LANE_RIGHT, 3: FASTER, 4: SLOWER
        # Note: The mapping depends on the environment config, but this is standard for highway-fast-v0
        
        # 1. Increase Collision Penalty
        if info.get('crashed', False):
             reward -= 5.0 
        
        # 2. Acceleration Reward / Deceleration Penalty (Discrete approximation)
        if action == 3: # FASTER
            reward += 0.5 
        elif action == 4: # SLOWER
            reward -= 0.05
            
        # 3. Distance Traveled Reward
        # Get speed from the vehicle (if available) or info
        # highway-env usually exposes the vehicle in unwrapped env
        try:
            speed = self.env.unwrapped.vehicle.speed
            # Distance = speed * dt. Default simulation frequency is 15Hz.
            # dt = 1 / 15
            distance_reward = speed / 15.0
            reward += distance_reward
        except AttributeError:
            pass # If vehicle speed is not accessible, skip this reward

        return obs, reward, done, truncated, info

# ==================================
#        Main script
# ==================================

if __name__ == "__main__":
    # Define the custom configuration
    env_config = {
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
    
    # Create wrapped environment for training
    def make_env():
        env = gym.make("highway-fast-v0", render_mode="rgb_array", config=env_config)
        env = CustomRewardWrapper(env)
        return env

    # Use SubprocVecEnv for parallel training
    env = make_vec_env(make_env, n_envs=n_cpu, vec_env_cls=SubprocVecEnv)
    
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
        n_steps=batch_size * 12 // n_cpu,
        batch_size=batch_size,
        n_epochs=10,
        learning_rate=5e-4,
        gamma=0.8,
        verbose=2,
        tensorboard_log="highway_ppo/",
    )
    
    print("Starting training...")
    # Train the agent
    model.learn(total_timesteps=int(2e4))
    # Save the agent
    save_path = "highway_ppo/model_discrete_v2"
    model.save(save_path)
    print(f"Training finished. Model saved to {save_path}.zip")
