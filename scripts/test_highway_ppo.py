import gymnasium as gym
from stable_baselines3 import PPO
import highway_env  # noqa: F401
import time
import numpy as np
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
        
        # 1. Increase Collision Penalty
        if info.get('crashed', False):
             reward -= 5.0 
        
        # 2. Acceleration Reward / Deceleration Penalty (Discrete approximation)
        if action == 3: # FASTER
            reward += 0.5 
        elif action == 4: # SLOWER
            reward -= 0.05
            
        # 3. Distance Traveled Reward
        try:
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

    # Path to the trained model
    model_path = "highway_ppo/model_discrete_v2" 

    print(f"Attempting to load model from: {model_path}")
    
    if not os.path.exists(model_path + ".zip"):
        print(f"Error: Model file '{model_path}.zip' not found.")
        print("Please run the training script first.")
        exit(1)

    # Load the trained model
    model = PPO.load(model_path)
    print("Model loaded successfully.")
    
    # Create single environment for testing
    env = gym.make("highway-fast-v0", render_mode="rgb_array", config=env_config)
    env = CustomRewardWrapper(env)
    
    # Action map for printing
    ACTION_MAP = {
        0: 'LANE_LEFT',
        1: 'IDLE',
        2: 'LANE_RIGHT',
        3: 'FASTER',
        4: 'SLOWER'
    }

    # Run only 1 episode as requested
    episodes = 1
    for e in range(episodes):
        obs, info = env.reset()
        done = truncated = False
        step = 0
        total_distance = 0.0
        print(f"\nStarting Test Episode {e+1}...")
        
        while not (done or truncated):
            action, _ = model.predict(obs)
            
            # Action is now a scalar (or 0-d array)
            action_scalar = action.item() if isinstance(action, np.ndarray) else action
            action_str = ACTION_MAP.get(action_scalar, str(action_scalar))
            
            print(f"Step {step}: Action={action_str}")
            
            obs, reward, done, truncated, info = env.step(action)
            
            # Calculate distance for reporting
            try:
                current_speed = env.unwrapped.vehicle.speed
                step_distance = current_speed / 15.0
                total_distance += step_distance
            except AttributeError:
                pass

            print(f"Step {step}: Reward: {reward:.4f}, Crashed: {info.get('crashed', False)}")
            
            env.render()
            time.sleep(0.5) # Pause for visibility
            step += 1
            
            if info.get('crashed', False):
                print("Vehicle Crashed! Episode ending.")
                break
        
        print(f"Episode finished. Total Distance Traveled: {total_distance:.2f} m")
    
    env.close()
    print("Testing finished.")
