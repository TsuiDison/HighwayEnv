import gymnasium as gym
from stable_baselines3 import PPO
import highway_env  # noqa: F401
import time
import numpy as np
import os
import matplotlib.pyplot as plt

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
        "duration": 1000000000000000000,
        "collision_reward": -5.0,
    }

    # Path to the trained model
    # Assuming script is run from MY_TEST folder, and model is in ../highway_ppo/
    model_path = "../highway_ppo/model_discrete_v2" 
    
    # Fallback if running from root
    if not os.path.exists(model_path + ".zip") and os.path.exists("highway_ppo/model_discrete_v2.zip"):
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
    
    # Run 1000 episodes for evaluation
    episodes = 1000
    rewards_list = []
    distances_list = []
    
    print(f"Starting evaluation for {episodes} episodes...")

    for e in range(episodes):
        obs, info = env.reset()
        done = truncated = False
        total_reward = 0.0
        total_distance = 0.0
        
        while not (done or truncated):
            action, _ = model.predict(obs)
            
            obs, reward, done, truncated, info = env.step(action)
            
            total_reward += reward
            
            # Calculate distance for reporting
            try:
                current_speed = env.unwrapped.vehicle.speed
                step_distance = current_speed / 15.0
                total_distance += step_distance
            except AttributeError:
                pass

            # env.render() # Commented out for faster evaluation
            # time.sleep(0.5) # Removed delay
            
            if info.get('crashed', False):
                # print("Vehicle Crashed!") # Optional: reduce noise
                break
        
        rewards_list.append(total_reward)
        distances_list.append(total_distance)
        print(f"Episode {e+1}: Total Reward: {total_reward:.2f}, Total Distance: {total_distance:.2f} m, Crashed: {info.get('crashed', False)}")
    
    env.close()
    print("Evaluation finished.")
    
    # Export results to txt
    results = list(zip(rewards_list, distances_list))
    # Sort by distance (index 1), descending
    results.sort(key=lambda x: x[1], reverse=True)
    
    txt_output_file = 'evaluation_data.txt'
    with open(txt_output_file, 'w') as f:
        f.write("Rank\tTotal Reward\tTotal Distance (m)\n")
        for i, (r, d) in enumerate(results):
            f.write(f"{i+1}\t{r:.2f}\t{d:.2f}\n")
    print(f"Data saved to '{txt_output_file}' (Sorted by Distance)")
    
    # Plotting results
    try:
        from scipy.stats import gaussian_kde
        plt.figure(figsize=(12, 5))
        
        # Reward Distribution
        plt.subplot(1, 2, 1)
        if len(rewards_list) > 1:
            # KDE (Continuous Probability Density)
            try:
                kde = gaussian_kde(rewards_list)
                r_min, r_max = min(rewards_list), max(rewards_list)
                # Extend range slightly for better visualization
                x_grid = np.linspace(r_min - 5, r_max + 5, 500)
                y_grid = kde(x_grid)
                plt.plot(x_grid, y_grid, color='blue', linewidth=2, label='KDE (Density)')
                plt.fill_between(x_grid, y_grid, color='blue', alpha=0.3) # Fill under curve
            except Exception as kde_err:
                print(f"KDE Error (Reward): {kde_err}")
        
        plt.title('Reward Probability Density')
        plt.xlabel('Total Reward')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(axis='y', alpha=0.75)

        # Distance Distribution
        plt.subplot(1, 2, 2)
        if len(distances_list) > 1:
            # KDE (Continuous Probability Density)
            try:
                kde = gaussian_kde(distances_list)
                d_min, d_max = min(distances_list), max(distances_list)
                x_grid = np.linspace(d_min - 5, d_max + 5, 500)
                y_grid = kde(x_grid)
                plt.plot(x_grid, y_grid, color='green', linewidth=2, label='KDE (Density)')
                plt.fill_between(x_grid, y_grid, color='green', alpha=0.3) # Fill under curve
            except Exception as kde_err:
                print(f"KDE Error (Distance): {kde_err}")

        plt.title('Distance Probability Density')
        plt.xlabel('Distance (m)')
        plt.ylabel('Density')
        plt.legend()
        plt.grid(axis='y', alpha=0.75)

        plt.tight_layout()
        output_file = 'evaluation_results.png'
        plt.savefig(output_file)
        print(f"Plots saved to '{output_file}'")
        # plt.show() # Optional: show if supported
    except ImportError:
        print("Error: scipy is required for KDE plots. Please install it using 'pip install scipy'.")
    except Exception as e:
        print(f"Error plotting results: {e}")
