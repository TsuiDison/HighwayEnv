import gymnasium as gym
from stable_baselines3 import PPO
import highway_env  # noqa: F401
import custom_env # Register custom env
import time
import os

def test():
    model_path = "model_complex" # Changed model name
    
    if not os.path.exists(model_path + ".zip"):
        print(f"Model not found at {model_path}.zip. Please run training first.")
        return

    # Load the model
    # We might need to map to CPU if trained on GPU and testing on CPU, but usually auto-handled
    model = PPO.load(model_path)
    print(f"Model loaded from {model_path}\n")

    # Create the environment with standard Kinematics observation
    config = {
        "duration": 1000,  # Very long duration, will stop only on crash
        "collision_reward": -5.0,
        "observation": {
            "type": "Kinematics"
        }
    }
    # Use custom environment
    env = gym.make("highway-complex-v0", render_mode="rgb_array", config=config)

    episodes = 5
    for e in range(episodes):
        obs, info = env.reset()
        crashed = False
        step = 0
        total_reward = 0
        start_x = env.unwrapped.vehicle.position[0]

        print(f"Episode {e + 1} started.")
        while not crashed:
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            step += 1
            
            # Only stop when crashed
            if info.get('crashed', False):
                print(f"  Crashed at step {step}!")
                crashed = True
            
            env.render()

        # Calculate distance traveled
        end_x = env.unwrapped.vehicle.position[0]
        distance = end_x - start_x
        
        # Print episode results
        crashed = info.get('crashed', False)
        print(f"Episode {e + 1} finished.")
        print(f"  Steps: {step}")
        print(f"  Distance: {distance:.2f} m")
        print(f"  Total Reward: {total_reward:.2f}")
        print(f"  Crashed: {crashed}")
        print()

    env.close()

if __name__ == "__main__":
    test()
