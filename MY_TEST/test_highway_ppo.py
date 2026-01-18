import gymnasium as gym
from stable_baselines3 import PPO
import highway_env  # noqa: F401
import time
import numpy as np
import os
import argparse
import glob
import matplotlib.pyplot as plt
from gymnasium.wrappers import RecordVideo

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
    # Parse arguments
    parser = argparse.ArgumentParser(description="Test trained Highway PPO agent")
    parser.add_argument("--run_id", type=int, default=None, help="The run ID to load (e.g., 1). Default: Latest run in the scenario folder.")
    parser.add_argument("--scenario", type=int, default=1, choices=[1, 2, 3], 
                        help="1: Highway/Merge (Distance), 2: Roundabout (Time), 3: Parking (Time)")
    args = parser.parse_args()

    # Define the custom configuration (only used for Discrete envs 1 & 2)
    common_config = {
        "action": {
            "type": "DiscreteMetaAction",
            "longitudinal": True,
            "lateral": True,
        },
        "duration": 60, # Longer duration for testing
        "collision_reward": -5.0,
    }

    # Map scenario to folder name
    scenario_names = {1: "highway_merge", 2: "roundabout", 3: "parking"}
    scenario_name = scenario_names[args.scenario]

    # Locate the model directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.join(script_dir, "highway_ppo", scenario_name)

    model_path = None
    
    if args.run_id is not None:
        # Load specific run
        run_path = os.path.join(base_dir, f"run_{args.run_id}")
        model_candidate = os.path.join(run_path, "model")
        if os.path.exists(model_candidate + ".zip"):
             model_path = model_candidate
        else:
             print(f"Warning: Model not found at {model_candidate}.zip")
    else:
        # Find latest run
        if os.path.exists(base_dir):
            existing_runs = glob.glob(os.path.join(base_dir, "run_*"))
            if existing_runs:
                def get_run_id(path):
                    try:
                        return int(os.path.basename(path).split("_")[-1])
                    except ValueError:
                        return -1
                latest_run = max(existing_runs, key=get_run_id)
                model_path = os.path.join(latest_run, "model")
                print(f"No run_id provided. Loading latest run for '{scenario_name}': {os.path.basename(latest_run)}")
        
        if model_path is None:
             print(f"No runs found in {base_dir}")
    
    if model_path is None or not os.path.exists(model_path + ".zip"):
        print(f"Error: Model not found.")
        exit(1)

    print(f"Loading model from: {model_path}")

    # Load the trained model
    model = PPO.load(model_path)
    print("Model loaded successfully.")
    
    # Create environment for testing
    env_id = "highway-fast-v0" # Default
    if args.scenario == 1:
        # Test on Merge as requested "Merge and Highway together"
        # We can test on highway or merge. Let's ask user or default to one.
        # Let's test on merge-v0 as it's harder/representative
        env_id = "merge-v0" 
        print("Testing on 'merge-v0' (Part of Scenario 1)")
    elif args.scenario == 2:
        env_id = "roundabout-v0"
        print("Testing on 'roundabout-v0'")
    elif args.scenario == 3:
        env_id = "parking-v0"
        print("Testing on 'parking-v0'")
    
    if args.scenario == 3:
        env = gym.make(env_id, render_mode="rgb_array")
    else:
        env = gym.make(env_id, render_mode="rgb_array", config=common_config)
        env = CustomRewardWrapper(env)
        
    env = RecordVideo(env, video_folder=os.path.dirname(model_path), episode_trigger=lambda x: True, name_prefix=f"test_{scenario_name}")
    
    # Run 10 episodes for evaluation
    episodes = 5
    # total_reward += reward # Deprecated in new loop structure

    print(f"Starting evaluation for {episodes} episodes...")

    for e in range(episodes):
        obs, info = env.reset()
        done = truncated = False
        steps = 0
        total_distance = 0.0
        crashed = False
        success = False
        
        while not (done or truncated):
            action, _ = model.predict(obs)
            obs, reward, done, truncated, info = env.step(action)
            steps += 1
            
            # Track distance for highway
            if args.scenario == 1:
                try:
                    speed = env.unwrapped.vehicle.speed
                    total_distance += speed / 15.0
                except: pass
            
            # Check for success in Roundabout/Parking
            if args.scenario == 2:
                # Roundabout Success Criteria: Exiting the roundabout
                # The roundabout is centered at 0. North exit is negative Y.
                # If vehicle Y is < -20 (passed the loop), consider it a success.
                try:
                    vehicle = env.unwrapped.vehicle
                    if vehicle.position[1] < -20: 
                        success = True
                        # Optional: Stop early if successful to measure time accurately
                        # done = True 
                        break 
                except: pass
            
            elif args.scenario == 3:
                # Parking Success Criteria: info['is_success']
                if info.get('is_success', False):
                    success = True
                    break

            if info.get('crashed', False):
                crashed = True
                break # Stop immediately on crash as per requirement
        
        # Result Evaluation
        time_taken = steps / 15.0 # dt = 1/15 s
        
        if args.scenario == 1:
            # Highway/Merge: Maximize Distance, Crash = Fail
            result_str = "FAILED (Crash)" if crashed else f"Distance: {total_distance:.2f} m"
            print(f"Episode {e+1}: {result_str}")
            
        elif args.scenario == 2:
            # Roundabout: Minimize Time, Crash = Fail
            if crashed:
                 print(f"Episode {e+1}: FAILED (Crash)")
            elif success:
                 print(f"Episode {e+1}: SUCCESS - Time: {time_taken:.2f} s")
            else:
                 # Timed out and didn't pass
                 print(f"Episode {e+1}: FAILED (Timeout - Did not pass)")
                 
        elif args.scenario == 3:
            # Parking: Minimize Time, Crash = Fail
            if crashed:
                 print(f"Episode {e+1}: FAILED (Crash)")
            elif success:
                 print(f"Episode {e+1}: SUCCESS - Time: {time_taken:.2f} s")
            else:
                 print(f"Episode {e+1}: FAILED (Timeout - Did not park)")
