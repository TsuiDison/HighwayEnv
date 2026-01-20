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
    """改进的离散动作环境奖励包装器"""
    def __init__(self, env):
        super().__init__(env)
        self.last_action = None
        self.last_speed = 0.0
        self.speed_history = []
        self.max_history = 5
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        
        # Action is discrete: 0: LANE_LEFT, 1: IDLE, 2: LANE_RIGHT, 3: FASTER, 4: SLOWER
        
        # 1. Collision Penalty - Strong penalty to discourage crashes
        if info.get('crashed', False):
            reward -= 10.0
        
        # 2. Get current speed for analysis
        try:
            speed = self.env.unwrapped.vehicle.speed
            self.speed_history.append(speed)
            if len(self.speed_history) > self.max_history:
                self.speed_history.pop(0)
            speed_variance = np.var(self.speed_history) if len(self.speed_history) > 1 else 0
        except AttributeError:
            speed = 0.0
            speed_variance = 0.0
        
        # 3. Smooth Action Encouragement with speed constraints
        if action == 1:  # IDLE
            reward += 0.3
        elif action == 3:  # FASTER
            reward += 0.1
            if speed > 28.0:
                reward -= 0.3
        elif action == 4:  # SLOWER
            reward += 0.1
        
        # Penalize abrupt action changes
        if self.last_action is not None:
            if (self.last_action == 3 and action == 4) or \
               (self.last_action == 4 and action == 3):
                reward -= 0.2
            
        # 4. Speed variance penalty
        if speed_variance > 0.5:
            reward -= speed_variance * 0.05
        
        # 5. Ideal speed reward
        if speed > 22:
            ideal_low, ideal_high = 25.0, 30.0
        else:
            ideal_low, ideal_high = 15.0, 25.0
        
        if ideal_low <= speed <= ideal_high:
            reward += 0.2
        else:
            speed_diff = min(abs(speed - ideal_low), abs(speed - ideal_high))
            reward -= speed_diff * 0.02
        
        self.last_action = action
        self.last_speed = speed
        return obs, reward, done, truncated, info


class ParkingRewardWrapper(gym.Wrapper):
    """针对停车场景的连续动作环境奖励优化"""
    def __init__(self, env):
        super().__init__(env)
        self.prev_distance_to_goal = None
        self.episode_step = 0
    
    def reset(self, **kwargs):
        self.prev_distance_to_goal = None
        self.episode_step = 0
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        self.episode_step += 1
        
        # 1. Collision penalty
        if info.get('crashed', False):
            reward -= 10.0
        
        # 2. Target distance reward
        try:
            if isinstance(obs, dict):
                achieved = obs.get('achieved_goal', np.array([0, 0, 0, 0, 0, 0]))
                desired = obs.get('desired_goal', np.array([0, 0, 0, 0, 0, 0]))
            else:
                achieved = obs[:6] if len(obs) >= 6 else np.array([0, 0, 0, 0, 0, 0])
                desired = obs[6:12] if len(obs) >= 12 else np.array([0, 0, 0, 0, 0, 0])
            
            pos_diff = np.sqrt((achieved[0] - desired[0])**2 + (achieved[1] - desired[1])**2)
            angle_diff = abs(achieved[4] - desired[4]) + abs(achieved[5] - desired[5])
            distance = pos_diff + angle_diff * 0.5
            
            if self.prev_distance_to_goal is not None:
                distance_improvement = self.prev_distance_to_goal - distance
                if distance_improvement > 0:
                    proximity_factor = max(0.5, 1.0 - distance / 10.0)
                    reward += distance_improvement * proximity_factor * 2.0
                else:
                    reward -= abs(distance_improvement) * 0.5
            
            self.prev_distance_to_goal = distance
            
            if self.episode_step > 1 and self.episode_step < 100:
                reward += 0.1
                    
        except (IndexError, ValueError):
            pass
        
        # 3. Success reward
        if info.get('is_success', False):
            reward += 10.0
        
        # 4. Action smoothness
        if hasattr(self, 'last_action') and self.last_action is not None:
            action_diff = np.sqrt(np.sum((action - self.last_action)**2))
            if action_diff > 0.5:
                reward -= action_diff * 0.2
        
        self.last_action = action.copy() if isinstance(action, np.ndarray) else action
        
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

    # Adjust config for scenario 1 to allow longer episodes until crash
    config = common_config
    if args.scenario == 1:
        config = config.copy()
        config["duration"] = 1000  # Extend duration for merge scenario to run until crash
        config["offroad_terminal"] = False  # Disable offroad termination, only stop on crash
        config["goal"] = None  # Disable goal termination, only stop on crash
        config["lanes_length"] = 100000  # Make the road much longer for infinite appearance
        config["vehicles_count"] = 100  # Increase number of vehicles for more dynamic traffic
        config["screen_width"] = 10000  # Increase screen width for wider view
        config["offscreen_rendering"] = True  # Enable offscreen rendering to avoid window size issues

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
        env = ParkingRewardWrapper(env)  # 使用停车专用wrapper
    else:
        env = gym.make(env_id, render_mode="rgb_array", config=config)
        env = CustomRewardWrapper(env)
        
        # For scenario 1, disable goal to prevent early termination
        if args.scenario == 1:
            env.unwrapped.goal = None
        
    # Create video directory if it doesn't exist
    video_dir = os.path.join(script_dir, "video")
    os.makedirs(video_dir, exist_ok=True)
    
    env = RecordVideo(env, video_folder=video_dir, episode_trigger=lambda x: True, name_prefix=f"test_{scenario_name}")
    
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
        
        if args.scenario == 1:
            # For merge scenario, run until crash or max steps, ignoring other terminations
            while steps < 1000 and not crashed:
                action, _ = model.predict(obs)
                obs, reward, done, truncated, info = env.step(action)
                steps += 1
                
                # Track distance for highway
                try:
                    speed = env.unwrapped.vehicle.speed
                    total_distance += speed / 15.0
                except: pass
                
                if info.get('crashed', False):
                    crashed = True
                    break
        else:
            # For other scenarios, use standard termination
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
