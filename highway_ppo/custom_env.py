import numpy as np
import gymnasium as gym
from gymnasium.envs.registration import register
from highway_env.envs import HighwayEnv
from highway_env.road.road import Road, RoadNetwork
from highway_env.road.lane import LineType, StraightLane, SineLane, CircularLane

class ComplexHighwayEnv(HighwayEnv):
    """
    A highway environment with random curves and merges.
    """
    def _create_road(self) -> None:
        self.road = Road(network=RoadNetwork(), np_random=self.np_random, record_history=self.config["show_trajectories"])
        net = self.road.network
        
        lanes_count = 4
        current_x = 0
        nodes = ["0"]
        n_segments = 20  # Generates a long road
        
        for i in range(n_segments):
            next_node = str(i + 1)
            # Random segment length
            length = self.np_random.uniform(150, 300)
            
            # Randomize road type
            # 0: Straight
            # 1: Gentle Curve (Sine)
            # 2: Sharp Curve / S-Turn
            segment_type = self.np_random.choice(["straight", "curve", "s_turn"], p=[0.4, 0.4, 0.2])
            
            start_x = current_x
            end_x = current_x + length
            
            # Base lanes parameters
            amp = 0
            pulsation = 0
            phase = 0
            
            if segment_type == "curve":
                amp = self.np_random.uniform(5, 15)
                pulsation = 2 * np.pi / length
                phase = 0
            elif segment_type == "s_turn":
                amp = self.np_random.uniform(10, 20)
                pulsation = 4 * np.pi / length  # Two periods (S-shape)
                phase = 0
            
            # Create lanes
            for l in range(lanes_count):
                # Line types visualization
                line_types = [LineType.STRIPED, LineType.STRIPED]
                if l == 0: line_types[0] = LineType.CONTINUOUS_LINE
                if l == lanes_count - 1: line_types[1] = LineType.CONTINUOUS_LINE
                
                # Lane positioning (y-offset)
                # Note: We apply the same curve offset to all lanes to keep them parallel
                start_y = l * 4
                end_y = l * 4
                
                if segment_type == "straight":
                    lane = StraightLane([start_x, start_y], [end_x, end_y], line_types=line_types, width=4)
                else:
                    # SineLane adds offset to y based on x
                    lane = SineLane([start_x, start_y], [end_x, end_y], amp, pulsation, phase, line_types=line_types, width=4)
                
                net.add_lane(nodes[-1], next_node, lane)
                
            # Randomly add an on-ramp (merge) in Straight segments
            if segment_type == "straight" and self.np_random.random() < 0.3:
                # Add a merging lane entering from the right (higher y)
                # Merges into the last lane (lanes_count - 1)
                
                # Ramp starts at x, y = (lanes_count * 4) + 10
                # Ends at x + 100, y = (lanes_count * 4)
                
                ramp_start_node = f"{nodes[-1]}_ramp"
                ramp_end_node = f"{nodes[-1]}_merge" # Connects to somewhere in the middle? 
                
                # To simplify valid graph connectivity for obstacles:
                # We add a lane that connects the start node to the end node, but visually appears as a merge.
                # Actually, simpler to just add a lane that terminates.
                
                ramp_lane = StraightLane(
                    [start_x, lanes_count * 4 + 6], 
                    [start_x + 100, (lanes_count - 1) * 4], 
                    width=4, 
                    line_types=[LineType.NONE, LineType.CONTINUOUS_LINE],
                    forbidden=True 
                )
                # We don't add it to the main graph flow to avoid routing issues for ego vehicle,
                # but we add it so visualizer sees it and OTHER vehicles might spawn on it if we configured them to.
                # For now, just decoration/obstacle potential.
                # net.add_lane(nodes[-1], next_node, ramp_lane) 
            
            nodes.append(next_node)
            current_x = end_x

# Register the environment
register(
    id='highway-complex-v0',
    entry_point='custom_env:ComplexHighwayEnv',
)
