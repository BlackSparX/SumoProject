import gym
import sumolib
from gym import spaces
import traci
import numpy as np
import logging
import matplotlib.pyplot as plt
import csv
from stable_baselines3 import PPO  # Make sure this is installed

class SumoTrafficLightEnv(gym.Env):
    def __init__(self):
        self.trafficlight_id = None  # Initialize to be retrieved later
        self.previous_wait= None
        self.model = None
        self.current_light_phase=None
        self.counter=0
        self.lightCounter=0
        self.currentHighestTrafficLane=None
        self.totalCars=None
        self.left=None
        self.right=None
        self.top=None
        self.bot=None
        self.previous_observation = None
        # Define action and observation spaces 
        self.action_space = spaces.Discrete(5)  # Example: 4 possible signal combinations
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(5,)) 
        logging.basicConfig(filename='ProjectData.csv', 
        level=logging.INFO,format='%(asctime)s,%(message)s')  # Configure logging
  # Example: car counts on each side
    def save_model(self, filename):
        self.model.save(filename)

    def load_model(self, filename):
        self.model = PPO.load(filename) 

    def reset(self):
        sumoBinary = sumolib.checkBinary('sumo-gui')
        traci.start([sumoBinary, "-c", 'C:\Program Files (x86)\Eclipse\Sumo\Project\Simulation.sumocfg'])
        self.trafficlight_id = traci.trafficlight.getIDList()[0]
        self.counter=0
        self.lightCounter=0
        observation = self._get_observation()  # Get the initial observation
        self.previous_observation = observation  # Store it 
        logging.info(f"Episode Start - Observation: {observation}")
       
        return observation  

    def step(self, action):
        # Implement traffic light change based on the 'action'
        if self.counter==0:
            self._apply_action(action)
        self.counter+=1
        self.lightCounter+=1
        traci.simulationStep()
        print(f"counter:{self.counter}")
        # ... your existing code to apply the action ...
        if self.lightCounter == 5:
            self._apply_action(action)
            observation = self._get_observation()
            reward = self._compute_reward()
            done=self._check_done()
            self.previous_observation = observation
            info = {}
            self.lightCounter=0
            logging.info(f"Step: {self.counter}, Action: {action}, Reward: {reward}, Observation: {observation}, Done: {done}")
            return observation, reward, done, info 

        else:  # Handling steps where observation is not updated
            # Option 1: Reuse previous observation (conditionally)

            if np.all(np.isfinite(self.previous_observation)):  # Check for NaNs
                return self.previous_observation, 0, False, {}
            else:
                observation = self._get_observation()  # Calculate a new observation
                return observation, 0, False, {}  

            
  
    def close(self):
        traci.close()

    # Helper functions
    def _get_observation(self):
            self.left = traci.edge.getLastStepVehicleNumber("fromLeft")
            self.right = traci.edge.getLastStepVehicleNumber("fromRight")
            self.top = traci.edge.getLastStepVehicleNumber("fromTop")
            self.bot = traci.edge.getLastStepVehicleNumber("fromBottom")
            self.currentHighestTrafficLane=self._getHighestTrafficLane()
            # ... other code ...

            waiting_time_left = max(traci.edge.getWaitingTime("fromLeft"), 0)  # Prevent NaN
            waiting_time_right = max(traci.edge.getWaitingTime("fromRight"), 0)
            waiting_time_top = max(traci.edge.getWaitingTime("fromTop"), 0)  # Prevent NaN
            waiting_time_bot = max(traci.edge.getWaitingTime("fromBottom"), 0)
            # ... do the same for other waiting time calls ... 

            self.previous_wait=waiting_time_top+waiting_time_bot+waiting_time_right+waiting_time_left
            self.current_light_phase=traci.trafficlight.getRedYellowGreenState(self.trafficlight_id)
            self.totalCars=self.left+self.right+self.top+self.bot
            # Traffic light encoding (example)
                    # New check for initial steps  
            if self.left == 0 and self.right == 0 and self.top == 0 and self.bot == 0:
                light_encoding = 0  # Assign a default encoding for the initial state
            else: 
                    # ... your existing encding logic ...
                light_encoding = 0  # If 'ggggrrrrrrrrrrrr' 
                if self.current_light_phase == "rrrrggggrrrrrrrr":
                    light_encoding = 1
                elif self.current_light_phase == "rrrrrrrrggggrrrr":
                    light_encoding = 2
                elif self.current_light_phase == "rrrrrrrrrrrrgggg":
                    light_encoding = 3
                elif self.current_light_phase == "rrrrrrrrrrrrrrrr":
                    light_encoding = 4
                # ... add cases for other light phases

            observation = np.array([
                self.left, 
                self.right, 
                self.top, 
                self.bot,
                light_encoding
            ]) 
         
    # ... all your other code ... 
            print("Vehicle Counts (before return):", self.left, self.right, self.top, self.bot)
            print("Final Observation:", observation)
            


            return observation
             

    def _getHighestTrafficLane(self):
            left = traci.edge.getLastStepVehicleNumber("fromLeft")
            right = traci.edge.getLastStepVehicleNumber("fromRight")
            top = traci.edge.getLastStepVehicleNumber("fromTop")
            bot = traci.edge.getLastStepVehicleNumber("fromBottom")
            if(left>=right&left>=top&left>=bot):
                 return "fromLeft"
            
            elif(right>=left&right>=top&right>=bot):
                 return "fromRight"
            
            elif(bot>=left&bot>=top&bot>=right):
                 return "fromBottom"
            
            else: return "fromTop"
            
    def _apply_action(self, action):

        light_phases = {
            0: "ggggrrrrrrrrrrrr",  # Top
            1: "rrrrggggrrrrrrrr",  # Right
            2: "rrrrrrrrggggrrrr",  # Bottom
            3: "rrrrrrrrrrrrgggg",  # Left
            4: "rrrrrrrrrrrrrrrr"   # Before changing light  
        }
        
        print("Action to be taken:", action)  # Print the action

        if action in light_phases:
            traci.trafficlight.setRedYellowGreenState(self.trafficlight_id, light_phases[action])
            traci.simulationStep()
            print("SUMO Output after applying Action:", traci.trafficlight.getRedYellowGreenState(self.trafficlight_id))  # Print SUMO's response    
        pass


    def _compute_reward(self):
        current_total_wait = sum(traci.edge.getWaitingTime(edge_id) for edge_id in ["fromLeft", "fromRight", "fromTop", "fromBottom"])
        wait_change = self.previous_wait - current_total_wait 
        
        throughput = traci.edge.getLastStepHaltingNumber("fromLeft") + \
                    traci.edge.getLastStepHaltingNumber("fromRight") + \
                    traci.edge.getLastStepHaltingNumber("fromTop") + \
                    traci.edge.getLastStepHaltingNumber("fromBottom")
        
        queue_penalty = 0
        for lane in ["fromLeft", "fromRight", "fromTop", "fromBottom"]:
            queue_penalty -= min(0.1 * traci.edge.getLastStepVehicleNumber(lane), 2)  # Adjust penalty amount
        
        dynamic_lane_factor = 0
        if self.currentHighestTrafficLane == "fromLeft": 
            dynamic_lane_factor = 0.2 if self.current_light_phase=="rrrrrrrrrrrrgggg" else -0.2
        elif self.currentHighestTrafficLane == "fromRight": 
            dynamic_lane_factor = 0.2 if self.current_light_phase=="rrrrggggrrrrrrrr" else -0.2
        elif self.currentHighestTrafficLane == "fromTop": 
            dynamic_lane_factor = 0.2 if self.current_light_phase=="ggggrrrrrrrrrrrr" else -0.2
        elif self.currentHighestTrafficLane == "fromBottom": 
            dynamic_lane_factor = 0.2 if self.current_light_phase=="rrrrrrrrggggrrrr" else -0.2
        # ... add similar cases for other directions 

        reward =10-( wait_change * 0.4 + throughput * 0.3 + queue_penalty + dynamic_lane_factor )
        return reward



    def _check_done(self):
        if(self.counter>=3600):
             self.close()
            
             return True
        elif(self.totalCars>=100):
             self.close()
             
             return True
        else: return False


       
def train_rl_agent():
    env = SumoTrafficLightEnv()
    model = PPO('MlpPolicy', env, verbose=1)  # Create a NEW model
    env.model=model
    model.learn(total_timesteps=100000)  
    model_save_path = "my_rl_traffic_model.zip"
    env.save_model(model_save_path) 
 
def trainUsingPreviousLearning():
    env = SumoTrafficLightEnv()
    model = PPO('MlpPolicy', env, verbose=1)
    model.set_parameters("my_rl_traffic_model.zip")
    model.learn(total_timesteps=20000)

if __name__ == "__main__":
    
    trainUsingPreviousLearning()
