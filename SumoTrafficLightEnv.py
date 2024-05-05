import gym
import sumolib
from gym import spaces
import traci
import numpy as np

class SumoTrafficLightEnv(gym.Env):
    def __init__(self):
        self.trafficlight_id = None  # Initialize to be retrieved later
        self.previous_wait= None
        self.current_light_phase=None
        self.counter=0
        self.lightCounter=0
        self.currentHighestTrafficLane=None
        self.totalCars=None
        self.left=None
        self.right=None
        self.top=None
        self.bot=None
        # Define action and observation spaces 
        self.action_space = spaces.Discrete(5)  # Example: 4 possible signal combinations
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(8,))  # Example: car counts on each side

    def reset(self):
        sumoBinary = sumolib.checkBinary('sumo-gui')

        traci.start([sumoBinary, "-c", 'C:\Program Files (x86)\Eclipse\Sumo\Project\Simulation.sumocfg'])
        self.trafficlight_id = traci.trafficlight.getIDList()[0]
        return self._get_observation() 

    def step(self, action):
        # Implement traffic light change based on the 'action'
        self._apply_action(action)
        self.counter+=1
        traci.simulationStep()
        if(self.counter%5==0):
             observation = self._get_observation()
             reward = self._compute_reward() 
        # Collect new observation, calculate reward
        
        
        done = self._check_done()  # Check if simulation is over

        info = {}  # Additional info for debugging

        return observation, reward, done, info 


    def close(self):
        traci.close()

    # Helper functions
    def _get_observation(self):
            self.left = traci.edge.getLastStepVehicleNumber("fromLeft")
            self.right = traci.edge.getLastStepVehicleNumber("fromRight")
            self.top = traci.edge.getLastStepVehicleNumber("fromTop")
            self.bot = traci.edge.getLastStepVehicleNumber("fromBottom")
            self.currentHighestTrafficLane=self._getHighestTrafficLane()
            self.previous_wait=traci.edge.getWaitingTime("fromLeft")+traci.edge.getWaitingTime("fromRight")+traci.edge.getWaitingTime("fromTop")+traci.edge.getWaitingTime("fromBottom")
            self.current_light_phase=traci.trafficlight.getRedYellowGreenState(self.trafficlight_id)
            self.totalCars=self.left+self.right+self.top+self.bot
            observation = {
                'left_cars': self.left,
                'right_cars': self.right,
                'top_cars': self.top,
                'bottom_cars': self.bot,
                'traffic_light_phase': self.current_light_phase  # Example encoding 
            }
            return {self.totalCars}

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
            4: "rrrrrrrrrrrrrrrr"   #Before changing light  
            
        # ... add more phases
        }
        if action in light_phases:
            traci.trafficlight.setRedYellowGreenState(self.trafficlight_id, light_phases[action])
        else:
        # Handle invalid action (optional)
            pass

    def _compute_reward(self):
        current_total_wait = sum(traci.edge.getWaitingTime(edge_id) for edge_id in ["fromLeft", "fromRight", "fromTop", "fromBottom"])
        wait_change = self.previous_wait - current_total_wait  

        throughput = traci.edge.getLastStepHaltingNumber("fromLeft") + traci.edge.getLastStepHaltingNumber("fromRight") + traci.edge.getLastStepHaltingNumber("fromTop") +  traci.edge.getLastStepHaltingNumber("fromBottom")

        dominant_lane_penalty = 0
        if self.currentHighestTrafficLane == "fromLeft" and  not self.current_light_phase=="rrrrrrrrrrrrgggg":  # Prioritize lanes with 'g' at the start
            dominant_lane_penalty = -0.5
        elif self.currentHighestTrafficLane == "fromRight" and  not self.current_light_phase=="rrrrggggrrrrrrrr":  # Prioritize lanes with 'g' at the start
            dominant_lane_penalty = -0.5
        elif self.currentHighestTrafficLane == "fromTop" and  not self.current_light_phase=="ggggrrrrrrrrrrrr":  # Prioritize lanes with 'g' at the start
            dominant_lane_penalty = -0.5
        elif self.currentHighestTrafficLane == "fromBottom" and  not self.current_light_phase=="rrrrrrrrggggrrrr":  # Prioritize lanes with 'g' at the start
            dominant_lane_penalty = -0.5
        # ... add similar cases for other directions 

        reward = wait_change * 0.6 + throughput * 0.2 + dominant_lane_penalty

        # ... (rest of your function to update previous_wait and current_light_phase)

        return reward


    def _check_done(self):
        if(self.counter>=3600):
             return True
        elif(self.totalCars>=100):
             return True
        else: return False
       
from stable_baselines3 import PPO  

env = SumoTrafficLightEnv()
model = PPO('MlpPolicy', env, verbose=1)  
model.learn(total_timesteps=50000)  
