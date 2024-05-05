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
        # This portion of the code is just initializing global variables which are going to be used often
        self.trafficlight_id = None  
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
        # self.action_space = spaces.Discrete(5) We've chosen action_space to be 5 because we have 5 different combination of traffic lights 
        self.action_space = spaces.Discrete(5) 
        #Observation space refers to what is returned in the _get_observation() function. It has 5 observations: the number cars on each road(Light,Right,Top,Bot) and the state of the traffic lights
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(5,)) 
        # Through this we're setting up the file where the data logging will happen
        logging.basicConfig(filename='ProjectData.csv', 
        level=logging.INFO,format='%(asctime)s,%(message)s')  


    # Reset is the first function that this program executes. It marks the start of the program.
    def reset(self):
        # We can choose sumobinary to be either sumo-gui or sumo. Over all, we're just configuring the sumocfg file using TRACI in the next few lines of code
        sumoBinary = sumolib.checkBinary('sumo-gui')
        traci.start([sumoBinary, "-c", 'C:\Program Files (x86)\Eclipse\Sumo\Project\Simulation.sumocfg'])
        self.trafficlight_id = traci.trafficlight.getIDList()[0]
        self.counter=0
        self.lightCounter=0
        # Before the simulation starts , we call _get_Observation to store some value in the global variables, else they would be NAN and would cause problems.
        observation = self._get_observation()  
        self.previous_observation = observation  # Store it 
        logging.info(f"Episode Start - Observation: {observation}")
       
        return observation  

    def step(self, action):
        if self.counter==0:
            self._apply_action(action)
        self.counter+=1
        self.lightCounter+=1
        # We're taking an action and checking if the program is complet after every 5 seconds in this loop
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

        else: 
            #If it is not time to apply action, we're only getting the observation without taking an action
            if np.all(np.isfinite(self.previous_observation)):  
                return self.previous_observation, 0, False, {}
            else:
                observation = self._get_observation() 
                return observation, 0, False, {}  

            
  
    def close(self):
        traci.close()

    # Helper functions
    def _get_observation(self):
            # The variable names are self-explantory. We get the number of cars on each lane, each lanes' wait time and the state of the traffic ligth in this function.
            self.left = traci.edge.getLastStepVehicleNumber("fromLeft")
            self.right = traci.edge.getLastStepVehicleNumber("fromRight")
            self.top = traci.edge.getLastStepVehicleNumber("fromTop")
            self.bot = traci.edge.getLastStepVehicleNumber("fromBottom")
            # We're getting the highest traffic lane because we need to prioritize the traffic of the highest Traffic lane, it moves on to become the most important reward component
            self.currentHighestTrafficLane=self._getHighestTrafficLane()


            waiting_time_left = max(traci.edge.getWaitingTime("fromLeft"), 0) 
            waiting_time_right = max(traci.edge.getWaitingTime("fromRight"), 0)
            waiting_time_top = max(traci.edge.getWaitingTime("fromTop"), 0) 
            waiting_time_bot = max(traci.edge.getWaitingTime("fromBottom"), 0)

            self.previous_wait=waiting_time_top+waiting_time_bot+waiting_time_right+waiting_time_left
            self.current_light_phase=traci.trafficlight.getRedYellowGreenState(self.trafficlight_id)
            self.totalCars=self.left+self.right+self.top+self.bot

            if self.left == 0 and self.right == 0 and self.top == 0 and self.bot == 0:
                light_encoding = 0  
            else: 
                    # Light encoding :0 => ggggrrrrrrrrrrrr (Which is Top)
                light_encoding = 0  
                if self.current_light_phase == "rrrrggggrrrrrrrr":
                    light_encoding = 1
                elif self.current_light_phase == "rrrrrrrrggggrrrr":
                    light_encoding = 2
                elif self.current_light_phase == "rrrrrrrrrrrrgggg":
                    light_encoding = 3
                elif self.current_light_phase == "rrrrrrrrrrrrrrrr":
                    light_encoding = 4
                

            observation = np.array([
                self.left, 
                self.right, 
                self.top, 
                self.bot,
                light_encoding
            ]) 
         

            print("Vehicle Counts (before return):", self.left, self.right, self.top, self.bot)
            print("Final Observation:", observation)
            

            # The entire point of this function is to return the 5 components of observation , which would then be used by the gym and stable_baselines 3 for reinforced learning
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
            # this if statement is responsible for selecting the traffic light signals
            traci.trafficlight.setRedYellowGreenState(self.trafficlight_id, light_phases[action])
            print("SUMO Output after applying Action:", traci.trafficlight.getRedYellowGreenState(self.trafficlight_id))  # Print SUMO's response    
        pass


    def _compute_reward(self):
        #This is the main function of the gym environment
        # It first calculates the total wait time of the cars using the code below. Then it decides if the wait time reduced , compared to the previous observation or not
        current_total_wait = sum(traci.edge.getWaitingTime(edge_id) for edge_id in ["fromLeft", "fromRight", "fromTop", "fromBottom"])
        wait_change = self.previous_wait - current_total_wait 
        
        #The throughput is calculated(Number of cars that are waiting for the traffic light to open)
        throughput = traci.edge.getLastStepHaltingNumber("fromLeft") + \
                    traci.edge.getLastStepHaltingNumber("fromRight") + \
                    traci.edge.getLastStepHaltingNumber("fromTop") + \
                    traci.edge.getLastStepHaltingNumber("fromBottom")
        
        # The queue penalty is similar to throughput but instead of total cars waiting on lights, it calculates the total cars on the road
        queue_penalty = 0
        for lane in ["fromLeft", "fromRight", "fromTop", "fromBottom"]:
            queue_penalty -= min(0.1 * traci.edge.getLastStepVehicleNumber(lane), 2)  # Adjust penalty amount
        
        # The dominant lane factor checks if the lane with the most traffic has it's light open or not. If it does, it gains points(reward) , if it doesn't, it loses points
        dynamic_lane_factor = 0
        if self.currentHighestTrafficLane == "fromLeft": 
            dynamic_lane_factor = 0.2 if self.current_light_phase=="rrrrrrrrrrrrgggg" else -0.2
        elif self.currentHighestTrafficLane == "fromRight": 
            dynamic_lane_factor = 0.2 if self.current_light_phase=="rrrrggggrrrrrrrr" else -0.2
        elif self.currentHighestTrafficLane == "fromTop": 
            dynamic_lane_factor = 0.2 if self.current_light_phase=="ggggrrrrrrrrrrrr" else -0.2
        elif self.currentHighestTrafficLane == "fromBottom": 
            dynamic_lane_factor = 0.2 if self.current_light_phase=="rrrrrrrrggggrrrr" else -0.2

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
    model = PPO('MlpPolicy', env, verbose=1)
    model.learn(total_timesteps=100000)  
    # The RL program would run for total_timesteps time(Note that this time is calculated in multiples of 1024, so if you chose 1. You would still get 1024 timesteps)
    model.save = ("my_rl_traffic_model.zip")
 
def trainUsingPreviousLearning():
    env = SumoTrafficLightEnv()
    model = PPO('MlpPolicy', env, verbose=1)
    model.set_parameters("my_rl_traffic_model.zip")
    model.learn(total_timesteps=20000)

if __name__ == "__main__":
    
    trainUsingPreviousLearning()
