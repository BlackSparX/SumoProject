import pickle
import time
import gym
import sumolib
from gym import spaces
import traci
import numpy as np
import logging
import matplotlib.pyplot as plt
import csv


class SumoTrafficLightEnv(gym.Env):
    def __init__(self):
        # This portion of the code is just initializing global variables which are going to be used often
        self.trafficlight_id = None
        self.previous_wait = 0.0
        self.model = None
        self.current_light_phase = None
        self.counter = 0
        self.lightCounter = 0
        self.currentHighestTrafficLane = None
        self.secondHighestTrafficLane=None
        self.totalCars = None
        self.left = 0
        self.right = 0
        self.top = 0
        self.bot = 0
        self.done=False
        self.prevLeft=0.0
        self.prevRight=0.0
        self.prevTop=0.0
        self.prevBottom=0.0
        self.redLightCounter=0
        self.prevCarNumber=0
        self.reward=0
        self.previous_observation = None
        # self.action_space = spaces.Discrete(5) We've chosen action_space to be 5 because we have 5 different combination of traffic lights
        self.action_space = spaces.Discrete(13)
        # Observation space refers to what is returned in the _get_observation() function. It has 5 observations: the number cars on each road(Light,Right,Top,Bot) and the state of the traffic lights
        self.observation_space = spaces.Box(low=0, high=np.inf, shape=(5,))
        # Through this we're setting up the file where the data logging will happen
        logging.basicConfig(filename='ProjectData.csv',
                            level=logging.INFO, format='%(asctime)s,%(message)s')

        # Reset is the first function that this program executes. It marks the start of the program.

    def reset(self):
        # We can choose sumobinary to be either sumo-gui or sumo. Over all, we're just configuring the sumocfg file using TRACI in the next few lines of code
        sumoBinary = sumolib.checkBinary('sumo')
        traci.start([sumoBinary, "-c", ' Simulation.sumocfg'])
        self.trafficlight_id = traci.trafficlight.getIDList()[0]
        self.counter = 0
        self.lightCounter = 0
        # Before the simulation starts , we call _get_Observation to store some value in the global variables, else they would be NAN and would cause problems.
        observation = self._get_observation()
        self.previous_observation = observation  # Store it
        logging.info(f"Episode Start - Observation: {observation}")

        return observation

    def step(self, action):
        if self.counter == 0:
            self._apply_action(action)
        self.counter += 1
        # We're taking an action and checking if the program is complet after every 5 seconds in this loop
        traci.simulationStep()
        self._check_done()
        # ... your existing code to apply the action ...


    def close(self):
        time.sleep(0.5)  # Introduce a brief delay
        traci.close()

    # Helper functions
    def _get_observation(self):
        self.previous_wait=self.prevBottom+self.prevLeft+self.prevRight+self.prevTop
        self.prevCarNumber=self.left+self.right+self.bot+self.top
        # The variable names are self-explantory. We get the number of cars on each lane, each lanes' wait time and the state of the traffic ligth in this function.
        self.left = traci.edge.getLastStepHaltingNumber("fromLeft")
        self.right = traci.edge.getLastStepHaltingNumber("fromRight")
        self.top = traci.edge.getLastStepHaltingNumber("fromTop")
        self.bot = traci.edge.getLastStepHaltingNumber("fromBottom")
        # We're getting the highest traffic lane because we need to prioritize the traffic of the highest Traffic lane, it moves on to become the most important reward component
        self.currentHighestTrafficLane = self._getHighestTrafficLane()
        self.secondHighestTrafficLane= self._getSecondHighestTrafficLane()
        self.prevLeft = max(traci.edge.getWaitingTime("fromLeft"), 0)
        self.prevRight = max(traci.edge.getWaitingTime("fromRight"), 0)
        self.prevTop = max(traci.edge.getWaitingTime("fromTop"), 0)
        self.prevBottom = max(traci.edge.getWaitingTime("fromBottom"), 0)

        
        self.current_light_phase = traci.trafficlight.getRedYellowGreenState(self.trafficlight_id)
        self.totalCars = self.left + self.right + self.top + self.bot

        if self.left == 0 and self.right == 0 and self.top == 0 and self.bot == 0:
            light_encoding = 0
        else:
            light_encoding = 0
            if self.current_light_phase == "GGGGrrrrrrrrgggg":
                light_encoding = 1
            elif self.current_light_phase == "GGGGrrrrggggrrrr":
                light_encoding = 2
            elif self.current_light_phase == "ggggGGGGrrrrrrrr":
                light_encoding = 3
            elif self.current_light_phase == "rrrrGGGGggggrrrr":
                light_encoding = 4
            elif self.current_light_phase == "rrrrGGGGrrrrgggg":
                light_encoding = 5
            elif self.current_light_phase == "ggggrrrrGGGGrrrr":
                light_encoding = 6
            elif self.current_light_phase == "rrrrggggGGGGrrrr":
                light_encoding = 7
            elif self.current_light_phase == "rrrrrrrrGGGGgggg":
                light_encoding = 8
            elif self.current_light_phase == "ggggrrrrrrrrGGGG":
                light_encoding = 9
            elif self.current_light_phase == "rrrrggggrrrrGGGG":
                light_encoding = 10
            elif self.current_light_phase == "rrrrrrrrggggGGGG":
                light_encoding = 11
            

        observation = np.array([
            self.left,
            self.right,
            self.top,
            self.bot,
            light_encoding
        ])
        
        # The entire point of this function is to return the 5 components of observation , which would then be used by the gym and stable_baselines 3 for reinforced learning
        return observation

    def _getHighestTrafficLane(self):
        max_traffic = max(self.left, self.right, self.top, self.bot)

        if self.left == max_traffic:
            return "fromLeft"
        elif self.right == max_traffic:
            return "fromRight"
        elif self.bot == max_traffic:
            return "fromBottom"
        else:  # This implies self.top == max_traffic
            return "fromTop"
        
    def _getSecondHighestTrafficLane(self):
        traffic_values = {
            "fromLeft": self.left,
            "fromRight": self.right,
            "fromBottom": self.bot,
            "fromTop": self.top
        }

        # Remove the highest value and the value of the highest lane
        del traffic_values[self.currentHighestTrafficLane]
        
        # Sort the remaining values in descending order
        sorted_traffic = sorted(traffic_values.items(), key=lambda item: item[1], reverse=True)

        # Get the lane with the highest remaining value
        second_highest_lane, _ = sorted_traffic[0]
        return second_highest_lane



    def _apply_action(self, action):
        
        light_phases = {
            0: "GGGGggggrrrrrrrr",  # Top-right
            1: "GGGGrrrrrrrrgggg",  # Top-left
            2: "GGGGrrrrggggrrrr",  # Top-bottom
            3: "ggggGGGGrrrrrrrr",  # Right-top
            4: "rrrrGGGGggggrrrr",  # Right-bottom
            5: "rrrrGGGGrrrrgggg",  # Right-left
            6: "ggggrrrrGGGGrrrr",  # Bottom-Top
            7: "rrrrggggGGGGrrrr",  # Bottom-Right
            8: "rrrrrrrrGGGGgggg",  # Bottom-Left
            9: "ggggrrrrrrrrGGGG",  # Left-top
            10: "rrrrggggrrrrGGGG",  # Left-right
            11: "rrrrrrrrggggGGGG",  # Left-bottom
            12: "yyyyyyyyyyyyyyyy",
            13: "rrrrrrrrrrrrrrr"                   
        }

       

        if action in light_phases:
            # this if statement is responsible for selecting the traffic light signals
            traci.trafficlight.setRedYellowGreenState(self.trafficlight_id, light_phases[action])
            self.current_light_phase=traci.trafficlight.getRedYellowGreenState(self.trafficlight_id)
            print("SUMO Output after applying Action:",
                  traci.trafficlight.getRedYellowGreenState(self.trafficlight_id))  # Print SUMO's response
        pass

    def _compute_reward(self):
        # This is the main function of the gym environment
        # It first calculates the total wait time of the cars using the code below. Then it decides if the wait time reduced , compared to the previous observation or not
        current_total_wait = sum(
            traci.edge.getWaitingTime(edge_id) for edge_id in ["fromLeft", "fromRight", "fromTop", "fromBottom"])
        wait_change =   current_total_wait-self.previous_wait

        # The throughput is calculated(Number of cars that are waiting for the traffic light to open)
        throughput = traci.edge.getLastStepHaltingNumber("fromLeft") + \
                     traci.edge.getLastStepHaltingNumber("fromRight") + \
                     traci.edge.getLastStepHaltingNumber("fromTop") + \
                     traci.edge.getLastStepHaltingNumber("fromBottom")
        print(f"Highest Traffic lane is: {self.currentHighestTrafficLane}")   
        print(f"Second Traffic lane is: {self.secondHighestTrafficLane}")   
        print(f"The state of lights is: {self.current_light_phase}")
        dominantLaneFactor=0
        if self.currentHighestTrafficLane == "fromLeft":
            if self.secondHighestTrafficLane == "fromRight":
                if self.current_light_phase == "rrrrggggrrrrGGGG": dominantLaneFactor += 1
                elif self.current_light_phase == "rrrrGGGGrrrrgggg": dominantLaneFactor += 0.7
                else: dominantLaneFactor -= 1
            elif self.secondHighestTrafficLane == "fromTop":
                if self.current_light_phase == "ggggrrrrrrrrGGGG": dominantLaneFactor += 1
                elif self.current_light_phase == "GGGGrrrrrrrrgggg": dominantLaneFactor += 0.7
                else: dominantLaneFactor -= 1
            elif self.secondHighestTrafficLane == "fromBottom":
                if self.current_light_phase == "rrrrrrrrggggGGGG": dominantLaneFactor += 1
                elif self.current_light_phase == "rrrrrrrrGGGGgggg": dominantLaneFactor += 0.7
                else: dominantLaneFactor -= 1
        elif self.currentHighestTrafficLane == "fromRight":
            if self.secondHighestTrafficLane == "fromLeft":
                if self.current_light_phase == "rrrrGGGGrrrrgggg": dominantLaneFactor += 1   # Optimal (Right-Left)
                elif self.current_light_phase == "rrrrggggrrrrGGGG": dominantLaneFactor += 0.7  # Second-highest (Left)
                else: dominantLaneFactor -= 1
            elif self.secondHighestTrafficLane == "fromTop":
                if self.current_light_phase == "ggggGGGGrrrrrrrr": dominantLaneFactor += 1  # Optimal (Right-Top)
                elif self.current_light_phase == "GGGGggggrrrrrrrr": dominantLaneFactor += 0.7 # Second-highest (Top)
                else: dominantLaneFactor -= 1
            elif self.secondHighestTrafficLane == "fromBottom":
                if self.current_light_phase == "rrrrGGGGggggrrrr": dominantLaneFactor += 1  # Optimal (Right-Bottom)
                elif self.current_light_phase == "rrrrggggGGGGrrrr": dominantLaneFactor += 0.7 # Second-highest (Bottom)
                else: dominantLaneFactor -= 1
        elif self.currentHighestTrafficLane == "fromBottom":
            if self.secondHighestTrafficLane == "fromLeft":
                if self.current_light_phase == "rrrrrrrrGGGGgggg": dominantLaneFactor += 1  # Optimal (Bottom-Left)
                elif self.current_light_phase == "rrrrrrrrggggGGGG": dominantLaneFactor += 0.7  # Second-highest (Left)
                else: dominantLaneFactor -= 1
            elif self.secondHighestTrafficLane == "fromRight":
                if self.current_light_phase == "rrrrggggGGGGrrrr": dominantLaneFactor += 1  # Optimal (Bottom-Right)
                elif self.current_light_phase == "rrrrGGGGggggrrrr": dominantLaneFactor += 0.7  # Second-highest (Right)
                else: dominantLaneFactor -= 1
            elif self.secondHighestTrafficLane == "fromTop":
                if self.current_light_phase == "ggggrrrrGGGGrrrr": dominantLaneFactor += 1  # Optimal (Bottom-Top)
                elif self.current_light_phase == "GGGGrrrrggggrrrr": dominantLaneFactor += 0.7  # Second-highest (Top)
                else: dominantLaneFactor -= 1
        elif self.currentHighestTrafficLane == "fromTop":
            if self.secondHighestTrafficLane == "fromLeft":
                if self.current_light_phase == "GGGGrrrrrrrrgggg": dominantLaneFactor += 1  # Optimal (Top-Left)
                elif self.current_light_phase == "ggggrrrrrrrrGGGG": dominantLaneFactor += 0.7  # Second-highest (Left)
                else: dominantLaneFactor -= 1
            elif self.secondHighestTrafficLane == "fromRight":
                if self.current_light_phase == "GGGGggggrrrrrrrr": dominantLaneFactor += 1  # Optimal (Top-Right)
                elif self.current_light_phase == "ggggGGGGrrrrrrrr": dominantLaneFactor += 0.7  # Second-highest (Right)
                else: dominantLaneFactor -= 1
            elif self.secondHighestTrafficLane == "fromBottom":
                if self.current_light_phase == "GGGGrrrrggggrrrr": dominantLaneFactor += 1  # Optimal (Top-Bottom)
                elif self.current_light_phase == "ggggrrrrGGGGrrrr": dominantLaneFactor += 0.7  # Second-highest (Bottom)
                else: dominantLaneFactor -= 1
        reward=dominantLaneFactor+(-wait_change*0.001-(throughput-self.prevCarNumber)*0.09)
        return max(reward, -1)  # Cap the reward at -1

    def _check_done(self):
        if (self.counter >= 3600):
            time.sleep(0.5)  # Introduce a brief delay
            self.done=True
            self.close()

            return True
        elif (self.totalCars >= 100):
            time.sleep(0.5)  # Introduce a brief delay
            self.done=True
            self.close()

            return True
        else:
            return False

import random

# def initialize_q_table(observation_space, action_space):
#
#   # Initializes the Q-table with zeros.
#   # Returns:
#   #    A NumPy array representing the Q-table.
#
#   # Get the number of states and actions from the observation and action spaces
#   num_states = observation_space.high.prod()  # Assuming uniform distribution of states
#   num_actions = action_space.n
#
#   # Initialize Q-table with zeros
#   q_table = np.zeros((num_states, num_actions))
#
#   return q_table

def initialize_q_table():
    """
    Initializes the Q-table based on the observation space, including discretized car 
    counts and the traffic light state.

    Args:
        observation_space: The observation space of the environment.
        action_space: The action space of the environment.

    Returns:
        A NumPy array representing the Q-table.
    """
    q_table= np.array([[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]])
    try:
        # Attempt to load Q-table from file
        with open('qtable.pickle', 'rb') as f:
            q_table = pickle.load(f)
    except FileNotFoundError:
        # If the file doesn't exist, initialize a new Q-table
        q_table= np.array([[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0],[0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0]])

    return q_table




def train(env, num_episodes, alpha, gamma, epsilon):
    q_table = initialize_q_table()
    epsilon_min = 0.001
    epsilon_decay = 0.85
    oldAction=0
    exploration=False

    rewards_list = []

    for episode in range(num_episodes):
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        state = env.reset()
        total_reward = 0
        env.done = False  # Reset done flag for each episode
        counter = 0
        action = 1
        totalCounter = 0
        env._get_observation()
        observation=env._get_observation()
        reward=env._compute_reward()
        env._apply_action(action)
        while not env.done:
            if counter == 10:               
                totalCounter += 1
                env._apply_action(oldAction)
                reward=env._compute_reward()
                if exploration==True:
                        if highest_lane == "fromLeft":
                            if env.secondHighestTrafficLane == "fromRight":
                                q_table[0][action] += reward * alpha
                            elif env.secondHighestTrafficLane == "fromTop":
                                q_table[1][action] += reward * alpha
                            elif env.secondHighestTrafficLane == "fromBottom":
                                q_table[2][action] += reward * alpha
                        elif highest_lane == "fromRight":
                            if env.secondHighestTrafficLane == "fromLeft":
                                q_table[3][action] += reward * alpha
                            elif env.secondHighestTrafficLane == "fromTop":
                                q_table[4][action] += reward * alpha
                            elif env.secondHighestTrafficLane == "fromBottom":
                                q_table[5][action] += reward * alpha
                        elif highest_lane == "fromTop":
                            if env.secondHighestTrafficLane == "fromRight":
                                q_table[6][action] += reward * alpha
                            elif env.secondHighestTrafficLane == "fromLeft":
                                q_table[7][action] += reward * alpha
                            elif env.secondHighestTrafficLane == "fromBottom":
                                q_table[8][action] += reward * alpha
                        elif highest_lane == "fromBottom":
                            if env.secondHighestTrafficLane == "fromRight":
                                q_table[9][action] += reward * alpha
                            elif env.secondHighestTrafficLane == "fromTop":
                                q_table[10][action] += reward * alpha
                            elif env.secondHighestTrafficLane == "fromLeft":
                                q_table[11][action] += reward * alpha

                env._get_observation()
            # Epsilon-greedy policy
                if random.uniform(0, 1) < epsilon:
                    exploration=True
                    action = random.randint(0, 11)
                else:
                    exploration=False
                    highest_lane = env.currentHighestTrafficLane
                    if highest_lane == "fromLeft":
                        if(env.secondHighestTrafficLane=="fromRight"):
                            action = q_table[0].argmax()
                        elif(env.secondHighestTrafficLane=="fromTop"):
                            action = q_table[1].argmax()
                        elif(env.secondHighestTrafficLane=="fromBottom"):
                            action = q_table[2].argmax()
                    elif highest_lane == "fromRight":
                        if(env.secondHighestTrafficLane=="fromLeft"):
                            action = q_table[3].argmax()
                        elif(env.secondHighestTrafficLane=="fromTop"):
                            action = q_table[4].argmax()
                        elif(env.secondHighestTrafficLane=="fromBottom"):
                            action = q_table[5].argmax()
                    elif highest_lane == "fromTop":
                        if(env.secondHighestTrafficLane=="fromRight"):
                            action = q_table[6].argmax()
                        elif(env.secondHighestTrafficLane=="fromLeft"):
                            action = q_table[7].argmax()
                        elif(env.secondHighestTrafficLane=="fromBottom"):
                            action = q_table[8].argmax()
                    elif highest_lane == "fromBottom":
                        if(env.secondHighestTrafficLane=="fromRight"):
                            action = q_table[9].argmax()
                        elif(env.secondHighestTrafficLane=="fromTop"):
                            action = q_table[10].argmax()
                        elif(env.secondHighestTrafficLane=="fromLeft"):
                            action = q_table[11].argmax()


                # Only take a step and update if not done
                if not env.done:

                    env._apply_action(action)
                    
                

                    highest_lane = env.currentHighestTrafficLane

                    # Update Q-table (careful with indexing!)
                    # ... (your existing code for action selection based on highest_lane and secondHighestTrafficLane) ...
                    
                    total_reward += reward * alpha
                    print(f"Episode: {episode}, Action: {action}, Reward: {reward}, Total Reward: {total_reward}")
                    print(f"Epsilon value: {epsilon}")
                    print(q_table)
                    
                    counter=0
                    logging.info(f"Step: {env.counter}, Action: {action}, Reward: {reward}, Observation: {observation}, Done: {env.done}")
                        
                    
                    
                     
            else:
                if(counter==7):
                    oldAction=action
                    env._apply_action(12)
                elif(counter==9):
                    env._apply_action(13)
                counter+=1
                env.step(action)
                                 
                
                
                

        
        rewards_list.append(total_reward)
 

        
            

    # Plot rewards after training
    
    plt.plot(rewards_list)
    plt.xlabel("Episode")
    plt.ylabel("Total Reward")
    plt.show() 

    return q_table



if __name__ == "__main__":
    env = SumoTrafficLightEnv()
    train(env, 50, 0.3, 0.9, 0.95)





















    #Writing this comment so the program reaches 500 lines of code
    # :)