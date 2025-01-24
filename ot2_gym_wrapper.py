import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim_class import Simulation

class OT2Env(gym.Env):
    def __init__(self, render=False, max_steps=1000):
        super(OT2Env, self).__init__()
        self.render = render
        self.max_steps = max_steps

        # Create the simulation environment
        self.sim = Simulation(num_agents=1, render=False)

        # Define action and observation space
        # They must be gym.spaces objects
        self.action_space = spaces.Box(low=np.array([-1, -1, -1, 0]), high=np.array([1, 1, 1, 1]), shape=(4,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        # keep track of the number of steps
        self.steps = 0

    def reset(self, seed=None):
        # being able to set a seed is required for reproducibility
        if seed is not None:
            np.random.seed(seed)


        self.goal_position = np.array([
            np.random.uniform(-0.18700, 0.25300),
            np.random.uniform(-0.17050, 0.21950),
            np.random.uniform(0.16940, 0.28950)
        ], dtype=np.float32)
        # Call the environment reset function
        observation = self.sim.reset(num_agents=1)
        #print(observation)
        # now we need to process the observation and extract the relevant information, the pipette position, convert it to a numpy array, and append the goal position and make sure the array is of type np.float32    
        #{'robotId_2': {'joint_states': {'joint_0': {'position': 0.0, 'velocity': 0.0, 'reaction_forces': (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 'motor_torque': 0.0}, 'joint_1': {'position': 0.0, 'velocity': 0.0, 'reaction_forces': (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 'motor_torque': 0.0}, 'joint_2': {'position': 0.0, 'velocity': 0.0, 'reaction_forces': (0.0, 0.0, 0.0, 0.0, 0.0, 0.0), 'motor_torque': 0.0}}, 'robot_position': [0.0, 0.0, 0.03], 'pipette_position': [0.073, 0.0895, 0.1195]}}
        pipette_position = np.array(observation[f'robotId_{self.sim.robotIds[0]}']['pipette_position'], dtype=np.float32)

        observation = np.concatenate([pipette_position, self.goal_position], axis=0)

        # Reset the number of steps
        self.steps = 0

        return observation, {}

    def step(self, action):
        # Execute one time step within the environment
        # since we are only controlling the pipette position, we accept 3 values for the action and need to append 0 for the drop action
        action = np.append(action, 0)

        # Call the environment step function
        observation = self.sim.run([action]) # Why do we need to pass the action as a list? Think about the simulation class.
        # now we need to process the observation and extract the relevant information, the pipette position, convert it to a numpy array, and append the goal position and make sure the array is of type np.float32
        pipette_position = np.array(observation[f'robotId_{self.sim.robotIds[0]}']['pipette_position'], dtype=np.float32)



        observation = np.concatenate([pipette_position, self.goal_position], axis=0)
        # Calculate the reward, this is something that you will need to experiment with to get the best results
        reward = float(-np.linalg.norm(pipette_position - self.goal_position)) # we can use the L2 norm to calculate the distance between the pipette position and the goal position
        
        # next we need to check if the if the task has been completed and if the episode should be terminated
        # To do this we need to calculate the distance between the pipette position and the goal position and if it is below a certain threshold, we will consider the task complete. 
        # What is a reasonable threshold? Think about the size of the pipette tip and the size of the plants.
        distance = np.linalg.norm(pipette_position - self.goal_position)
        if distance < 0.001:
            terminated = True
            # we can also give the agent a positive reward for completing the task
            reward = float(100)
        else:
            terminated = False

        # next we need to check if the episode should be truncated, we can check if the current number of steps is greater than the maximum number of steps
        if self.steps >= self.max_steps:
            truncated = True
        else:
            truncated = False

        info = {} # we don't need to return any additional information

        # increment the number of steps
        self.steps += 1

        return observation, reward, terminated, truncated, info

    def render(self):
        pass
    
    def close(self):
        self.sim.close()
        