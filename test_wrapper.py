import gymnasium as gym
import numpy as np
from ot2_env_wrapper import OT2Env

# Load your custom environment
env = OT2Env()

# Run a single episode
obs = env.reset()
done = False
step = 0

while not done:
    # Take a random action from the environment's action space
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)

    print(f"Step: {step + 1}, Action: {action}, Reward: {reward}, Observation: {obs}, Terminated: {terminated}, Truncated: {truncated}, Info: {info}")

    step += 1
    if terminated or truncated:
        print(f"Episode finished after {step} steps. Info: {info}")
        break
