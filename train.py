import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
# Import the necessary libraries
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
import argparse
from clearml import Task
import wandb
import typing_extensions as TypeIs
import tensorflow

# After running the setup script we can upload this
from ot2_environment import OT2Env
# Load the API key for wandb
os.environ['WANDB_API_KEY'] = '4dc86acf289861c548514bbebcf1e5ae3cb63a5d'
run = wandb.init(project="2024-Y2B-RoboSuite", sync_tensorboard=True)


task = Task.init(project_name='Mentor Group M/Group 2/JoeyVA', task_name='Sample1')
task.set_base_docker('deanis/2023y2b-rl:latest')

# Set base docker image and queue
task.set_base_docker("deanis/2023y2b-rl:latest")
task.execute_remotely(queue_name="default")

#Define the model
env = OT2Env()

# Initialate wandb
run = wandb.init(project="task11",sync_tensorboard=True)
save_path = f"models/{run.id}"
os.makedirs(save_path, exist_ok=True)

# Set the amount of steps for the training
timesteps = 5000000

# Define the arguments
parser = argparse.ArgumentParser()
# Set the default parameters.
parser.add_argument("--learning_rate", type=float, default=0.0001)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_steps", type=int, default=2048)
parser.add_argument("--n_epochs", type=int, default=10)



# Extra parameters chosen because of the training
parser.add_argument("--gamma", type=float, default=0.98)
parser.add_argument("--policy", type=str, default="MlpPolicy")
parser.add_argument("--clip_range", type=float, default=0.15)
parser.add_argument("--value_coefficient", type=float, default=0.5)
args = parser.parse_args()

# Create the PPO Model
model = PPO(args.policy, env, verbose=1,
            learning_rate=args.learning_rate, 
            batch_size=args.batch_size, 
            n_steps=args.n_steps, 
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            clip_range=args.clip_range,
            vf_coef=args.value_coefficient,
            tensorboard_log=f"runs/{run.id}")

# Callback for wandb
wandb_callback = WandbCallback(
    model_save_freq=100000,
    model_save_path=f"models/{run.id}",
    verbose=2,)

# Train the model
model.learn(total_timesteps=timesteps, callback=wandb_callback, progress_bar=True, reset_num_timesteps=False,tb_log_name=f"runs/{run.id}")
# Save the model.
model.save(f"models/{run.id}/{timesteps}_baseline")
# Save the model to wandb
wandb.save(f"models/{run.id}/{timesteps}_baseline")


# python train.py --learning_rate 0.0001 --batch_size 32 --n_steps 2048 --n_epochs 10 --gamma 0.98 --policy MlpPolicy --clip_range 0.15 --value_coefficient 0.5
#python train.py --learning_rate 0.0003 --batch_size 64 --n_steps 2048 --n_epochs 10 --gamma 0.99 --policy MlpPolicy --clip_range 0.2 --value_coefficient 0.5
