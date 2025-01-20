import os
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, CallbackList, BaseCallback
from stable_baselines3.common.env_checker import check_env
import gymnasium as gym
import argparse
from clearml import Task, Logger
import wandb
from ot2_environment import OT2Env

# Set CUDA environment (disable GPU for now)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Load the API key for wandb
os.environ['WANDB_API_KEY'] = '4dc86acf289861c548514bbebcf1e5ae3cb63a5d'

# Initialize ClearML Task
task = Task.init(project_name='Mentor Group M/Group 2', task_name='Bless Us jezus')
task.set_base_docker('deanis/2023y2b-rl:latest')
task.execute_remotely(queue_name="default")

# Initialize wandb
run = wandb.init(project="2024-Y2B-RoboSuite", sync_tensorboard=True)

# Define environment
env = OT2Env()

# Create output directory for models
save_path = f"models/{run.id}"
os.makedirs(save_path, exist_ok=True)

# Training parameters
timesteps = 5000000

# Define the arguments
parser = argparse.ArgumentParser()
parser.add_argument("--learning_rate", type=float, default=0.0001)
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--n_steps", type=int, default=2048)
parser.add_argument("--n_epochs", type=int, default=10)
parser.add_argument("--gamma", type=float, default=0.98)
parser.add_argument("--policy", type=str, default="MlpPolicy")
parser.add_argument("--clip_range", type=float, default=0.15)
parser.add_argument("--value_coefficient", type=float, default=0.5)
args = parser.parse_args()

# Create the PPO model
model = PPO(args.policy, env, verbose=1,
            learning_rate=args.learning_rate,
            batch_size=args.batch_size,
            n_steps=args.n_steps,
            n_epochs=args.n_epochs,
            gamma=args.gamma,
            clip_range=args.clip_range,
            vf_coef=args.value_coefficient,
            tensorboard_log=f"runs/{run.id}")

# Define ClearML Callback for Rollout Chart
class ClearMLRolloutCallback(BaseCallback):
    def __init__(self, log_interval=1000, verbose=0):
        super(ClearMLRolloutCallback, self).__init__(verbose)
        self.log_interval = log_interval
        self.logger = Logger.current_logger()

    def _on_step(self) -> bool:
        if self.n_calls % self.log_interval == 0:
            rewards = self.locals.get('rewards', [])
            episode_rewards = sum(rewards)
            self.logger.report_scalar(
                title="Rollout Chart",
                series="Episode Reward",
                value=episode_rewards,
                iteration=self.num_timesteps
            )
        return True

# Combine callbacks
wandb_callback = WandbCallback(
    model_save_freq=100000,
    model_save_path=f"models/{run.id}",
    verbose=2,
)

clearml_callback = ClearMLRolloutCallback(log_interval=1000)

callback_list = CallbackList([wandb_callback, clearml_callback])

# Train the model
model.learn(total_timesteps=timesteps,
            callback=callback_list,
            progress_bar=True,
            reset_num_timesteps=False,
            tb_log_name=f"runs/{run.id}")

# Save the model
model.save(f"models/{run.id}/{timesteps}_baseline")
wandb.save(f"models/{run.id}/{timesteps}_baseline")
