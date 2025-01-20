import os
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from wandb.integration.sb3 import WandbCallback
from ot2_environment import OT2Env
from clearml import Task
import argparse
import wandb


# Custom callback to log rollout metrics
class RolloutLoggerCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(RolloutLoggerCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log rollout rewards and episode lengths
        if len(self.locals.get("infos", [])) > 0:
            for info in self.locals["infos"]:
                if "episode" in info:
                    wandb.log({
                        "rollout/episode_reward": info["episode"]["r"],
                        "rollout/episode_length": info["episode"]["l"],
                    })
        return True


# Load the API key for wandb
os.environ['WANDB_API_KEY'] = '4dc86acf289861c548514bbebcf1e5ae3cb63a5d'

# Initialize wandb and ClearML task
run = wandb.init(project="2024-Y2B-RoboSuite", sync_tensorboard=True)
task = Task.init(project_name='Mentor Group M/Group 2', task_name='Bless Us jezus')
task.set_base_docker('deanis/2023y2b-rl:latest')
task.execute_remotely(queue_name="default")

# Define environment
env = OT2Env()

# Create directories
save_path = f"models/{run.id}"
os.makedirs(save_path, exist_ok=True)

# Define training parameters
timesteps = 5000000
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
model = PPO(
    args.policy, env, verbose=1,
    learning_rate=args.learning_rate,
    batch_size=args.batch_size,
    n_steps=args.n_steps,
    n_epochs=args.n_epochs,
    gamma=args.gamma,
    clip_range=args.clip_range,
    vf_coef=args.value_coefficient,
    tensorboard_log=f"runs/{run.id}"
)

# Initialize callbacks
rollout_logger_callback = RolloutLoggerCallback()
wandb_callback = WandbCallback(
    model_save_freq=100000,
    model_save_path=f"models/{run.id}",
    verbose=2,
)
callbacks = [rollout_logger_callback, wandb_callback]

# Train the model with callbacks
model.learn(
    total_timesteps=timesteps,
    callback=callbacks,
    progress_bar=True,
    reset_num_timesteps=False,
    tb_log_name=f"runs/{run.id}"
)

# Save the final model
model.save(f"models/{run.id}/{timesteps}_baseline")
wandb.save(f"models/{run.id}/{timesteps}_baseline")
