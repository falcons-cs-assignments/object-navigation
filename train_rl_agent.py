
import gym
from gym_minigrid.wrappers import ImgObsWrapper
from mini_behavior.utils.wrappers import MiniBHFullyObsWrapper
from mini_behavior.register import register
import mini_behavior
from stable_baselines3 import PPO
import numpy as np
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn
import torch
import argparse
import wandb
from wandb.integration.sb3 import WandbCallback
import json
from object_nav import envs

import os

script_dir = os.path.dirname(__file__)


parser = argparse.ArgumentParser()
parser.add_argument("--task", required=False, help='name of task to train on')
parser.add_argument("--partial_obs", default=True)
parser.add_argument("--room_size", type=int, default=10)
parser.add_argument("--max_steps", type=int, default=1000)
parser.add_argument("--total_timesteps", type=int, default=8e3)
parser.add_argument("--dense_reward", action="store_true")
parser.add_argument("--policy_type", default="CnnPolicy")
parser.add_argument(
    "--auto_env",
    default=True,
    help='flag to procedurally generate floorplan'
)
# NEW
parser.add_argument(
    "--auto_env_config",
    help='Path to auto environment JSON file',
    default='./object_nav/envs/floorplans/one_room.json'
)
args = parser.parse_args()
partial_obs = args.partial_obs


class MinigridFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.Space, features_dim: int = 512, normalized_image: bool = False) -> None:
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 32, (2, 2)),
            nn.ReLU(),
            nn.Conv2d(32, 64, (2, 2)),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute shape by doing one forward pass
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]

        # Adding LSTM layer
        self.lstm_hidden_size = 128
        self.lstm = nn.LSTM(input_size=n_flatten, hidden_size=self.lstm_hidden_size, batch_first=True)

        self.linear = nn.Sequential(nn.Linear(self.lstm_hidden_size, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        cnn_features = self.cnn(observations)
        # Assuming observations are of shape (batch_size, seq_len, features)
        lstm_out, _ = self.lstm(cnn_features.unsqueeze(1))  # Add seq_len dimension
        lstm_out = lstm_out[:, -1, :]  # Take the last output from the LSTM
        return self.linear(lstm_out)

policy_kwargs = dict(
    features_extractor_class=MinigridFeaturesExtractor,
    features_extractor_kwargs=dict(features_dim=128),
)

# Env wrapping
env_name = "MiniGrid-NavigateToObj-16x16-N2-v0"

if args.dense_reward:
    assert args.task in ["PuttingAwayDishesAfterCleaning", "WashingPotsAndPans", "NavigateToObj"]
    kwargs["dense_reward"] = True

# wandb init
config = {
    "policy_type": args.policy_type,
    "total_timesteps": args.total_timesteps,
    "env_name": env_name,
}

print('init wandb')
run = wandb.init(
    entity="rl_mo",
    project=env_name,
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=False,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)

print('make env')
env = gym.make(env_name)
if not args.partial_obs:
    env = MiniBHFullyObsWrapper(env)
env = ImgObsWrapper(env)

print('begin training')
# Policy training
model = PPO(config["policy_type"], env, n_steps=500, policy_kwargs=policy_kwargs, verbose=1, tensorboard_log=f"./runs/{run.id}")
model.learn(config["total_timesteps"], callback=WandbCallback(model_save_path=f"models/{run.id}"))

if not partial_obs:
    model.save(f"models/ppo_cnn/{env_name}")
else:
    model.save(f"models/ppo_cnn_partial/{env_name}")

run.finish()
