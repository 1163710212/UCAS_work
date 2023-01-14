import os

from deep_rl_method.a2c import A2C
from deep_rl_method.common.utils import get_system_info
from deep_rl_method.ddpg import DDPG
from deep_rl_method.dqn import DQN
from deep_rl_method.her.her_replay_buffer import HerReplayBuffer
from deep_rl_method.ppo import PPO
from deep_rl_method.sac import SAC
from deep_rl_method.td3 import TD3

# Read version from file
version_file = os.path.join(os.path.dirname(__file__), "version.txt")
with open(version_file) as file_handler:
    __version__ = file_handler.read().strip()


def HER(*args, **kwargs):
    raise ImportError(
        "Since Stable Baselines 2.1.0, `HER` is now a replay buffer class `HerReplayBuffer`.\n "
        "Please check the documentation for more information: https://stable-baselines3.readthedocs.io/"
    )


__all__ = [
    "A2C",
    "DDPG",
    "DQN",
    "PPO",
    "SAC",
    "TD3",
    "HerReplayBuffer",
    "get_system_info",
]
