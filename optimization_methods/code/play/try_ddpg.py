from deep_rl_method.common.logger import configure
from deep_rl_method import DDPG

continuous_envs = ["Pendulum-v1", "MountainCarContinuous-v0", "BipedalWalker-v3"]
for env in continuous_envs:
    model = DDPG("MlpPolicy", env, verbose=1)
    new_logger = configure(f"./logs/{env}/ddpg/", ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    model.learn(total_timesteps=100_0000)