from deep_rl_method.common.logger import configure
from deep_rl_method import PPO

discrete_envs = ["CartPole-v1", "Acrobot-v1"]
discrete_steps = 20_0000
continuous_envs = ["Pendulum-v1", "MountainCarContinuous-v0", "BipedalWalker-v3"]
continuous_steps = 100_0000
for env in continuous_envs:
    model = PPO("MlpPolicy", env, verbose=1)
    new_logger = configure(f"./logs/{env}/ppo/", ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    model.learn(total_timesteps=continuous_steps)