from deep_rl_method.common.logger import configure
from deep_rl_method import A2C

discrete_envs = ["CartPole-v1", "Acrobot-v1", "MountainCar-v0"]
discrete_steps = 20_0000
continuous_envs = ["Pendulum-v1", "MountainCarContinuous-v0", "BipedalWalker-v3"]
continuous_steps = 100_0000
for env in continuous_envs:
    model = A2C("MlpPolicy", env, verbose=1)
    new_logger = configure(f"./logs/{env}/a2c/", ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    model.learn(total_timesteps=continuous_steps)