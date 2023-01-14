from deep_rl_method.common.logger import configure
from deep_rl_method import DQN

discrete_envs = ["CartPole-v1", "Acrobot-v1", "MountainCar-v0"]
for env in discrete_envs:
    model = DQN("MlpPolicy", env, verbose=1)
    new_logger = configure(f"./logs/{env}/dqn/", ["stdout", "csv", "tensorboard"])
    model.set_logger(new_logger)
    model.learn(total_timesteps=20_0000)