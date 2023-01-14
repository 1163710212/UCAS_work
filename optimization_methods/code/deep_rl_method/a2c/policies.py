# This file is here just to define MlpPolicy/CnnPolicy
# that work for A2C
from deep_rl_method.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, MultiInputActorCriticPolicy

MlpPolicy = ActorCriticPolicy
CnnPolicy = ActorCriticCnnPolicy
MultiInputPolicy = MultiInputActorCriticPolicy
