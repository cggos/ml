import gymnasium as gym
from gymnasium.wrappers import FlattenObservation

env = gym.make("CarRacing-v3")

print(env.observation_space.shape)

wrapped_env = FlattenObservation(env)

print(wrapped_env.observation_space.shape)
