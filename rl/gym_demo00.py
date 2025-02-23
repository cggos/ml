import gymnasium as gym

# from gym import envs

# print(envs.registry.keys())
# print(envs.registry.values())

# env = gym.make("LunarLander-v2", render_mode="human")
env = gym.make("CarRacing-v2", render_mode="human")
# env = gym.make("CartPole-v1", render_mode="human")

print(env.action_space)
print(env.observation_space)

observation, info = env.reset()

timesteps = 1000

for _ in range(timesteps):
    # print(observation)
    action = (
        env.action_space.sample()
    )  # agent policy that uses the observation and info
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        print("{} timesteps".format(_ + 1))
        observation, info = env.reset()

env.close()
