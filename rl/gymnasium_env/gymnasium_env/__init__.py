from gymnasium.envs.registration import register

register(
    id="gymnasium_env/Grid-v0",
    entry_point="gymnasium_env.envs:GridEnv",
    max_episode_steps=200,
    reward_threshold=100.0,
)

register(
    id="gymnasium_env/GridWorld-v0",
    entry_point="gymnasium_env.envs:GridWorldEnv",
)

