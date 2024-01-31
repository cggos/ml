import gymnasium as gym
from gymnasium import logger, spaces
import numpy as np
from typing import Optional, Tuple, Union


class Car2DEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 2
    }

    def __init__(self):
        self.xth = 0
        self.target_x = 0
        self.target_y = 0
        self.L = 10
        self.action_space = spaces.Discrete(5)  # 0, 1, 2，3，4: 不动，上下左右
        self.observation_space = spaces.Box(np.array([-self.L, -self.L]), np.array([self.L, self.L]))
        self.state = None
        self.counts = 0

    def step(self, action):
        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        x, y = self.state
        if action == 0:
            x = x
            y = y
        if action == 1:
            x = x
            y = y + 1
        if action == 2:
            x = x
            y = y - 1
        if action == 3:
            x = x - 1
            y = y
        if action == 4:
            x = x + 1
            y = y
        self.state = np.array([x, y])
        self.counts += 1

        terminated = (np.abs(x) + np.abs(y) <= 1) or (np.abs(x) + np.abs(y) >= 2 * self.L + 1)
        terminated = bool(terminated)

        if not terminated:
            reward = -0.1
        else:
            if np.abs(x) + np.abs(y) <= 1:
                reward = 10
            else:
                reward = -50
        return self.state, reward, terminated, False, {}

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            options: Optional[dict] = None,
    ):
        self.state = np.ceil(np.random.rand(2) * 2 * self.L) - self.L
        self.counts = 0
        return self.state

    def render(self, mode='human'):
        return None

    def close(self):
        return None


if __name__ == '__main__':
    env = Car2DEnv()
    observation = env.reset()

    for _ in range(10):
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)
        print(env.state)

    env.close()
