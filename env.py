import gym
from matplotlib import pyplot as plt
from gym import spaces
import numpy as np
from parallel_env.multi_discrete import MultiDiscrete

# 定义环境
class MyWrapper(gym.Wrapper):
    def __init__(self):
        # env = gym.make('CartPole-v1', render_mode='rgb_array')
        env = gym.make('CartPole-v1')
        super().__init__(env)
        self.env = env
        self.step_n = 0
        self.inference_action_space = [MultiDiscrete([[0, 1], [0, 1], [0, 6], [0, 5]])]
        self.share_observation_space = [spaces.Box(
            low=-np.inf, high=+np.inf, shape=(1,), dtype=np.float32) for _ in range(1)]

    def reset(self):
        state = self.env.reset()
        # print("run to reset env!")
        self.step_n = 0
        # return state, None
        return state

    def step(self, action):
        state, reward, terminated, info = self.env.step(action)
        # self.env.render(mode='rgb_array')
        done = terminated
        self.step_n += 1
        if self.step_n >= 200:
            done = True
        return state, reward, done, info



