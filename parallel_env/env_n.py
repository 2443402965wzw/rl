import time

from parallel_env.env_wrappers import SubprocVecEnv, DummyVecEnv
import argparse
from parallel_env.env_original import MyWrapper

def make_train_env(parallel_num):
    def get_env_fn(rank):
        def init_env():
            env = MyWrapper()
            return env
        return init_env
    if parallel_num == 1:
        return DummyVecEnv([get_env_fn(0)])
    else:
        return SubprocVecEnv([get_env_fn(i) for i in range(parallel_num)])



