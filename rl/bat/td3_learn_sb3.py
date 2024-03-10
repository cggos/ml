import argparse
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE" 
os.chdir(os.path.dirname(__file__))
import sys
sys.path.append('bat_env')

import tkinter as tk
from tkinter import filedialog

import bat_env
from bat_env import regi
import pprint
import gymnasium as gym
import numpy as np
import torch
print(torch.__version__)
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
from torch.utils.tensorboard import SummaryWriter

from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import (
    NormalActionNoise,
    OrnsteinUhlenbeckActionNoise,
    VectorizedActionNoise,
)
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.buffers import ReplayBuffer,DictReplayBuffer
import pickle
import time
from collections import deque
import pybamm

class Color:
    RED = '\033[91m'
    GREEN = '\033[92m'
    END = '\033[0m'

class SaveOnStepCallback(BaseCallback):
    def __init__(self, check_freq: int, save_path: str, initial_step: int = 0, verbose=0, env=None):
        super(SaveOnStepCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.env = env
        self.initial_step = initial_step
        self.last_100_ep_ratio = deque([0]*100,maxlen=100)

        self.done_counter = 0
        self.t_count = 0
    def _on_step(self) -> bool:
        # print(self.training_env.env_method("get_solutions"))
        infos = self.locals.get("infos")
        if infos:
            for x, info in enumerate(infos):
                if info.get('terminated') or info.get('truncated'):

                    self.t_count += 1
                    print(f't = {self.t_count}')
                    print('___________________________________')

                    if self.t_count % 10 == 0:
                        print("creating gif")
                        solutions = self.env.envs[x].get_solutions()
                        if not solutions:
                            print("Warning: solutions list is empty!")
                            return True  # continue training
                        plot = pybamm.QuickPlot(solutions, output_variables=["Current [A]", "Voltage [V]"], labels=False)
                        os.makedirs(os.path.join(self.save_path, "plots"), exist_ok=True)
                        plot.savefig(os.path.join(self.save_path, "plots", f"plot_step_{self.n_calls + self.initial_step}.png"))
                        plot.create_gif(output_filename=os.path.join(self.save_path, "plots", f"plot_step_{self.n_calls + self.initial_step}.gif"))

                    if info.get('terminated'):
                        print(f"{Color.GREEN}terminated {info.get('step_count')} {info.get('reward')}{Color.END}")

                        self.last_100_ep_ratio.append(1)

                    elif info.get('truncated'):

                        print(f"{Color.RED}truncated {info.get('step_count')} {info.get('reward')}{Color.END}")
                        
                        self.last_100_ep_ratio.append(0)
                        # self.last_100_ep_capacity.append(info.get('cumulative_capacity'))
            terminated_percentage = sum(self.last_100_ep_ratio) / len(self.last_100_ep_ratio)
            # average_capacity = sum(self.last_100_ep_capacity) / len(self.last_100_ep_capacity)
            # self.logger.record("average_capacity", average_capacity)
            self.logger.record("terminated_percentage", terminated_percentage)
            self.logger.record("buffer_size", self.model.replay_buffer.size())

        if self.n_calls % self.check_freq == 0:
            self.model.save(os.path.join(self.save_path, f"td3_step_{self.n_calls + self.initial_step}"))
            print(f"Saved model at step {self.n_calls + self.initial_step}")
        return True  # 返回True以继续训练

def select_file():
    root = tk.Tk()
    root.withdraw()
    root.call('wm', 'attributes', '.', '-topmost', True)
    file_path = filedialog.askopenfilename(filetypes=[("Model files", "*.zip")])
    return file_path


def main(timesteps=100, n_env=1, save_path=None, buffer_path=None):
    n_actions = 1

    # base_noise = OrnsteinUhlenbeckActionNoise(
    #     mean=np.zeros(n_actions), sigma=0.25 * np.ones(n_actions), theta=0.15,
    #     dt = 0.01
    # )
    base_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.5 * np.ones(n_actions))
    vectorized_noise = VectorizedActionNoise(base_noise, n_envs=n_env)
    current_path = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(
        current_path, "td3_tensorboard_test", "td3_multi"
    )
    os.makedirs(save_path, exist_ok=True)
    print(f"tb files saved under {repr(save_path)}")
    

    start_time = time.time()
    seed_v = int(time.time())
    
    vec_env = make_vec_env("EcmEnv_v0", n_envs=n_env, seed=seed_v , vec_env_cls=SubprocVecEnv , env_kwargs={"render_mode": "rgb_array"})
    eval_env = make_vec_env("EcmEnv_v0", n_envs=1, seed=seed_v , vec_env_cls=SubprocVecEnv)
    save_callback = SaveOnStepCallback(
        check_freq=100, save_path=os.path.join(current_path, "td3_subproc"), verbose=1, env = vec_env
    )

    eval_callback = EvalCallback(eval_env,callback_on_new_best=save_callback, 
                                 n_eval_episodes=2, eval_freq=100, deterministic=True, verbose=1,
                                 render=True)
    callbacklist = [save_callback, eval_callback]
    # net_arch = {
    # "pi": [1024,2048,2048,256],  # Actor的网络结构
    # "qf": [1024,2048,2048,256]  # Critic的网络结构
    # }
    net_arch = {
        "pi": [400,300],  # Actor的网络结构
        "qf": [400,300]  # Critic的网络结构
    }
    model = TD3(
        "MultiInputPolicy", 
        vec_env,
        action_noise=vectorized_noise,
        verbose=1,
        device="cuda",
        batch_size=256,
        train_freq=(20,'step'),
        tensorboard_log=save_path,
        replay_buffer_class=DictReplayBuffer,
        buffer_size=10000*n_env,
        learning_rate=2.32e-03,
        tau=0.001,
        policy_kwargs={"net_arch": net_arch,},
        gamma=0.999,
            )

    if buffer_path is not None:
        model.load_replay_buffer(buffer_path)
        # print(buffer.size())

    initial_step = 0

    try:
        # Continue training
        # print(model.get_parameters())
        model.learn(total_timesteps=timesteps, log_interval=5, progress_bar=False, callback=callbacklist, reset_num_timesteps=True)
    except KeyboardInterrupt:
        print("Interrupted by user. Saving the model...")
        current_step = int(model.num_timesteps/model.n_envs) + initial_step
        model.save(os.path.join(os.path.join(current_path, "td3_subproc"), f"td3_step_{current_step}_interrupted"))
        print(f"Model saved at step {current_step} due to interruption.")
        with open("./codes/buffer.pkl", "wb") as f:
            pickle.dump(model.replay_buffer, f)

    finally:
        end_time = time.time() - start_time
        print("n_env", n_env, "time", end_time)
        model.save(current_path+"./td3_subproc")

if __name__ == '__main__':
    # with open("./codes/buffer.pkl","rb") as f:
    #     buffer = pickle.load(f)
    #     loaded_buffer_size = buffer.size()
    #     print(loaded_buffer_size)
    # buffer_path = "./codes/buffer_6.pkl"
    main(timesteps=1000, n_env=1, save_path=None, buffer_path=None)
