import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.noise import NormalActionNoise

from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.callbacks import CheckpointCallback, EveryNTimesteps
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize, VecFrameStack, VecTransposeImage, stacked_observations, VecMonitor

import sys
sys.path.insert(0, "/home/asalvi/code_workspace/Husky_CS_SB3/train/HuskyCP-gym") #######Change module path here
import huskyCP_gym ###### and here

class GetEnvVar(BaseCallback):
    """
    Custom callback for plotting additional values in tensorboard.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)
        #self.training_env = env

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        value0 =  self.training_env.get_attr("self.lin_Vel")
        #value0 = self.locals_['self.lin_Vel']
        #value = self.get_attr('self.log_err_feat')
        #self.logger.record("random_value", value)
        print(value0)
        #return value

def make_env(env_id, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        port_no = str(23004 + 2*rank)
        print(port_no)
        seed = 1 + rank
        env = gym.make(env_id, port = port_no,seed = seed)
        #env.seed(seed + rank)
        return env
    #set_random_seed(seed)
    return _init
   
tmp_path = "/home/asalvi/code_workspace/tmp/RedRes2/2WE/Eval" # Path to save logs ######change this
# Create environment

env_id = "huskyCP_gym/HuskyRL-v0" ######## change this
num_cpu = 1  # Number of processes to use
env = SubprocVecEnv([make_env(env_id, i) for i in range(num_cpu)], start_method='fork')
env = VecMonitor(env, filename=tmp_path)
env = VecTransposeImage(env, skip=False)
env = VecNormalize(env, training=True, norm_obs=True, norm_reward=True, clip_obs=10.0, clip_reward=1000.0, gamma=0.99, epsilon=1e-08, norm_obs_keys=None)

#model_path = '/home/asalvi/Downloads/WP150.zip'
model_path = '/home/asalvi/code_workspace/tmp/RedRes2/2WsUnTr/2WsUnTr.zip' ######### update model path

model = PPO.load(model_path, env=env, print_system_info=True)

mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=25, deterministic = True)
obs = env.reset()
