
import json
import select
import time
import logging
import os
import threading


from typing import Callable

import aicrowd_helper
import gym
import minerl
import abc
import numpy as np

import coloredlogs
coloredlogs.install(logging.DEBUG)

# our dependencies
import joblib

import sys
sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir, 'mod')))
from dqn_family import get_agent
from env_wrappers import wrap_env

GPU = -1

ARCH = 'distributed_dueling'
NOISY_NET_SIGMA = 0.5
FINAL_EPSILON = 0.01
FINAL_EXPLORATION_FRAMES = 10 ** 6
LR = 0.0000625
ADAM_EPS = 0.00015
PRIORITIZED = True
UPDATE_INTERVAL = 4
REPLAY_CAPACITY = 300000
NUM_STEP_RETURN = 10
AGENT_TYPE = 'CategoricalDoubleDQN'
GAMMA = 0.99
REPLAY_START_SIZE = 5000
TARGET_UPDATE_INTERVAL = 10000
CLIP_DELTA = True
BATCH_ACCUMULATOR = 'mean'
FRAME_SKIP = 4
GRAY_SCALE = False
FRAME_STACK = 4
RANDOMIZE_ACTION = NOISY_NET_SIGMA is None
EVAL_EPSILON = 0.001

maximum_frames = 8000000
STEPS = maximum_frames // FRAME_SKIP


# All the evaluations will be evaluated on MineRLObtainDiamondVectorObf-v0 environment
MINERL_GYM_ENV = os.getenv('MINERL_GYM_ENV', 'MineRLObtainDiamondVectorObf-v0')
MINERL_MAX_EVALUATION_EPISODES = int(os.getenv('MINERL_MAX_EVALUATION_EPISODES', 5))

# Parallel testing/inference, **you can override** below value based on compute
# requirements, etc to save OOM in this phase.
EVALUATION_THREAD_COUNT = int(os.getenv('EPISODES_EVALUATION_THREAD_COUNT', 1))

class EpisodeDone(Exception):
    pass

class Episode(gym.Env):
    """A class for a single episode.
    """
    def __init__(self, env):
        self.env = env
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self._done = False

    def reset(self):
        if not self._done:
            return self.env.reset()

    def step(self, action):
        s,r,d,i = self.env.step(action)
        if d:
            self._done = True
            raise EpisodeDone()
        else:
            return s,r,d,i



# DO NOT CHANGE THIS CLASS, THIS IS THE BASE CLASS FOR YOUR AGENT.
class MineRLAgentBase(abc.ABC):
    """
    To compete in the competition, you are required to implement a
    SUBCLASS to this class.

    YOUR SUBMISSION WILL FAIL IF:
        * Rename this class
        * You do not implement a subclass to this class

    This class enables the evaluator to run your agent in parallel,
    so you should load your model only once in the 'load_agent' method.
    """

    @abc.abstractmethod
    def load_agent(self):
        """
        This method is called at the beginning of the evaluation.
        You should load your model and do any preprocessing here.
        THIS METHOD IS ONLY CALLED ONCE AT THE BEGINNING OF THE EVALUATION.
        DO NOT LOAD YOUR MODEL ANYWHERE ELSE.
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def run_agent_on_episode(self, single_episode_env : Episode):
        """This method runs your agent on a SINGLE episode.

        You should just implement the standard environment interaction loop here:
            obs  = env.reset()
            while not done:
                env.step(self.agent.act(obs))
                ...

        NOTE: This method will be called in PARALLEL during evaluation.
            So, only store state in LOCAL variables.
            For example, if using an LSTM, don't store the hidden state in the class
            but as a local variable to the method.

        Args:
            env (gym.Env): The env your agent should interact with.
        """
        raise NotImplementedError()


#######################
# YOUR CODE GOES HERE #
#######################
class MineRLRainbowBaselineAgent(MineRLAgentBase):
    def __init__(self, env):
        self.env = env

    def load_agent(self):
        self.agent = get_agent(
            n_actions=self.env.action_space.n, arch=ARCH, n_input_channels=self.env.observation_space.shape[0],
            noisy_net_sigma=NOISY_NET_SIGMA, final_epsilon=FINAL_EPSILON,
            final_exploration_frames=FINAL_EXPLORATION_FRAMES, explorer_sample_func=self.env.action_space.sample,
            lr=LR, adam_eps=ADAM_EPS,
            prioritized=PRIORITIZED, steps=STEPS, update_interval=UPDATE_INTERVAL,
            replay_capacity=REPLAY_CAPACITY, num_step_return=NUM_STEP_RETURN,
            agent_type=AGENT_TYPE, gpu=GPU, gamma=GAMMA, replay_start_size=REPLAY_START_SIZE,
            target_update_interval=TARGET_UPDATE_INTERVAL, clip_delta=CLIP_DELTA,
            batch_accumulator=BATCH_ACCUMULATOR,
        )

        self.agent.load(os.path.abspath(os.path.join(__file__, os.pardir, 'train')))

    def run_agent_on_episode(self, single_episode_env: Episode):
        with self.agent.eval_mode():
            obs = single_episode_env.reset()
            while True:
                a = self.agent.act(obs)
                obs, r, done, info = single_episode_env.step(a)


#####################################################################
# IMPORTANT: SET THIS VARIABLE WITH THE AGENT CLASS YOU ARE USING   #
######################################################################
AGENT_TO_TEST = MineRLRainbowBaselineAgent # MineRLMatrixAgent, MineRLRandomAgent, YourAgentHere



####################
# EVALUATION CODE  #
####################
def main():
    # agent = AGENT_TO_TEST()
    # assert isinstance(agent, MineRLAgentBase)
    # agent.load_agent()
    #
    # assert MINERL_MAX_EVALUATION_EPISODES > 0
    # assert EVALUATION_THREAD_COUNT > 0
    #
    # # Create the parallel envs (sequentially to prevent issues!)
    # envs = [gym.make(MINERL_GYM_ENV) for _ in range(EVALUATION_THREAD_COUNT)]
    # episodes_per_thread = [MINERL_MAX_EVALUATION_EPISODES // EVALUATION_THREAD_COUNT for _ in range(EVALUATION_THREAD_COUNT)]
    # episodes_per_thread[-1] += MINERL_MAX_EVALUATION_EPISODES - EVALUATION_THREAD_COUNT *(MINERL_MAX_EVALUATION_EPISODES // EVALUATION_THREAD_COUNT)
    # # A simple funciton to evaluate on episodes!
    # def evaluate(i, env):
    #     print("[{}] Starting evaluator.".format(i))
    #     for i in range(episodes_per_thread[i]):
    #         try:
    #             agent.run_agent_on_episode(Episode(env))
    #         except EpisodeDone:
    #             print("[{}] Episode complete".format(i))
    #             pass
    #
    # evaluator_threads = [threading.Thread(target=evaluate, args=(i, envs[i])) for i in range(EVALUATION_THREAD_COUNT)]
    # for thread in evaluator_threads:
    #     thread.start()
    #
    # # wait fo the evaluation to finish
    # for thread in evaluator_threads:
    #     thread.join()

    assert MINERL_MAX_EVALUATION_EPISODES > 0
    assert EVALUATION_THREAD_COUNT == 1

    kmeans = joblib.load(os.path.abspath(os.path.join(__file__, os.pardir, 'train', 'kmeans.joblib')))

    core_env = gym.make(MINERL_GYM_ENV)
    env = wrap_env(
        env=core_env, test=True, monitor=False, outdir=None,
        frame_skip=FRAME_SKIP, gray_scale=GRAY_SCALE, frame_stack=FRAME_STACK,
        randomize_action=RANDOMIZE_ACTION, eval_epsilon=EVAL_EPSILON,
        action_choices=kmeans.cluster_centers_,
    )

    agent = AGENT_TO_TEST(env)
    assert isinstance(agent, MineRLAgentBase)
    agent.load_agent()

    for i in range(MINERL_MAX_EVALUATION_EPISODES):
        print("[{}] Starting evaluator.".format(i))
        try:
            agent.run_agent_on_episode(Episode(env))
        except EpisodeDone:
            print("[{}] Episode complete".format(i))
            pass


if __name__ == "__main__":
    main()
