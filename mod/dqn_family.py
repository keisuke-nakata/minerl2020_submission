import os
import logging
import argparse

import numpy as np
import torch
import minerl  # noqa: register MineRL envs as Gym envs.
import gym

import pfrl


# local modules
import sys
sys.path.append(os.path.abspath(os.path.join(__file__, os.pardir)))
import utils
from env_wrappers import wrap_env
from q_functions import parse_arch
from cached_kmeans import cached_kmeans

logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()

    env_choices = [
        # basic envs
        'MineRLTreechop-v0',
        'MineRLNavigate-v0', 'MineRLNavigateDense-v0', 'MineRLNavigateExtreme-v0', 'MineRLNavigateExtremeDense-v0',
        'MineRLObtainIronPickaxe-v0', 'MineRLObtainIronPickaxeDense-v0',
        'MineRLObtainDiamond-v0', 'MineRLObtainDiamondDense-v0',
        # obfuscated envs
        'MineRLTreechopVectorObf-v0',
        'MineRLNavigateVectorObf-v0', 'MineRLNavigateExtremeVectorObf-v0',
        # MineRL data pipeline fails for these envs: https://github.com/minerllabs/minerl/issues/364
        # 'MineRLNavigateDenseVectorObf-v0', 'MineRLNavigateExtremeDenseVectorObf-v0',
        'MineRLObtainDiamondVectorObf-v0', 'MineRLObtainDiamondDenseVectorObf-v0',
        'MineRLObtainIronPickaxeVectorObf-v0', 'MineRLObtainIronPickaxeDenseVectorObf-v0',
        # for debugging
        'MineRLNavigateDenseFixed-v0', 'MineRLObtainTest-v0',
    ]
    parser.add_argument('--env', type=str, choices=env_choices, required=True,
                        help='MineRL environment identifier.')

    # meta settings
    parser.add_argument('--outdir', type=str, default='results',
                        help='Directory path to save output files. If it does not exist, it will be created.')
    parser.add_argument('--seed', type=int, default=0, help='Random seed [0, 2 ** 31)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU to use, set to -1 if no GPU.')
    parser.add_argument('--demo', action='store_true', default=False)
    parser.add_argument('--load', type=str, default=None)
    parser.add_argument('--logging-level', type=int, default=20, help='Logging level. 10:DEBUG, 20:INFO etc.')
    parser.add_argument('--eval-n-runs', type=int, default=3)
    parser.add_argument('--monitor', action='store_true', default=False,
                        help='Monitor env. Videos and additional information are saved as output files when evaluation.')

    # training scheme (agent)
    parser.add_argument('--agent', type=str, default='DQN', choices=['DQN', 'DoubleDQN', 'PAL', 'CategoricalDoubleDQN'])

    # network architecture
    parser.add_argument('--arch', type=str, default='dueling', choices=['dueling', 'distributed_dueling'],
                        help='Network architecture to use.')

    # update rule settings
    parser.add_argument('--update-interval', type=int, default=4, help='Frequency (in timesteps) of network updates.')
    parser.add_argument('--frame-skip', type=int, default=None, help='Number of frames skipped (None for disable).')
    parser.add_argument('--gamma', type=float, default=0.99, help='Discount rate.')
    parser.add_argument('--no-clip-delta', dest='clip_delta', action='store_false')
    parser.set_defaults(clip_delta=True)
    parser.add_argument('--num-step-return', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2.5e-4, help='Learning rate.')
    parser.add_argument('--adam-eps', type=float, default=1e-8, help='Epsilon for Adam.')
    parser.add_argument('--batch-accumulator', type=str, default='sum', choices=['sum', 'mean'], help='accumulator for batch loss.')

    # observation conversion related settings
    parser.add_argument('--gray-scale', action='store_true', default=False, help='Convert pov into gray scaled image.')
    parser.add_argument('--frame-stack', type=int, default=None, help='Number of frames stacked (None for disable).')

    # exploration related settings
    parser.add_argument('--final-exploration-frames', type=int, default=10 ** 6,
                        help='Timesteps after which we stop annealing exploration rate')
    parser.add_argument('--final-epsilon', type=float, default=0.01, help='Final value of epsilon during training.')
    parser.add_argument('--eval-epsilon', type=float, default=0.001, help='Exploration epsilon used during eval episodes.')
    parser.add_argument('--noisy-net-sigma', type=float, default=None,
                        help='NoisyNet explorer switch. This disables following options: '
                        '--final-exploration-frames, --final-epsilon, --eval-epsilon')

    # experience replay buffer related settings
    parser.add_argument('--replay-capacity', type=int, default=10 ** 6, help='Maximum capacity for replay buffer.')
    parser.add_argument('--replay-start-size', type=int, default=5 * 10 ** 4,
                        help='Minimum replay buffer size before performing gradient updates.')
    parser.add_argument('--prioritized', action='store_true', default=False, help='Use prioritized experience replay.')

    # target network related settings
    parser.add_argument('--target-update-interval', type=int, default=3 * 10 ** 4,
                        help='Frequency (in timesteps) at which the target network is updated.')

    # K-means related settings
    parser.add_argument('--kmeans-n-clusters', type=int, default=30, help='#clusters for K-means')

    args = parser.parse_args()

    args.outdir = pfrl.experiments.prepare_output_dir(args, args.outdir)

    log_format = '%(levelname)-8s - %(asctime)s - [%(name)s %(funcName)s %(lineno)d] %(message)s'
    logging.basicConfig(filename=os.path.join(args.outdir, 'log.txt'), format=log_format, level=args.logging_level)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(args.logging_level)
    console_handler.setFormatter(logging.Formatter(log_format))
    logging.getLogger('').addHandler(console_handler)  # add hander to the root logger

    logger.info('Output files will be saved in {}'.format(args.outdir))

    utils.log_versions()

    try:
        dqn_family(
            # meta setttings
            env_id=args.env,
            outdir=args.outdir,
            seed=args.seed,
            gpu=args.gpu,
            demo=args.demo,
            load=args.load,
            eval_n_runs=args.eval_n_runs,
            monitor=args.monitor,
            # hyper params
            agent_type=args.agent,
            arch=args.arch,
            update_interval=args.update_interval,
            frame_skip=args.frame_skip,
            gamma=args.gamma,
            clip_delta=args.clip_delta,
            num_step_return=args.num_step_return,
            lr=args.lr,
            adam_eps=args.adam_eps,
            batch_accumulator=args.batch_accumulator,
            gray_scale=args.gray_scale,
            frame_stack=args.frame_stack,
            final_exploration_frames=args.final_exploration_frames,
            final_epsilon=args.final_epsilon,
            eval_epsilon=args.eval_epsilon,
            noisy_net_sigma=args.noisy_net_sigma,
            replay_capacity=args.replay_capacity,
            replay_start_size=args.replay_start_size,
            prioritized=args.prioritized,
            target_update_interval=args.target_update_interval,
            kmeans_n_clusters=args.kmeans_n_clusters,
        )
    except:  # noqa
        logger.exception('execution failed.')
        raise


def dqn_family(
        # meta settings
        env_id,
        outdir,
        seed,
        gpu,
        demo,
        load,
        eval_n_runs,
        monitor,
        # hyper params
        agent_type,
        arch,
        update_interval,
        frame_skip,
        gamma,
        clip_delta,
        num_step_return,
        lr,
        adam_eps,
        batch_accumulator,
        gray_scale,
        frame_stack,
        final_exploration_frames,
        final_epsilon,
        eval_epsilon,
        noisy_net_sigma,
        replay_capacity,
        replay_start_size,
        prioritized,
        target_update_interval,
        kmeans_n_clusters,
):
    os.environ['MALMO_MINECRAFT_OUTPUT_LOGDIR'] = outdir

    # Set a random seed used in PFRL.
    pfrl.utils.set_random_seed(seed)

    # Set different random seeds for train and test envs.
    train_seed = seed  # noqa: never used in this script
    test_seed = 2 ** 31 - 1 - seed

    # K-Means
    kmeans = cached_kmeans(
        cache_dir=os.environ.get('KMEANS_CACHE'),
        env_id=env_id,
        n_clusters=kmeans_n_clusters,
        random_state=seed)

    # create & wrap env
    def wrap_env_partial(env, test):
        randomize_action = test and noisy_net_sigma is None
        wrapped_env = wrap_env(
            env=env, test=test,
            monitor=monitor, outdir=outdir,
            frame_skip=frame_skip,
            gray_scale=gray_scale, frame_stack=frame_stack,
            randomize_action=randomize_action, eval_epsilon=eval_epsilon,
            action_choices=kmeans.cluster_centers_)
        return wrapped_env
    logger.info('The first `gym.make(MineRL*)` may take several minutes. Be patient!')
    core_env = gym.make(env_id)
    # training env
    env = wrap_env_partial(env=core_env, test=False)
    # env.seed(int(train_seed))  # TODO: not supported yet
    # evaluation env
    eval_env = wrap_env_partial(env=core_env, test=True)
    # env.seed(int(test_seed))  # TODO: not supported yet (also requires `core_eval_env = gym.make(args.env)`)

    # calculate corresponding `steps` and `eval_interval` according to frameskip
    # 8,000,000 frames = 1333 episodes if we count an episode as 6000 frames,
    # 8,000,000 frames = 1000 episodes if we count an episode as 8000 frames.
    maximum_frames = 8000000
    if frame_skip is None:
        steps = maximum_frames
        eval_interval = 6000 * 100  # (approx.) every 100 episode (counts "1 episode = 6000 steps")
    else:
        steps = maximum_frames // frame_skip
        eval_interval = 6000 * 100 // frame_skip  # (approx.) every 100 episode (counts "1 episode = 6000 steps")

    agent = get_agent(
        n_actions=env.action_space.n, arch=arch, n_input_channels=env.observation_space.shape[0],
        noisy_net_sigma=noisy_net_sigma, final_epsilon=final_epsilon,
        final_exploration_frames=final_exploration_frames, explorer_sample_func=env.action_space.sample,
        lr=lr, adam_eps=adam_eps,
        prioritized=prioritized, steps=steps, update_interval=update_interval,
        replay_capacity=replay_capacity, num_step_return=num_step_return,
        agent_type=agent_type, gpu=gpu, gamma=gamma, replay_start_size=replay_start_size,
        target_update_interval=target_update_interval, clip_delta=clip_delta,
        batch_accumulator=batch_accumulator,
    )

    if load:
        agent.load(load)

    # experiment
    if demo:
        eval_stats = pfrl.experiments.eval_performance(env=eval_env, agent=agent, n_steps=None, n_episodes=eval_n_runs)
        logger.info('n_runs: {} mean: {} median: {} stdev {}'.format(
            eval_n_runs, eval_stats['mean'], eval_stats['median'], eval_stats['stdev']))
    else:
        pfrl.experiments.train_agent_with_evaluation(
            agent=agent, env=env, steps=steps,
            eval_n_steps=None, eval_n_episodes=eval_n_runs, eval_interval=eval_interval,
            outdir=outdir, eval_env=eval_env, save_best_so_far_agent=True,
        )

    env.close()
    eval_env.close()


def parse_agent(agent):
    return {'DQN': pfrl.agents.DQN,
            'DoubleDQN': pfrl.agents.DoubleDQN,
            'PAL': pfrl.agents.PAL,
            'CategoricalDoubleDQN': pfrl.agents.CategoricalDoubleDQN}[agent]


def get_agent(
        n_actions, arch, n_input_channels,
        noisy_net_sigma, final_epsilon, final_exploration_frames, explorer_sample_func,
        lr, adam_eps,
        prioritized, steps, update_interval, replay_capacity, num_step_return,
        agent_type, gpu, gamma, replay_start_size, target_update_interval, clip_delta, batch_accumulator
):
    # Q function
    q_func = parse_arch(arch, n_actions, n_input_channels=n_input_channels)

    # explorer
    if noisy_net_sigma is not None:
        pfrl.nn.to_factorized_noisy(q_func, sigma_scale=noisy_net_sigma)
        # Turn off explorer
        explorer = pfrl.explorers.Greedy()
    else:
        explorer = pfrl.explorers.LinearDecayEpsilonGreedy(
            1.0, final_epsilon, final_exploration_frames, explorer_sample_func)

    opt = torch.optim.Adam(q_func.parameters(), lr, eps=adam_eps)  # NOTE: mirrors DQN implementation in MineRL paper

    # Select a replay buffer to use
    if prioritized:
        # Anneal beta from beta0 to 1 throughout training
        betasteps = steps / update_interval
        rbuf = pfrl.replay_buffers.PrioritizedReplayBuffer(
            replay_capacity, alpha=0.5, beta0=0.4, betasteps=betasteps, num_steps=num_step_return)
    else:
        rbuf = pfrl.replay_buffers.ReplayBuffer(replay_capacity, num_step_return)

    # build agent
    def phi(x):
        # observation -> NN input
        return np.asarray(x)
    Agent = parse_agent(agent_type)
    agent = Agent(
        q_func, opt, rbuf, gpu=gpu, gamma=gamma, explorer=explorer, replay_start_size=replay_start_size,
        target_update_interval=target_update_interval, clip_delta=clip_delta, update_interval=update_interval,
        batch_accumulator=batch_accumulator, phi=phi)

    return agent


if __name__ == '__main__':
    main()
