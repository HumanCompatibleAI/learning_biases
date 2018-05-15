import argparse
import os
import subprocess as sp
import sys

from multiprocessing.pool import ThreadPool
from utils import concat_folder

INTERPRETER="python"

FLAGS = [
    ('agent', [
        'naive', 'optimal', 'sophisticated', 'myopic', 'underconfident',
        'overconfident'
    ]),
    ('algorithm', [
        'given_rewards', 'em_with_init', 'boltzmann_planner', 'vi_inference',
        'optimal_planner', 'joint_with_init', 'em_without_init',
        'joint_without_init'
    ]),
]

CONSTANT_FLAGS = [
    ('simple_mdp', False),
    ('imsize', 16),
    ('noise', 0.2),
    ('num_rewards', 7),
    ('num_human_trajectories', 8000),
    ('vin_regularizer_C', 1e-4),
    ('reward_regularizer_C', 0),
    ('lr', 0.01),
    ('reward_lr', 1.0),
    ('epochs', 20),
    ('reward_epochs', 50),
    ('k', 10),
    ('ch_h', 150),
    ('ch_p', 5),
    ('ch_q', 5),
    ('num_actions', 5),
    ('batchsize', 20),
    ('gamma', 0.95),
    ('num_iters', 50),
    ('max_delay', 10),
    ('hyperbolic_constant', 1.0),
    ('display_step', 1),
    ('log', False),
    ('verbosity', 1),
    ('plot_rewards', False),
    ('use_gpu', True),
    ('strict', False),
]

def get_algorithm_specific_flags(flags):
    [alg] = [val for name, val in flags if name == 'algorithm']
    flag_names = ['em_iterations', 'num_simulated', 'num_with_rewards', 'num_validation', 'model']
    if alg == 'given_rewards':
        flag_values = [0, 0, 7000, 2000, 'VIN']
    elif alg in ['boltzmann_planner', 'optimal_planner']:
        flag_values = [0, 5000, 0, 2000, 'VIN']
    elif alg == 'em_with_init':
        flag_values = [2, 5000, 0, 2000, 'VIN']
    elif alg == 'joint_with_init':
        flag_values = [0, 5000, 0, 2000, 'VIN']
    elif alg == 'em_without_init':
        flag_values = [2, 0, 0, 0, 'VIN']
    elif alg == 'joint_without_init':
        flag_values = [0, 0, 0, 0, 'VIN']
    elif alg == 'vi_inference':
        flag_values = [0, 0, 0, 0, 'VI']
    else:
        raise ValueError('Unknown algorithm {}'.format(alg))

    return list(zip(flag_names, flag_values))

def get_agent_specific_flags(flags):
    [agent] = [val for name, val in flags if name == 'agent']
    if agent == 'overconfident':
        return [('calibration_factor', 5)]
    elif agent == 'underconfident':
        return [('calibration_factor', 0.5)]
    elif agent in ['optimal', 'naive', 'sophisticated', 'myopic']:
        return [('calibration_factor', 1)]
    else:
        raise ValueError('Unknown agent {}'.format(agent))

def get_beta_flag(flags):
    [agent] = [val for name, val in flags if name == 'agent']
    if agent in ['optimal', 'overconfident']:
        return ('beta', 0.1)
    elif agent in ['naive', 'sophisticated', 'myopic', 'underconfident']:
        return ('beta', 1.0)
    else:
        raise ValueError('Unknown agent {}'.format(agent))

def flag_generator(flags):
    """Returns a generator that yields list of (flag, value) tuples."""
    if not flags:
        yield []
        return

    flag_name, flag_values = flags[0]
    for value in flag_values:
        for sublst in flag_generator(flags[1:]):
            yield [(flag_name, value)] + sublst


class CommandRunner(object):
    def __init__(self, pool_size):
        self.pool = ThreadPool(pool_size)

    def add_command_to_pool(self, command, error_file):
        command_str = ' '.join(command)
        try:
            proc = sp.Popen(command, stdout=sp.PIPE, stderr=sp.PIPE)
            out, err = proc.communicate()
            print('Ran command: {}'.format(command_str))
            print(out)
            with open(error_file, 'a') as errtxt:
                errtxt.write(command_str + '\n')
                errtxt.write(err)
            return True
        except Exception as e:
            print("Failed to run: {} because of exception {}".format(command_str))
            return False

    def run_command(self, interpreter, flags, dest):
        base_command = [interpreter, 'train.py', '--output_folder={}'.format(dest)]
        flag_strs = ['--{}={}'.format(name, val) for name, val in flags]
        command = base_command + flag_strs
        error_file = concat_folder(dest, 'errors.log')
        self.pool.apply_async(self.add_command_to_pool, (command, error_file))

    def close(self):
        self.pool.close()

    def join(self):
        self.pool.join()

def run_benchmarks(low, high, interpreter, flag_parameters, constant_flags, pool_size, dest):
    """
    :param interpreter: path to relevant python executable
    :param flags: dictionary of flags: [benchmark_values]
    """
    if not os.path.isdir(dest):
        os.mkdir(dest)

    runner = CommandRunner(pool_size)
    for start in range(low, high):
        seeds = range(10 * start, 10 * (start + 1))
        seed_flag = ('seeds', ','.join([str(seed) for seed in seeds]))
        for flags in flag_generator(flag_parameters):
            algorithm_flags = get_algorithm_specific_flags(flags)
            agent_flags = get_agent_specific_flags(flags)
            beta_flag = get_beta_flag(flags)
            all_flags = [seed_flag] + flags + constant_flags + algorithm_flags + agent_flags
            runner.run_command(interpreter, all_flags, dest)
            runner.run_command(interpreter, [beta_flag] + all_flags, dest)

    runner.close()
    runner.join()

    # Delete the generated gridworld data, since it is quite large
    for seed in range(10 * low, 10 * high):
        sp.call('rm datasets/*-seed-{}-*.npz'.format(seed), shell=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--low', required=True)
    parser.add_argument('--high', required=True)
    parser.add_argument('-f', '--folder', required=True)
    parser.add_argument('-s', '--pool_size', required=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    low, high, pool_size = map(int, (args.low, args.high, args.pool_size))
    run_benchmarks(low, high, INTERPRETER, FLAGS, CONSTANT_FLAGS, pool_size, args.folder)
