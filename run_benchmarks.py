import subprocess as sp
import os
import sys
from utils import concat_folder

INTERPRETER="/home/ngundotra/.conda/envs/IRL/bin/python"

FLAGS = [
    ('agent', ['naive', 'optimal', 'sophisticated', 'myopic']),
    ('algorithm', [
        'given_rewards', 'no_rewards', 'boltzmann_planner',
        'vi_inference', 'joint_no_rewards', 'optimal_planner'
    ]),
]

CONSTANT_FLAGS = [
    ('simple_mdp', False),
    ('imsize', 16),
    ('num_rewards', 5),
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
    elif alg == 'no_rewards':
        flag_values = [2, 5000, 0, 2000, 'VIN']
    elif alg in ['boltzmann_planner', 'optimal_planner']:
        flag_values = [0, 5000, 0, 2000, 'VIN']
    elif alg == 'joint_no_rewards':
        flag_values = [0, 0, 0, 0, 'VIN']
    elif alg == 'vi_inference':
        flag_values = [0, 0, 0, 0, 'VI']
    else:
        raise ValueError('Unknown algorithm {}'.format(alg))

    return list(zip(flag_names, flag_values))

def get_beta_flag(flags):
    [agent] = [val for name, val in flags if name == 'agent']
    if agent == 'optimal':
        return ('beta', 0.1)
    elif agent in ['naive', 'sophisticated', 'myopic']:
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


def run_command(interpreter, flags, dest):
    base_command = [interpreter, 'train.py', '--output_folder={}'.format(dest)]
    flag_strs = ['--{}={}'.format(name, val) for name, val in flags]
    command = base_command + flag_strs
    command_str = ' '.join(command)
    error_file = concat_folder(dest, 'errors.log')
    print('Running {}'.format(command_str))
    try:
        with open(error_file, 'a') as errtxt:
            proc = sp.call(command, stderr=errtxt)
        return True
    except Exception as e:
        print("Failed to run: {} because of exception {}".format(command_str, e))
        return False

def run_benchmarks(low, high, interpreter, flag_parameters, constant_flags, dest):
    """
    :param interpreter: path to relevant python executable
    :param flags: dictionary of flags: [benchmark_values]
    """
    if not os.path.isdir(dest):
        os.mkdir(dest)

    base_command = [interpreter, 'train.py', '--output_folder={}'.format(dest)]
    success, count_calls = 0, 0
    for start in range(low, high):
        seeds = range(10 * start, 10 * (start + 1))
        seed_flag = ('seeds', ','.join([str(seed) for seed in seeds]))
        for flags in flag_generator(flag_parameters):
            algorithm_flags = get_algorithm_specific_flags(flags)
            beta_flag = get_beta_flag(flags)
            all_flags = [seed_flag] + flags + constant_flags + algorithm_flags
            if run_command(interpreter, all_flags, dest):
                success += 1
            if run_command(interpreter, [beta_flag] + all_flags, dest):
                success += 1
            count_calls += 2

        # Delete the generated gridworld data, since it is quite large
        for seed in range(10*low, 10*high):
            sp.call('rm datasets/*-seed-{}-*.npz'.format(seed), shell=True)

    print("{} out of {} calls ran (but may have thrown an exception)".format(success, count_calls))


if __name__ == '__main__':
    _, low, high = sys.argv
    low, high = int(low), int(high)
    run_benchmarks(low, high, INTERPRETER, FLAGS, CONSTANT_FLAGS, 'benchmark_data/')
