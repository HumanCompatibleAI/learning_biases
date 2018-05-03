import argparse
import numpy as np
import os
import pickle
import scipy.stats

###################
# Data structures #
###################

class Experiment(object):
    def __init__(self, unique_id, flags, means_data, sterrs_data):
        self.unique_id = unique_id
        self.flags = flags
        self.means_data = means_data
        self.sterrs_data = sterrs_data

    def __str__(self):
        return 'Experiment: ' + str(self.flags)


#################
# Reading input #
#################

def concat_folder(folder, element):
    """folder and element are strings"""
    if folder[-1] == '/':
        return folder + element
    return folder + '/' + element

def get_flag_vals(folder):
    with open(concat_folder(folder, 'flags.pickle'), 'rb') as f:
        flags_dict = pickle.load(f)
        key = tuple(sorted(flags_dict.items()))
    return key, flags_dict

def load_experiment_run(filename):
    """Loads the data from <filename>, which has saved logs in .npz format.

    Returns a dictionary mapping keys to lists of numbers.
    """
    # TODO: Is this a dictionary?
    result = dict(np.load(filename).items())
    if not result: # TODO: Check for failures
        return None

    planner_costs = result['train_planner_costs']
    reward_costs = result['train_reward_costs']
    joint_costs = result['train_joint_costs']

    if len(reward_costs) > 0:
        result['em_iterations'] = np.arange(1, len(reward_costs) + 1)
        result['reward_iterations'] = np.arange(1, len(reward_costs[0]) + 1)
    if len(planner_costs) > 0:
        # If we have planner costs, we must also have reward costs.
        assert result['em_iterations'][-1] == len(planner_costs)
        result['planner_iterations'] = np.arange(1, len(planner_costs[0]) + 1)
    if len(joint_costs) > 0:
       result['joint_iterations'] = np.arange(1, len(joint_costs[0]) + 1)
    return result

def load_experiment(folder):
    """Loads the data from <folder>, specifically from flags.pickle and
    seeds-*.npz, and aggregates the data across seeds.

    Returns two things:
    - means_data: Dictionary mapping keys to lists of numbers (means).
    - sterr_data: Dictionary mapping keys to lists of numbers (std errors).
    """
    all_results = []
    for filename in os.listdir(folder):
        if not filename.startswith('seeds-') or not filename.endswith('.npz'):
            continue
        result = load_experiment_run(concat_folder(folder, filename))
        if result is not None:
            all_results.append(result)

    means_data, sterrs_data = {}, {}
    for key in all_results[0].keys():
        data = np.stack([result[key] for result in all_results], axis=0)
        means_data[key] = np.mean(data, axis=0)
        sterrs_data[key] = scipy.stats.sem(data, axis=0)
    return means_data, sterrs_data

def load_data(folder):
    """Loads all experiment data from <folder>.

    Returns a dictionary from keys of the form ((var, val), ...) to Experiment
    objects.
    """
    experiments = {}
    for sha_hash in os.listdir(folder):
        if len(sha_hash) != 56 or not set(sha_hash) <= set('0123456789abcdef'):
            continue
        subfolder = concat_folder(folder, sha_hash)
        if not os.path.isdir(subfolder):
            continue

        key, flags_dict = get_flag_vals(subfolder)
        means, sterrs = load_experiment(subfolder)
        assert key not in experiments, '{}: {} and {}'.format(key, sha_hash, experiments[key].unique_id)
        experiments[key] = Experiment(sha_hash, flags_dict, means, sterrs)

    print('Loaded {} experiments'.format(len(experiments.items())))
    return experiments


##############
# Processing #
##############

def simplify_keys(experiments):
    """Identifies experiment flags that are constant across the dataset and
    removes them from the keys, leaving shorter, simpler keys.

    experiments: Dictionary from keys of the form ((var, val), ...) to
        Experiment objects
    Returns two things:
      - new_experiments: Same type as experiments, but with smaller keys
      - controls: Dictionary of the form {var : val} containing the flags and
          their values that did not change over the experiments.
    """
    keys = list(experiments.keys())
    first_key = keys[0]

    indices_with_no_variation = []
    indices_with_variation = []
    for index in range(len(first_key)):
        if all(key[index] == first_key[index] for key in keys):
            indices_with_no_variation.append(index)
        else:
            indices_with_variation.append(index)

    def simple_key(key):
        return tuple((key[index] for index in indices_with_variation))

    new_experiments = {simple_key(k):v for k, v in experiments.items()}
    controls = dict([first_key[index] for index in indices_with_no_variation])
    return new_experiments, controls

def fix_special_cases(experiments):
    """Does postprocessing to handle any special cases (none currently).

    - experiments: Dictionary from keys of the form ((var, val), ...) to
          Experiment objects
    Returns: None (mutates the given experiments)
    """
    pass

def process_data(experiments):
    """Processes experiments into a more useful form for graphing.

    - experiments: Dictionary from keys of the form ((var, val), ...) to
          Experiment objects

    Returns three things:
    - experiments: Same type of object as the input experiments
    - changing_vars: List of strings, the variables that have more than one
          distinct value across Experiments
    - control_var_vals: Dictionary of the form {var : val} containing the
          flags and their values that did not change over the experiments.
    """
    experiments, control_var_vals = simplify_keys(experiments)
    fix_special_cases(experiments)
    changing_vars = [var for var, val in list(experiments.keys())[0]]
    return experiments, changing_vars, control_var_vals


###################
# Creating graphs #
###################

def get_matching_experiments(experiments, flags_to_match):
    """Returns a list of Experiments whose flag values match the bindings in
    flags_to_match.

    - experiments: Dictionary from keys of the form ((var, val), ...) to
          Experiment objects
    - flags_to_match: Tuple of the form ((var, val), ...) where var is a string
          and val is a string or number. The flag values to match.
    """
    check = lambda exp: all(exp.flags[k] == v for k, v in flags_to_match)
    return [exp for exp in experiments.values() if check(exp)]

def write_table(experiments, output_file):
    """Writes a table of final results with standard errors.

    - experiments: Dictionary from keys of the form ((var, val), ...)
          to Experiment objects
    """
    row_names = ['Optimal', 'Naive', 'Sophisticated', 'Myopic',
                 'Boltzmann-Optimal', 'Boltzmann-Naive',
                 'Boltzmann-Sophisticated', 'Boltzmann-Myopic']
    col_names = ['optimal_planner', 'boltzmann_planner', 'given_rewards',
                 'em_with_init', 'joint_with_init', 'em_without_init',
                 'joint_without_init', 'vi_inference']

    def get_row_col_names(exp):
        col = exp.flags['algorithm']
        if col == 'no_rewards': col = 'em_with_init'
        if col == 'joint_no_rewards': col = 'joint_without_init'
        agent, beta = exp.flags['agent'], exp.flags['beta']
        row = agent[0:1].upper() + agent[1:]
        if beta != None:
            row = 'Boltzmann-' + row
        return row, col

    results = [[(None, None)] * len(col_names) for _ in range(len(row_names))]
    for exp in experiments.values():
        mean = exp.means_data['Average %reward']
        sterr = exp.sterrs_data['Average %reward']
        row_name, col_name = get_row_col_names(exp)
        row, col = row_names.index(row_name), col_names.index(col_name)
        assert results[row][col] == (None, None)
        results[row][col] = [mean, sterr]

    def stringify(mean, sterr):
        if output_file.endswith('means.csv'):
            return 'N/A' if mean is None else '%.1f' % (100 * mean)
        elif output_file.endswith('sterrs.csv'):
            return 'N/A' if sterr is None else '%.1f' % (100 * sterr)
        else:
            return 'N/A' if mean is None else '"[%.1f, %.1f]"' % (100*(mean-sterr), 100*(mean+sterr))

    with open(output_file, 'w') as f:
        f.write('Agent,' + ','.join(col_names) + '\n')
        for row in range(len(row_names)):
            elem_strs = ','.join([stringify(*x) for x in results[row]])
            f.write(row_names[row] + ',' + elem_strs + '\n')


##########################
# Command Line Interface #
##########################

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', required=True)
    parser.add_argument('-o', '--output_file', required=True)
    parser.add_argument('-x', '--x_var')
    parser.add_argument('-d', '--dependent_var', action='append')
    parser.add_argument('-i', '--independent_var', action='append')
    parser.add_argument('-c', '--control_var_val', action='append', default=[])
    parser.add_argument('-e', '--experiment', action='append', default=[])
    return parser.parse_args()

def parse_kv_pairs(lst):
    result = [kv_pair.split('=') for kv_pair in lst]
    return [(k, maybe_num(v)) for k, v in result]

def maybe_num(x):
    """Converts string x to an int if possible, otherwise a float if possible,
    otherwise returns it unchanged."""
    try: return int(x)
    except ValueError:
        try: return float(x)
        except ValueError: return x

if __name__ == '__main__':
    args = parse_args()
    experiments = load_data(args.folder)
    experiments, all_vars, _ = process_data(experiments)
    controls = parse_kv_pairs(args.control_var_val)
    extra_experiments = [parse_kv_pairs(x.split(',')) for x in args.experiment]
    write_table(experiments, args.output_file)
    # graph_all(experiments, all_vars, args.x_var, args.dependent_var,
    #           args.independent_var, controls, extra_experiments, args.folder, args)
