import argparse
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
import scipy
import seaborn as sns

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

def write_table(experiments):
    """Writes a table of final results with standard errors.

    - experiments: Dictionary from keys of the form ((var, val), ...)
          to Experiment objects
    """
    row_names = ['Optimal', 'Naive', 'Sophisticated', 'Myopic',
                 'Boltzmann-Optimal', 'Boltzmann-Naive',
                 'Boltzmann-Sophisticated', 'Boltzmann-Myopic']
    col_names = ['Optimal VIN', 'Boltzmann VIN', 'VIN with rewards',
                 'Coordinate ascent', 'Joint training', 'Differentiable VI']

    def get_row_col_names(exp):
        alg = exp.flags['algorithm']
        if alg == 'given_rewards':
            col = 'VIN with rewards'
        elif alg == 'boltzmann_planner':
            col = 'Boltzmann VIN'
        elif alg == 'optimal_planner':
            col = 'Optimal VIN'
        elif alg == 'no_rewards':
            col = 'Coordinate ascent'
        elif alg == 'joint_no_rewards':
            col = 'Joint training'
        elif alg == 'vi_inference':
            col = 'Differentiable VI'
        else:
            raise ValueError('Unknown algorithm')

        agent, beta = exp.flags['agent'], exp.flags['beta']
        row = agent[0:1].upper() + agent[1:]
        if beta != None:
            row = 'Boltzmann-' + row
        return row, col

    results = [['N/A'] * len(col_names) for _ in range(len(row_names))]
    for exp in experiments.values():
        mean = exp.means_data['Average %reward']
        sterr = exp.sterrs_data['Average %reward']
        row_name, col_name = get_row_col_names(exp)
        row, col = row_names.index(row_name), col_names.index(col_name)
        assert results[row][col] == 'N/A'
        results[row][col] = [mean-sterr, mean+sterr]

    def stringify(low, high):
        return '%.1f' % (100 * (low + high) / 2.0)
        #return '"[%.1f, %.1f]"' % (100*low, 100*high)

    lines = []
    lines.append(',' + ','.join(col_names))
    for row in range(len(row_names)):
        lines.append(row_names[row] + ',' + ','.join([stringify(*x) for x in results[row]]))
    print('\n'.join(lines))

def graph_all(experiments, all_vars, x_var, dependent_vars, independent_vars,
              controls, extra_experiment_flags, folder, args):
    """Graphs data and saves them.

    Each graph generated plots the dependent_vars against x_var for all
    valuations of independent_vars, with the control variables set to the values
    specified in controls. For every valuation of variables not in x_var,
    dependent_vars, independent_vars, or controls, a separate graph is
    generated and saved in folder.

    - experiments: Dictionary from keys of the form ((var, val), ...) to
          Experiment objects
    - all_vars: List of strings, all the variables that have some variation
    - x_var: Variable that provides the data for the x-axis
    - dependent_vars: List of strings, variables to plot on the y-axis
    - independent_vars: List of strings, experimental conditions to plot on the
          same graph
    - controls: Tuple of the form ((var, val), ...) where var is a string and
          val is a string or number. The values of control variables.
    - folder: Graphs are saved to graph/<folder>/
    """
    control_vars = [var for var, val in controls]
    vars_so_far = [x_var] + dependent_vars + independent_vars + control_vars
    remaining_vars = list(set(all_vars) - set(vars_so_far))
    graphs_data = {}

    extra_experiments = []
    for exp_flags in extra_experiment_flags:
        identified_experiments = get_matching_experiments(experiments, exp_flags)
        assert len(identified_experiments) == 1
        extra_experiments.append(identified_experiments[0])

    for exp in get_matching_experiments(experiments, controls):
        key = ','.join(['{0}={1}'.format(k, exp.flags[k]) for k in remaining_vars])
        if key not in graphs_data:
            graphs_data[key] = extra_experiments[:]  # Make a copy of the list
        graphs_data[key].append(exp)

    for key, exps in graphs_data.items():
        graph(exps, x_var, dependent_vars, independent_vars, controls, key, folder)


def graph(exps, x_var, dependent_vars, independent_vars, controls,
          other_vals, folder):
    """Creates and saves a single graph.

    Arguments are almost the same as for graph_all.
    - other_vals: String of the form "{var}={val},..." specifying values of
          variables not in x_var, dependent_vars, independent_vars, or
          controls.
    """
    # Whole figure layout setting
    set_style()
    assert len(dependent_vars) == 1
    y_var = dependent_vars[0]
    fig, ax = plt.subplots()
    sns.set_context(rc={'lines.markeredgewidth': 1.0})   # Thickness or error bars
    capsize = 0.    # length of horizontal line on error bars
    spacing = 100.0

    # Draw all lines and labels
    for experiment in exps:
        flags = experiment.flags
        means, sterrs = experiment.means_data, experiment.sterrs_data
        var = ', '.join([str(flags[k]) for k in independent_vars])
        label = var_to_label(var)   # name in legend
        x_data = np.array(means[x_var]) + 1
        color = var_to_color(var)
        # TODO: The [0] hardcoding here is bad, fix it somehow. It currently
        # is used to select the first instance that train_planner is called,
        # so that we get the training curves for the one call to
        # train_planner.
        print(x_data)
        print(means[y_var][0])
        ax.errorbar(x_data, means[y_var][0], yerr=sterrs[y_var][0], color=color,
                    capsize=capsize, capthick=1, label=label)#,
        # marker='o', markerfacecolor='white', markeredgecolor=color,
        # markersize=4)

        ax.set_xlim([0,21])
        ax.set_ylim(-0.2)

        # Set ylabel
        ax.set_ylabel(var_to_label(y_var), fontsize=15)

        # Set title
        title = 'Data for {0}'.format(', '.join(independent_vars))
        ax.set_title(title, fontsize=16, fontweight='normal')


    'Make legend'
    plt.sca(ax)
    # plt.legend(fontsize=12)


    'Change global layout'
    sns.despine(fig)    # Removes top and right graph edges
    # plt.suptitle('Number of queries asked', y=0.02, fontsize=16)
    # fig.suptitle('Bandits', y=0.98, fontsize=18)
    # plt.tight_layout(w_pad=0.02, rect=[0, 0.03, 1, 0.95])  # w_pad adds horizontal space between graphs
    # plt.subplots_adjust(top=1, wspace=0.35)     # adds space at the top or bottom
    # plt.subplots_adjust(bottom=.2)
    # fig.set_figwidth(15)     # Can be adjusted by resizing window
    # fig.set_figheight(5)

    'Save file'
    subtitle = ','.join(['{0}={1}'.format(k, v) for k, v in controls])
    subtitle = '{0},{1}'.format(subtitle, other_vals).strip(',')
    folder = concat_folder('graph', folder)
    filename = '{0}-vs-{1}-for-{2}-with-{3}.png'.format(
        ','.join(dependent_vars), x_var, ','.join(independent_vars), subtitle)
    if not os.path.exists(folder):
        os.mkdir(folder)
    plt.savefig(concat_folder(folder, filename))
    plt.show()


def set_style():
    mpl.rcParams['text.usetex'] = True
    mpl.rc('font', family='serif', serif=['Palatino'])  # Makes font thinner

    sns.set(font='serif', font_scale=1.4)   # Change font size of (sub) title and legend. Serif seems to have no effect.

    # Make the background a dark grid, and specify the
    # specific font family
    sns.set_style("white", {     # Font settings have no effect
        "font.family": "serif",
        "font.weight": "normal",
        "font.serif": ["Times", "Palatino", "serif"]})
        # 'axes.facecolor': 'darkgrid'})
        # 'lines.markeredgewidth': 1})


def var_to_label(var):
    return str(var).replace('_', '\_')

def var_to_color(var):
    colors = ['darkorange', 'lightblue', 'crimson', 'grey']
    return colors[0]  # Placeholder


##########################
# Command Line Interface #
##########################

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', required=True)
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
    write_table(experiments)
    # graph_all(experiments, all_vars, args.x_var, args.dependent_var,
    #           args.independent_var, controls, extra_experiments, args.folder, args)
