import argparse
import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from math import floor
from matplotlib.gridspec import GridSpec

ALL_BIASES = [
    'Average', 'Optimal', 'Naive', 'Sophisticated', 'Myopic', 'Overconfident',
    'Underconfident', 'Boltzmann-Optimal', 'Boltzmann-Naive',
    'Boltzmann-Sophisticated', 'Boltzmann-Myopic', 'Boltzmann-Overconfident',
    'Boltzmann-Underconfident'
]
ALL_ALGORITHMS = [
    'optimal_planner', 'boltzmann_planner', 'given_rewards', 'em_with_init',
    'joint_with_init', 'em_without_init', 'joint_without_init', 'vi_inference'
]

def concat_folder(folder, element):
    """folder and element are strings"""
    if folder[-1] == '/':
        return folder + element
    return folder + '/' + element

def get_algorithm_name(alg):
    if alg == 'given_rewards':
        return 'Algorithm 1'
    elif alg == 'boltzmann_planner':
        return 'Boltzmann'
    elif alg == 'optimal_planner':
        return 'Optimal'
    elif alg == 'em_with_init' or alg == 'no_rewards':
        return 'Coord ascent, with initialization'
    elif alg == 'em_without_init':
        return 'Coord ascent, no initialization'
    elif alg == 'joint_with_init':
        #return 'Algorithm 2'
        return 'Joint, with initialization'
    elif alg == 'joint_without_init' or alg == 'joint_no_rewards':
        return 'Joint, no initialization'
    elif alg == 'vi_inference':
        return 'Differentiable VI'
    else:
        raise ValueError('Unknown algorithm ' + alg)

def get_algorithm_color(alg):
    if alg == 'given_rewards':
        return '#b41b1b'
    elif alg == 'boltzmann_planner':
        return '#cccccc'
    elif alg == 'optimal_planner':
        return '#999999'
    elif alg in ['em_with_init', 'em_without_init', 'no_rewards']:
        return '#bbbb00'
    elif alg in ['joint_with_init', 'joint_without_init', 'joint_no_rewards']:
        return '#f79646'
    elif alg == 'vi_inference':
        return '#0000cc'
    else:
        raise ValueError('Unknown algorithm ' + alg)

def get_algorithm_hatch(alg):
    if alg in ['em_without_init', 'joint_without_init', 'joint_no_rewards']:
        return '/'
    elif alg in ['given_rewards', 'boltzmann_planner', 'optimal_planner', 'em_with_init', 'no_rewards', 'joint_with_init', 'vi_inference']:
        return None
    else:
        raise ValueError('Unknown algorithm ' + alg)

def get_bias_name(bias):
    if bias == 'Boltzmann-Optimal':
        return 'Boltzmann'
    elif bias.startswith('Boltzmann'):
        bias_name = bias.split('-')[1]
        return 'B-{}'.format(bias_name)
    return bias

def set_style():
    sns.set(font='serif', font_scale=1.4)
    
   # Make the background white, and specify the
    # specific font family
    sns.set_style("white", {
        "font.family": "serif",
        "font.weight": "normal",
        "font.serif": ["Times", "Palatino", "serif"],
        'axes.facecolor': 'white',
        'lines.markeredgewidth': 1})

def drop_irrelevant_data(data, args):
    cols_to_skip = [alg for alg in ALL_ALGORITHMS if alg not in args.algorithm]
    data.drop(columns=cols_to_skip, inplace=True)
    bias_names = [bias.split('_')[0] for bias in args.bias]
    data = data.query('Agent in {}'.format(repr(bias_names)))
    return data

def read_csv(filename, args):
    df = pd.read_csv(concat_folder(args.folder, filename))
    df = drop_irrelevant_data(df, args)
    return df

def create_graph_structure(args):
    ax_position_to_biases = {}
    max_row, max_col = 0, 0
    for bias in args.bias:
        if '_' in bias:
            bias, pos = bias.split('_')
            row, col = map(int, pos.split(','))
        else:
            row, col = 0, 0
        if (row, col) not in ax_position_to_biases:
            ax_position_to_biases[(row, col)] = []
        ax_position_to_biases[(row, col)].append(bias)
        max_row, max_col = max(max_row, row), max(max_col, col)

    num_rows, num_cols = max_row + 1, max_col + 1
    bias_matrix = [[[] for _ in range(num_cols)] for _ in range(num_rows)]
    col_widths = [0 for _ in range(num_cols)]
    # All rows have height 1
    for (row, col), biases in ax_position_to_biases.items():
        bias_matrix[row][col] = biases
        col_widths[col] = max(col_widths[col], len(biases))

    assert all((width > 0 for width in col_widths)), 'There is an empty column'

    row_starts = list(range(num_rows + 1))
    col_starts = [0]
    for width in col_widths:
        col_starts.append(col_starts[-1] + width)

    return bias_matrix, row_starts, col_starts, num_rows, num_cols

def graph(args):
    var = args.dependent_var
    bias_matrix, row_starts, col_starts, num_rows, num_cols = create_graph_structure(args)
    means_file, sterrs_file = var + '-means.csv', var + '-sterrs.csv'
    means, sterrs = read_csv(means_file, args), read_csv(sterrs_file, args)
    means = {mean['Agent']:mean for _, mean in means.iterrows()}
    sterrs = {sterr['Agent']:sterr for _, sterr in sterrs.iterrows()}
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rc('font',family='serif', serif=['Palatino'])
    sns.set_style('white')

    fig = plt.figure()
    axes = [[None for _ in range(num_cols)] for _ in range(num_rows)]
    gs = GridSpec(row_starts[-1], col_starts[-1])
    for row in range(num_rows):
        for col in range(num_cols):
            if bias_matrix[row][col] == []: continue
            row_start, col_start = row_starts[row], col_starts[col]
            row_end, col_end = row_starts[row+1], col_starts[col+1]
            axes[row][col] = plt.subplot(gs[row_start:row_end, col_start:col_end])
    
    set_style()
    plt.subplots_adjust(wspace=0.35)
    spacing, bar_width = 0.125, 0.5
    num_algs = len(args.algorithm)
    bias_width = num_algs * bar_width + 2 * spacing

    def plot_ax(ax, biases, alg_to_bar):
        num_biases = len(biases)
    
        def make_bar(alg, bias, anum, bnum, mean, sterr, hatch=None):
            # Move to the correct bias, then spacing, then to the correct algorithm
            x_coord = bnum * bias_width + spacing + anum * bar_width
            color = get_algorithm_color(alg)
            hatch = get_algorithm_hatch(alg)
            kwds = {
                'yerr': [sterr],
                'color': [color],
                'edgecolor': 'black',
                'width': bar_width,
                'error_kw': dict(elinewidth=2, capsize=0),
            }
            if hatch is not None:
                kwds['hatch'] = hatch * 5
            return ax.bar([x_coord], [mean], **kwds)

        min_val = 0
        for bnum, bias in enumerate(biases):
            for anum, alg in enumerate(args.algorithm):
                mean, sterr = means[bias][alg], sterrs[bias][alg]
                bar = make_bar(alg, bias, anum, bnum, mean, sterr)
                min_val = min(min_val, mean - sterr)
                alg_to_bar[alg] = bar

        ax.set_ylim([int(10 * floor(min_val / 10)), 100])
        plt.sca(ax)
        bias_names = list(map(get_bias_name, biases))
        plt.xticks([bias_width * (i + 0.5) - 0.5 * bar_width for i in range(num_biases)], bias_names, fontsize=12)
        # ax.set_ylabel('time (s)', fontsize=16, fontweight='normal')

    alg_to_bar = {}
    for row in range(num_rows):
        for col in range(num_cols):
            if axes[row][col] is None: continue
            plot_ax(axes[row][col], bias_matrix[row][col], alg_to_bar)

    if num_cols == 1:
        ax = axes[0][0]
        xmin, xmax = ax.xaxis.get_minpos(), ax.get_position().xmax
        xmin, xmax = xmin - 0.09, xmax + 0.07
        ax.legend(map(alg_to_bar.get, args.algorithm),
                  map(get_algorithm_name, args.algorithm),
                  loc='lower left',
                  bbox_to_anchor=(xmin, 0.9, xmax - xmin, 0.1),
                  mode='expand',
                  ncol=len(args.algorithm),
                  fontsize=12,
        )
    else:
        # Hardcode the legend position
        fig.legend(map(alg_to_bar.get, args.algorithm),
                   map(get_algorithm_name, args.algorithm),
                   loc=(0.07, 0.1),
                   ncol=1,
                   fontsize=12,
        )

    sns.despine(fig)
    fig.set_figwidth(15)
    fig.set_figheight(2.5 * num_rows)
    plt.savefig(args.output_file, bbox_inches='tight', dpi=500)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', required=True)
    parser.add_argument('-o', '--output_file', required=True)
    parser.add_argument('-d', '--dependent_var', required=True)
    parser.add_argument('-b', '--bias', action='append')
    parser.add_argument('-a', '--algorithm', action='append')
    args = parser.parse_args()
    if args.bias is None:
        args.bias = ALL_BIASES[:]
    if args.algorithm is None:
        args.algorithm = ALL_ALGORITHMS[:]

    for bias in args.bias:
        assert bias.split('_')[0] in ALL_BIASES, 'Bad bias {}'.format(bias)
    for alg in args.algorithm:
        assert alg in ALL_ALGORITHMS, 'Bad algorithm {}'.format(alg)
    return args

if __name__ == '__main__':
    args = parse_args()
    graph(args)
