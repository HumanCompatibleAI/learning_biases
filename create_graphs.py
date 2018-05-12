import argparse
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ALL_BIASES = [
    'Optimal', 'Naive', 'Sophisticated', 'Myopic', 'Overconfident',
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
        return 'VIN with rewards'
    elif alg == 'boltzmann_planner':
        return 'Boltzmann VIN'
    elif alg == 'optimal_planner':
        return 'Optimal VIN'
    elif alg == 'em_with_init' or alg == 'no_rewards':
        return 'Coordinate ascent with initialization'
    elif alg == 'em_without_init':
        return 'Coordinate ascent without initialization'
    elif alg == 'joint_with_init':
        return 'Joint training with initialization'
    elif alg == 'joint_without_init' or alg == 'joint_no_rewards':
        return 'Joint training without initialization'
    elif alg == 'vi_inference':
        return 'Differentiable VI'
    else:
        raise ValueError('Unknown algorithm ' + alg)

def get_algorithm_color(alg):
    if alg == 'given_rewards':
        return '#f79646'
    elif alg == 'boltzmann_planner':
        return '#cccccc'
    elif alg == 'optimal_planner':
        return '#00cc00'
    elif alg == 'em_with_init' or alg == 'no_rewards':
        return 'Coordinate ascent with initialization'
    elif alg == 'em_without_init':
        return 'Coordinate ascent without initialization'
    elif alg == 'joint_with_init':
        return 'Joint training with initialization'
    elif alg == 'joint_without_init' or alg == 'joint_no_rewards':
        return 'Joint training without initialization'
    elif alg == 'vi_inference':
        return '#0000cc'
    else:
        raise ValueError('Unknown algorithm ' + alg)

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
    data = data.query('Agent in {}'.format(repr(args.bias)))
    return data

def read_csv(filename, args):
    df = pd.read_csv(concat_folder(args.folder, filename))
    df = drop_irrelevant_data(df, args)
    return df

def graph(args):
    var = args.dependent_var
    means_file, sterrs_file = var + '-means.csv', var + '-sterrs.csv'
    means, sterrs = read_csv(means_file, args), read_csv(sterrs_file, args)
    matplotlib.rcParams['text.usetex'] = True
    matplotlib.rc('font',family='serif', serif=['Palatino'])
    sns.set_style('white')

    fig, ax = plt.subplots()
    set_style()
    plt.subplots_adjust(wspace=0.35)

    num_algs, num_biases = map(len, (args.algorithm, args.bias))
    spacing, bar_width = 0.125, 0.5
    bias_width = num_algs * bar_width + 2 * spacing
    
    def make_bar(alg, bias, mean, sterr):
        anum = args.algorithm.index(alg)
        bnum = args.bias.index(bias)
        # Move to the correct bias, then spacing, then to the correct algorithm
        x_coord = bnum * bias_width + spacing + anum * bar_width
        color = get_algorithm_color(alg)
        return ax.bar([x_coord], [mean], yerr=[sterr],
                       color=[color], ecolor='black',
                       width=bar_width,
                       error_kw=dict(elinewidth=2, capsize=0))

    alg_to_bar = {}
    ax.set_title('Reward obtained', fontsize=16, fontweight='normal')
    for ((_, mean), (_, sterr)) in zip(means.iterrows(), sterrs.iterrows()):
        assert mean['Agent'] == sterr['Agent']
        bias = mean['Agent']
        for alg in args.algorithm:
            bar = make_bar(alg, bias, mean[alg], sterr[alg])
            alg_to_bar[alg] = bar

    ax.set_ylim([0, 100])
    # ax.set_xlim([-0.25, 2.25])
    plt.sca(ax)
    plt.xticks([bias_width * (i + 0.5) for i in range(num_biases)], args.bias)
    # ax.set_ylabel('time (s)', fontsize=16, fontweight='normal')

    leg = ax.legend(map(alg_to_bar.get, args.algorithm),
                    map(get_algorithm_name, args.algorithm),
                    #loc='upper right',
                    ncol=1, 
                    fontsize=11,
                    #bbox_to_anchor=(1.8, 1.0),
    )

    sns.despine(fig)
    fig.set_figwidth(15)
    fig.set_figheight(4)
    plt.savefig(args.output_file, bbox_inches='tight')

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

    assert all((bias in ALL_BIASES for bias in args.bias)), 'Bad bias'
    assert all((alg in ALL_ALGORITHMS for alg in args.algorithm)), 'Bad algorithm'
    return args

if __name__ == '__main__':
    args = parse_args()
    graph(args)
