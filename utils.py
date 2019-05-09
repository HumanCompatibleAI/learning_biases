import os
import tensorflow as tf
import numpy as np
import random
import matplotlib
matplotlib.use("tkagg")
import seaborn as sns
import matplotlib.pyplot as plt

# Comment this line out to return to matplotlib plot defaults
sns.set(rc={ # 'text.usetex': True,
            'font.family': 'Times New Roman',
            # This controls linewidth of the hatching that represents the walls
            'hatch.linewidth': 1.5,
            }
        )

# Code taken from https://github.com/TheAbhiKumar/tensorflow-value-iteration-networks
# helper methods to print nice table (taken from CGT code)
def fmt_item(x, l):
    if isinstance(x, np.ndarray):
        assert x.ndim==0
        x = x.item()
    if isinstance(x, float): rep = "%g"%x
    else: rep = str(x)
    return " "*(l - len(rep)) + rep

def fmt_row(width, row):
    out = " | ".join(fmt_item(x, width) for x in row)
    return out

# </End borrowed code>

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

def softmax(v):
    return np.exp(v)/np.sum(np.exp(v))

def squish(v, reward=False):
    if v.any():
        if reward:
            cons = 0
            v = v.copy()
            v[v > 0] += cons
        return v / np.max(v)
    return v

def visualizeReward(reward):
    pos_reward = np.where(reward > 0, reward,  0)
    neg_reward = -1*np.where(reward < 0, reward, 0)
    pos_reward = squish(pos_reward, reward=True)
    neg_reward = squish(neg_reward, reward=True)
    return pos_reward, neg_reward

def plot_reward(reward, walls, ax_title, fig, ax, alpha=1):
    """
    Plots a single reward + wall combination on an axis of the figure given.

    Alpha argument creates alpha values for the reward squares \alpha \in [0, 1]
    """

    # Clean up the arrays (imshow only takes values in [0, 1])
    pos_label, neg_label = visualizeReward(reward)

    # set up plot
    def make_pic(pos_label, walls, neg_label):
        """Combine colors to make the walls + rewards achieve desired color"""
        alphas = np.ones(pos_label.shape)
        alphas[pos_label > 0] = alpha
        alphas[neg_label > 0] = alpha

        # Coloring the walls brown
        # BROWN = np.array((133, 87, 35, 0)) / 255.0
        # wall_color = np.einsum("ij,k->ijk", walls, BROWN)

        # to get our true reward (blue) values on the right scale, we'll create our own color scale
        # Another possibility: 123, 176, 32
        small_positive = np.array((150,189,3, 0)) / 255.0
        # Another possibility: 26,147,111
        big_positive = np.array((85,135,80, 0)) / 255.0
        diff = big_positive - small_positive
        blue = np.stack([np.zeros(neg_label.shape), np.zeros(neg_label.shape), pos_label.copy(), np.zeros(neg_label.shape)], axis=-1)
        blue[pos_label > 0, :] = np.einsum('i,j->ij', pos_label[pos_label > 0], diff) + small_positive


        # Negative reward
        # Another possibility: 223, 161, 177
        small_negative = np.array((227,126,126, 0)) / 255.0
        # Another possibility: 195, 75, 123
        big_negative = np.array((180,27,27, 0)) / 255.0
        diff = big_negative - small_negative
        neg_color = np.stack([neg_label.copy(), np.zeros_like(neg_label), np.zeros_like(neg_label), np.zeros_like(neg_label)], axis=-1)
        neg_color[neg_label > 0, :] = np.einsum('i,j->ij', neg_label[neg_label > 0], diff) + small_negative

        label = np.stack([np.zeros_like(neg_label), np.zeros(pos_label.shape), np.zeros(pos_label.shape), alphas], axis=-1)
        # label = label + blue + wall_color
        label = label + blue + neg_color

        # Set all the black (0,0,0,1) RGBA tuples to be white
        label[np.sum(label, 2) == 1] = np.array([0.9, 0.9, 0.9,1])
        return label.reshape(list(walls.shape)+[4])

    # truth plot
    true = ax.imshow(make_pic(pos_label, walls, neg_label))
    hatch_walls(walls, ax)

    ax.set_title(ax_title)

    # Remove xticks, yticks
    ax.set_yticks([])
    ax.set_xticks([])

    return fig, ax

def hatch_walls(walls, ax, mark='/'):
    """Hatches wall colors.
    Acceptable marks: [‘/’ | ‘' | ‘|’ | ‘-‘ | ‘+’ | ‘x’ | ‘o’ | ‘O’ | ‘.’ | ‘*’]"""
    for row in range(len(walls)):
        for col in range(len(walls[row])):
            if walls[col][row] == 1:
                # Draw via XY points
                Xs = [row - 0.5, row - 0.5, row + 0.5, row + 0.5]
                Ys = [col - 0.5, col + 0.5, col + 0.5, col - 0.5]
                ax.fill(Xs, Ys, hatch=mark*5, fill=False, color='grey')

def plot_policy(walls, policy, fig, ax):
    """Plots arrows in direction of arg max policy"""
    from gridworld.gridworld import Direction
    dir2mark = {
        Direction.NORTH: '^',
        Direction.SOUTH: 'v',
        Direction.EAST: '>',
        Direction.WEST: '<',
        Direction.STAY: '*',
    }
    policy = np.argmax(policy, axis=-1)
    for row in range(len(walls)):
        for col in range(len(walls[row])):
            if walls[col][row] != 1:
                dist = Direction.ALL_DIRECTIONS[policy[col, row]]
                mark = dir2mark[dist]
                plot_pos((row, col), marker=mark, color='black', grid_size=len(walls), ax=ax)

def plot_policy_diff(predicted, true, walls, fig, ax):
    """Plots policy, boxes wrong answers"""
    from matplotlib.patches import Rectangle

    plot_policy(walls, predicted, fig, ax)

    predicted = np.argmax(predicted, axis=-1)
    true = np.argmax(true, axis=-1)

    for i in range(len(predicted)):
        for j in range(len(predicted)):
            if predicted[i, j] != true[i, j]:
                ax.add_patch(
                    Rectangle(
                        (j-0.5, i-0.5), 1, 1, fill=False, edgecolor='red', linewidth=1.5
                ))



def plot_trajectory(wall, reward, start, agent, fig, ax, arrow_width=0.5, EPISODE_LENGTH=35,
                    animate=False, fname=None):
    """Simulates a rollout of an agent given an MDP specified
    by the wall, reward, and start state. And plots it.

    If animate is true, an animation object will be returned
    """
    from gridworld.gridworld import GridworldMdp
    from mdp_interface import Mdp
    from agent_runner import run_agent

    mdp = GridworldMdp.from_numpy_input(wall, reward, start)

    agent.set_mdp(mdp)
    env = Mdp(mdp)
    trajectory = run_agent(agent, env, episode_length=EPISODE_LENGTH, determinism=True)

    if len(trajectory) <= 1:
        raise ValueError("Trajectory rolled out unsuccessfully")

    # Tuples of (state, next) - to be used for plotting
    state_trans = [(info[0], info[2]) for info in trajectory]
    count = 0
    for trans in state_trans:
        if trans[0] == trans[1]:
            count += 1
    if count == len(state_trans):
        print("Yes, the agent given stayed in the same spot for {} iterations...".format(len(state_trans)))

    if fig is None or ax is None:
        fig, ax = plt.subplots(1, 1)
    if ax is not None and type(ax) is list:
        raise ValueError("Given {} axes, but can only use 1 axis".format(len(ax)))

    # Plot starting point
    plot_pos(start, ax=ax, color='k', marker='o', grid_size=len(wall))
    # Plot ending trajectory point
    finish = state_trans[-1][0]
    plot_pos(finish, ax=ax, color='k', marker='*', grid_size=len(wall))
    plot_lines(ax, fig, trans_list=state_trans, color='black', arrow_width=arrow_width, grid_size=len(wall),
               animate=animate, fname=fname)
    ax.set_xticks([])
    ax.set_yticks([])
    return fig, ax

def plot_reward_and_trajectories(true_reward, inferred_reward, walls, start, config, filename='reward_comparison.png',
                                 animate=False):
    """Plots reward vs inferred reward. On the true reward, plot the biased agent's trajectory. On the
    inferred reward, plot the optimal agent's trajectory.

    true_reward(ndarray): shape=(imsize x imsize)
    inferred_reward(ndarray): same as above
    walls(ndarray): shape=(imsize x imsize) of 0s and 1s, where 1s are walls
    start(tuple): containing (row, col)
    config(tf.config): config with agent params set
    filename(string): pathname of saved figure
    """
    from agents import OptimalAgent
    from gridworld.gridworld_data import create_agents_from_config
    dirs = os.path.dirname(filename)
    os.makedirs(dirs, exist_ok=True)

    true_agent, other_agent = create_agents_from_config(config)
    inferred_agent = OptimalAgent()

    _plot_reward_and_trajectories_helper(true_reward, inferred_reward, walls, start, true_agent, inferred_agent,
                                         filename)


def _plot_reward_and_trajectories_helper(true_reward, inferred_reward, walls, start, true_agent, inferred_agent,
                                          filename='reward_comparison.png', animate=False):
    """Plots same thing as plot_reward_and_trajectories, but using only agents, no config"""
    from agents import OptimalAgent
    from gridworld.gridworld_data import create_agents_from_config
    # 1 Figure, 2 Plots (in a row)
    # True reward on leftmost plot (axes[0])
    # Inferred reward on rightmost plot (axes[1])
    fig, axes = plt.subplots(1, 2)

    # Plot the rewards
    plot_reward(true_reward, walls, 'True Reward', fig=fig, ax=axes[0])
    plot_reward(inferred_reward, walls,'Inferred Reward', fig=fig, ax=axes[1])
    # Plot the agents' trajectories (will perform rollout)
    plot_trajectory(walls, true_reward, start, true_agent, fig=fig, ax=axes[0], animate=animate,
                    fname=filename+'0')
    plot_trajectory(walls, inferred_reward, start, inferred_agent, fig=fig, ax=axes[1], animate=animate,
                    fname=filename+'1')
    # Plot starting positions for agents in both the true and inferred reward plots
    plot_pos(start, color='m', grid_size=len(walls), ax=axes[0])
    plot_pos(start, color='m', grid_size=len(walls), ax=axes[1])

    # titleing
    fig.suptitle("Comparison of Reward Functions")
    fig.set_tight_layout(True)

    # saving to file
    fig.savefig(filename)


def test_trajectory_plotting():
    """Tests trajectory plotting"""
    from gridworld.gridworld import GridworldMdp
    from agents import OptimalAgent, MyopicAgent
    agent = OptimalAgent()
    mdp = GridworldMdp.generate_random(12, 12, pr_wall=0.1, pr_reward=0.1)
    agent.set_mdp(mdp)
    walls, reward, start = mdp.convert_to_numpy_input()
    myopic = MyopicAgent(horizon=10)
    _plot_reward_and_trajectories_helper(reward, reward, walls, start, myopic, OptimalAgent(), filename="trajectory.png")
    # fig, axes = plt.subplots(1, 2)
    # fig, axes = plot_reward(reward, reward, walls, filename='trajectory_test.png', fig=fig, axes=axes)
    # plot_pos(start, color='m', grid_size=len(walls), ax=axes[1])
    # plot_trajectory(walls, reward, start, agent_type=OptimalAgent, fig=fig, axes=axes[1])

def plot_pos(start, color=None, marker='*', grid_size=None, ax=None):
    """Plots a small dot on the start location"""
    if grid_size is None:
        raise ValueError("Need a value for `grid_size`. Nothing was passed in.")
    if ax is None:
        raise ValueError("Please pass MPL axes.")
    col, row = start
    if color is None:
        color = 'r'
    ax.scatter([col], [row], color=color, s=30, marker=marker)

def plot_lines(ax, fig, trans_list, arrow_width=0.5, color='w', grid_size=None, animate=False, fname=None):
    from matplotlib.animation import FuncAnimation
    """Plots transitions as lines on a grid (centered on grid points)"""
    from gridworld.gridworld import Direction
    if grid_size is None:
        raise ValueError("Need a value for `grid_size`. Nothing was passed in.")
    # from matplotlib.colors import LinearSegmentedColormap

    # RGBA vals that go from pinkish to yellowish -- for dynamic coloring
    # reds = [(1, 0, 1, 1), (1, 1, 0, 1)]
    # cgrad = LinearSegmentedColormap.from_list(name="reds", colors=reds, N=num_trans)

    def drawMove(i):
        trans = trans_list[i]
        start, end = trans
        p1, p2 = start, end
        # This just draws arrows of form, arrow(x, y, dx, dy)
        # line = ax.arrow(p1[0], p1[1], p2[0] - p1[0], p2[1] - p1[1], color=color, head_width=0.3, head_length=0.25, length_includes_head=True)
        line = ax.plot((p1[0], p2[0]), (p1[1], p2[1]), color=color, ls='-')
        # midX = (4*p2[0] + p1[0]) / 5.0
        # midY = (4*p2[1] + p1[1]) / 5.0
        midX = p2[0]
        midY = p2[1]
        arrow_style = 'simple,head_width={},tail_width=0'.format(arrow_width)
        ax.annotate('', xy=(midX, midY), xytext=(p1[0], p1[1]), arrowprops=dict(arrowstyle=arrow_style, facecolor='k'))
        # For dynamic coloring
        # line = ax.plot((p1[0], p2[0]), (p1[1], p2[1]), color=cgrad(i), ls='--')
        return ax

    if not animate:
        # line_artists = []
        # # For future matplotlib usage (just in case)
        # line_artists.append(line)
        for i in range(len((trans_list))):
            drawMove(i)
    else:
        anim = FuncAnimation(fig=fig, frames=np.arange(0, len(trans_list)), func=drawMove, interval=70)
        anim.save(fname+'.mp4', dpi=100)



def init_flags():
    # Algorithm
    tf.app.flags.DEFINE_string(
        'algorithm', 'given_rewards', 'Which algorithm to run')
    tf.app.flags.DEFINE_integer(
        'em_iterations', 2, 'Number of iterations for the EM-like algorithm')

    # Data flags
    #   Generate data
    tf.app.flags.DEFINE_boolean(
        'simple_mdp', False, 'Whether to use the simple random MDP generator')
    tf.app.flags.DEFINE_integer('imsize', 14, 'Size of input image')
    tf.app.flags.DEFINE_float(
        'wall_prob', 0.05,
        'Probability of having a wall at any particular space in the gridworld. '
        'Has no effect if --simple_mdp is False.')
    tf.app.flags.DEFINE_float(
        'reward_prob', 0.05,
        'Probability of having a reward at any particular space in the gridworld. '
        'Has no effect if --simple_mdp is False.')
    tf.app.flags.DEFINE_integer(
        'num_rewards', 5,
        'Number of positions in the gridworld that should have reward. '
        'Has no effect if --simple_mdp is True.')
    tf.app.flags.DEFINE_float(
        'noise', 0.0, 'Probability that the intended action fails')
    tf.app.flags.DEFINE_float(
        'action_distance_threshold', 0.5,
        'Minimum distance between two action distributions to be "different"')
    tf.app.flags.DEFINE_integer(
        'num_human_trajectories', 8000, 'Number of human trajectories we see')
    tf.app.flags.DEFINE_integer(
        'num_validation', 2000,
        'Number of extra trajectories to generate to validate the planning module')
    tf.app.flags.DEFINE_integer(
        'num_with_rewards', 0, 'Number of MDPs where reward info is known')
    tf.app.flags.DEFINE_integer(
        'num_simulated', 0, 'Number of MDPs with simulated trajectories')

    # Hyperparameters
    tf.app.flags.DEFINE_string(
        'model','SIMPLE','VIN, SIMPLE, or VI')
    tf.app.flags.DEFINE_float(
        'vin_regularizer_C', 0.0001, 'Regularization constant for the VIN')
    tf.app.flags.DEFINE_float(
        'reward_regularizer_C', 0, 'Regularization constant for the reward')
    tf.app.flags.DEFINE_float(
        'lr', 0.01, 'Learning rate when training the planning module')
    tf.app.flags.DEFINE_float(
        'reward_lr', 1.0, 'Learning rate when inferring a reward function')
    tf.app.flags.DEFINE_integer(
        'epochs', 20, 'Number of epochs to train the planning module for')
    tf.app.flags.DEFINE_integer(
        'reward_epochs', 50, 'Number of epochs when inferring a reward function')
    tf.app.flags.DEFINE_integer('k', 10, 'Number of value iterations')
    tf.app.flags.DEFINE_integer('ch_h', 150, 'Channels in initial hidden layer')
    tf.app.flags.DEFINE_integer('ch_p', 5, 'Channels in proxy reward layer')
    tf.app.flags.DEFINE_integer('ch_q', 5, 'Channels in q layer')
    tf.app.flags.DEFINE_integer('num_actions', 5, 'Number of actions')
    tf.app.flags.DEFINE_integer('batchsize', 20, 'Batch size')

    # Agent
    tf.app.flags.DEFINE_string(
        'agent', 'optimal', 'Agent to generate training data with')
    tf.app.flags.DEFINE_float('gamma', 0.95, 'Discount factor')
    tf.app.flags.DEFINE_float(
        'beta', None, 'Noise for the agent when selecting actions')
    tf.app.flags.DEFINE_integer(
        'num_iters', 50,
        'Number of iterations of value iteration the agent should run.')
    tf.app.flags.DEFINE_integer(
        'max_delay', 10,
        'Maximum delay that the agent should use. '
        'Only affects naive, sophisticated and myopic agents.')
    tf.app.flags.DEFINE_float(
        'hyperbolic_constant', 1.0,
        'Discount for the future for hyperbolic time discounters')
    tf.app.flags.DEFINE_float(
        'calibration_factor', 1.0,
        'Calibration factor for uncalibrated agents.')
    tf.app.flags.DEFINE_integer(
        'eval_horizon', 20,
        'Number of steps after which to stop running the agent when evaluating final rewards'
    )

    # Other Agent
    tf.app.flags.DEFINE_string(
        'other_agent', None,
        'Agent to distinguish from. '
        'In particular, when generating training data, we print the number of '
        'training examples on which agent and other_agent would choose different '
        'action distributions.')
    tf.app.flags.DEFINE_float('other_gamma', 0.9, 'Gamma for other agent')
    tf.app.flags.DEFINE_float('other_beta', None, 'Beta for other agent')
    tf.app.flags.DEFINE_integer('other_num_iters', 50, 'Num iters for other agent')
    tf.app.flags.DEFINE_integer('other_max_delay', 5, 'Max delay for other agent')
    tf.app.flags.DEFINE_float(
        'other_hyperbolic_constant', 1.0, 'Hyperbolic constant for other agent')
    tf.app.flags.DEFINE_float(
        'other_calibration_factor', 1.0, 'Calibration factor for other agent')

    # Output
    tf.app.flags.DEFINE_string(
        'output_folder', 'data/scratch_data/', 'Folder to write statistics to')
    tf.app.flags.DEFINE_integer(
        'display_step', 1, 'Print summary output every n epochs')
    tf.app.flags.DEFINE_boolean('log', False, 'Enables tensorboard summary')
    tf.app.flags.DEFINE_string(
        'logdir', '/tmp/planner-vin/', 'Directory to store tensorboard summary')
    tf.app.flags.DEFINE_integer(
        'verbosity', 3,
        """Level of output to terminal (higher means more output).
        Level 0 suppresses all output.
        Level 1 includes only output about key metrics.
        Level 2 includes infrequent progress updates.
        Level 3 provides detailed information on training progress.
        """)
    tf.app.flags.DEFINE_boolean(
        'plot_rewards', True, 'Whether or not to plot rewards')
    tf.app.flags.DEFINE_boolean(
        'savemodel', False, 'Whether or not to save the model')

    # Miscellaneous
    tf.app.flags.DEFINE_string(
        'seeds', '1,2,3,5,8,13,21,34', 'Random seeds for both numpy and random')
    tf.app.flags.DEFINE_bool('use_gpu', False, 'Enables GPU usage')
    tf.app.flags.DEFINE_bool('strict', False, 'Disables permissive flags')

    config = tf.app.flags.FLAGS
    config.seeds = list(map(int, config.seeds.split(',')))
    alg = config.algorithm

    def warn_or_error(message):
        if config.strict:
            raise ValueError(message)
        else:
            print(message)

    def check_zero(flag):
        if getattr(config, flag) != 0:
            warn_or_error('{} > 0 is useless for algorithm {}'.format(flag, alg))

    def check_nonzero(flag, default):
        if getattr(config, flag) == 0:
            warn_or_error('{} must be nonzero for algorithm {}'.format(flag, alg))
            setattr(config, flag, default)
            print('Setting it to ' + str(default))

    check_nonzero('num_human_trajectories', 8000)
    if alg == 'given_rewards':
        check_zero('em_iterations')
        check_zero('num_simulated')
        check_nonzero('num_with_rewards', config.num_human_trajectories - 1000)
        check_nonzero('num_validation', 2000)
    elif alg in ['boltzmann_planner', 'optimal_planner']:
        check_zero('em_iterations')
        check_zero('num_with_rewards')
        check_nonzero('num_simulated', 5000)
        check_nonzero('num_validation', 2000)
    elif alg == 'em_with_init':
        check_zero('num_with_rewards')
        check_nonzero('em_iterations', 2)
        check_nonzero('num_simulated', 5000)
        check_nonzero('num_validation', 2000)
    elif alg == 'em_without_init':
        check_zero('num_with_rewards')
        check_zero('num_simulated')
        check_zero('num_validation')
        check_nonzero('em_iterations', 2)
    elif alg == 'joint_with_init':
        check_zero('em_iterations')
        check_zero('num_with_rewards')
        check_nonzero('num_simulated', 5000)
        check_nonzero('num_validation', 2000)
    elif alg in ['joint_without_init', 'vi_inference']:
        check_zero('em_iterations')
        check_zero('num_with_rewards')
        check_zero('num_simulated')
        check_zero('num_validation')
    else:
        raise ValueError('Unknown algorithm {}'.format(alg))

    if config.agent == 'overconfident':
        assert config.calibration_factor > 1.0
    elif config.agent == 'underconfident':
        assert config.calibration_factor < 1.0

    return config

class Distribution(object):
    """Represents a probability distribution.

    The distribution is stored in a canonical form where items are mapped to
    their probabilities. The distribution is always normalized (so that the
    probabilities sum to 1).
    """
    def __init__(self, probability_mapping):
        # Convert to a list so that we aren't iterating over the dictionary and
        # removing at the same time
        for key in list(probability_mapping.keys()):
            prob = probability_mapping[key]
            if prob == 0:
                del probability_mapping[key]
            elif prob < 0:
                raise ValueError('Cannot have negative probability!')

        assert len(probability_mapping) > 0
        self.dist = probability_mapping
        self.normalize()

    def factor(self, key, factor):
        """Updates the probability distribution as though we see evidence that
        is `factor` times more likely for `key` than for any other key."""
        self.dist[key] *= factor
        self.normalize()

    def normalize(self):
        Z = float(sum(self.dist.values()))
        for key in list(self.dist.keys()):
            self.dist[key] /= Z

    def sample(self):
        keys, probabilities = zip(*self.dist.items())
        return keys[np.random.choice(np.arange(len(keys)), p=probabilities)]

    def get_dict(self):
        return self.dist.copy()

    def as_numpy_array(self, fn=None, length=None):
        if fn is None:
            fn = lambda x: x
        keys = list(self.dist.keys())
        numeric_keys = [fn(key) for key in keys]
        if length is None:
            length = max(numeric_keys) + 1

        result = np.zeros(length)
        for key, numeric_key in zip(keys, numeric_keys):
            result[numeric_key] = self.dist[key]
        return result

    def __eq__(self, other):
        return self.dist == other.dist

    def __str__(self):
        return str(self.dist)

    def __repr__(self):
        return 'Distribution(%s)' % repr(self.dist)


def concat_folder(folder, element):
    """folder and element are strings"""
    if folder[-1] == '/':
        return folder + element
    return folder + '/' + element


if __name__ == '__main__':
    test_trajectory_plotting()
