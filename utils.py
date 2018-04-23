import tensorflow as tf
import numpy as np
import random
import re
import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt

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

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

def softmax(v):
    return np.exp(v)/np.sum(np.exp(v))

def squish(v):
    if v.any():
        return (v - np.min(v)) / np.max(v-np.min(v))
    return v

def visualizeReward(reward):
    pos_reward = np.where(reward > 0, reward,  0)
    neg_reward = -1*np.where(reward < 0, reward, 0)
    pos_reward = squish(pos_reward)
    neg_reward = squish(neg_reward)
    return pos_reward,neg_reward

def plot_reward(label, inferred_reward, walls, filename='reward_comparison.png'):
    """Plots rewards (true and predicted) and saves them to a file.

    Inferred_reward should be normalized before.
    """

    # Clean up the arrays (imshow only takes values in [0, 1])
    pos_label, neg_label = visualizeReward(label)
    pos_reward, neg_reward = visualizeReward(inferred_reward)

    # set up plot
    fig, axes = plt.subplots(1,2)
    label = np.stack([pos_label, walls, neg_label],axis=-1).reshape(list(walls.shape)+[3])

    # truth plot
    true = axes[0].imshow(label)
    axes[0].set_title("Truth")

    # inferred plot
    rew = np.stack([pos_reward, walls, neg_reward],axis=-1).reshape(list(walls.shape)+[3])
    tensor = axes[1].imshow(rew)
    axes[1].set_title("Predicted")

    # titleing
    fig.suptitle("Comparison of Reward Functions")

    # saving to file
    fig.savefig(filename)

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
    tf.app.flags.DEFINE_integer('imsize', 16, 'Size of input image')
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
    tf.app.flags.DEFINE_float('beta', None, 'Noise when selecting actions')
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

    # Output
    tf.app.flags.DEFINE_string(
        'output_folder', 'data/', 'Folder to write statistics to')
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
    elif alg == 'no_rewards':
        check_zero('num_with_rewards')
        check_nonzero('em_iterations', 2)
        check_nonzero('num_simulated', 5000)
        check_nonzero('num_validation', 2000)
    elif alg in ['boltzmann_planner', 'optimal_planner']:
        check_zero('em_iterations')
        check_zero('num_with_rewards')
        check_nonzero('num_simulated', 5000)
        check_nonzero('num_validation', 2000)
    elif alg in ['joint_no_rewards', 'vi_inference']:
        check_zero('em_iterations')
        check_zero('num_with_rewards')
        check_zero('num_simulated')
        check_zero('num_validation')
    else:
        raise ValueError('Unknown algorithm {}'.format(alg))

    return config

class Distribution(object):
    """Represents a probability distribution.

    The distribution is stored in a canonical form where items are mapped to
    their probabilities. The distribution is always normalized (so that the
    probabilities sum to 1).
    """
    def __init__(self, probability_mapping):
        Z = float(sum(probability_mapping.values()))
        # Convert to a list so that we aren't iterating over the dictionary and
        # removing at the same time
        for key in list(probability_mapping.keys()):
            prob = probability_mapping[key]
            if prob == 0:
                del probability_mapping[key]
            elif prob < 0:
                raise ValueError('Cannot have negative probability!')
            else:
                probability_mapping[key] = prob / Z

        assert len(probability_mapping) > 0
        self.dist = probability_mapping

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
