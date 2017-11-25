# Code taken from https://github.com/TheAbhiKumar/tensorflow-value-iteration-networks
import tensorflow as tf
import numpy as np
import re

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

def init_flags():
    # Data flags
    #   Load data
    tf.app.flags.DEFINE_string('datafile', None, 'Where to get data from, only used it not None')
    #   Generate data
    tf.app.flags.DEFINE_boolean(
        'simple_mdp', False, 'Whether to use the simple random MDP generator')
    tf.app.flags.DEFINE_integer('imsize', 8, 'Size of input image')
    tf.app.flags.DEFINE_float(
        'wall_prob', 0.05,
        'Probability of having a wall at any particular space in the gridworld. '
        'Has no effect if --simple_mdp is False.')
    tf.app.flags.DEFINE_float(
        'reward_prob', 0.05,
        'Probability of having a reward at any particular space in the gridworld')
    tf.app.flags.DEFINE_float(
        'action_distance_threshold', 0.5,
        'Minimum distance between two action distributions to be "different"')
    tf.app.flags.DEFINE_integer(
        'num_train', 500, 'Number of examples for training the planning module')
    tf.app.flags.DEFINE_integer(
        'num_test', 200, 'Number of examples for testing the planning module')

    # Hyperparameters
    tf.app.flags.DEFINE_float(
        'vin_regularizer_C', 0.0001, 'Regularization constant for the VIN')
    tf.app.flags.DEFINE_float(
        'reward_regularizer_C', 0.0001, 'Regularization constant for the reward')
    tf.app.flags.DEFINE_float(
        'lr', 0.025, 'Learning rate when training the planning module')
    tf.app.flags.DEFINE_float(
        'reward_lr', 0.1, 'Learning rate when inferring a reward function')
    tf.app.flags.DEFINE_integer(
        'epochs', 30, 'Number of epochs to train the planning module for')
    tf.app.flags.DEFINE_integer(
        'reward_epochs', 50, 'Number of epochs when inferring a reward function')
    tf.app.flags.DEFINE_integer('k', 10, 'Number of value iterations')
    tf.app.flags.DEFINE_integer('ch_h', 150, 'Channels in initial hidden layer')
    tf.app.flags.DEFINE_integer('ch_q', 5, 'Channels in q layer')
    tf.app.flags.DEFINE_integer('num_actions', 5, 'Number of actions')
    tf.app.flags.DEFINE_integer('batchsize', 12, 'Batch size')
    tf.app.flags.DEFINE_integer(
        'statebatchsize', 10,
        'Number of state inputs for each sample (real number, technically is k+1)')

    # Agent
    tf.app.flags.DEFINE_string(
        'agent', 'optimal', 'Agent to generate training data with')
    tf.app.flags.DEFINE_float('gamma', 1.0, 'Discount factor')
    tf.app.flags.DEFINE_float('beta', None, 'Noise when selecting actions')
    tf.app.flags.DEFINE_integer(
        'num_iters', 50,
        'Number of iterations of value iteration the agent should run.')
    tf.app.flags.DEFINE_integer(
        'max_delay', 5,
        'Maximum delay that the agent should use. '
        'Only affects naive/sophisticated and myopic agents.')
    tf.app.flags.DEFINE_float(
        'hyperbolic_constant', 1.0,
        'Discount for the future for hyperbolic time discounters')

    # Other Agent
    tf.app.flags.DEFINE_string(
        'other_agent', None,
        'Agent to distinguish from. '
        'In particular, when generating training data, we print the number of '
        'training examples on which agent and other_agent would choose different '
        'action distributions.')
    tf.app.flags.DEFINE_float('other_gamma', 1.0, 'Gamma for other agent')
    tf.app.flags.DEFINE_float('other_beta', None, 'Beta for other agent')
    tf.app.flags.DEFINE_integer('other_num_iters', 50, 'Num iters for other agent')
    tf.app.flags.DEFINE_integer('other_max_delay', 5, 'Max delay for other agent')
    tf.app.flags.DEFINE_float(
        'other_hyperbolic_constant', 1.0, 'Hyperbolic constant for other agent')

    # Miscellaneous
    tf.app.flags.DEFINE_integer('seed', 0, 'Random seed for both numpy and random')
    tf.app.flags.DEFINE_integer(
        'display_step', 1, 'Print summary output every n epochs')
    tf.app.flags.DEFINE_boolean('log', False, 'Enables tensorboard summary')
    tf.app.flags.DEFINE_string(
        'logdir', '/tmp/planner-vin/', 'Directory to store tensorboard summary')


    config = tf.app.flags.FLAGS

    # It is required that the number of unknown reward functions be equal to the
    # batch size. If we tried to train multiple batches, then they would all be
    # modifying the same reward function, which would be bad.
    config.num_mdps = config.batchsize

    if config.datafile:
        get_flag_data_from_filename(config, config.datafile) # gets everything including seed
    return config

def get_flag_data_from_filename(config, fname):
    """ From a filename, get all the hyperparameters and push them into config """

    names = ['num_train', 'num_test', 'seed', 'imsize',
    'reward_prob', 'batchsize', 'statebatchsize', 'simple_mdp',
    'action_distance_threshold', 'agent', 'gamma', 'beta', 'max_delay', 'hyperbolic_constant'
    ]

    values = re.findall(r"-([^-]*)[-\.]", fname)
    for name, val in zip(names, values):
        if val == 'None':
            val = None
        elif val == 'True':
            val = True
        elif val =='False':
            val = False
        elif '.' in val:
            val = float(val)
        elif re.search('\d', val):
            val = int(val)
        setattr(config, name, val)
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


