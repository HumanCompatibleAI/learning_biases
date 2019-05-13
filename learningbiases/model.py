# Code taken from https://github.com/TheAbhiKumar/tensorflow-value-iteration-networks
import numpy as np
import tensorflow as tf

class Model(object):
    """Encapsulates a model that given an MDP predicts an agent's policy.

    Saves references to useful Tensors, such as logits.
    """

    def __init__(self, logits, output_probs):
        self.logits = logits
        self.output_probs = output_probs


def create_model(image, reward, config):
    X  = tf.stack([image, reward], axis=-1)
    if config.model == 'VIN':
        return VI_Block(X, config)
    elif config.model == "SIMPLE":
        return simple_model(X, config)
    elif config.model == 'VI':
        return tf_value_iter(X, config)
    else:
        raise ValueError('Unknown model: ' + config.model)


def simple_model(X, config):
    """ Creates Conv-Net to run on 2-channel Grid Input (walls, rewards)
    However, to ensure that the entire grid is convolved over, each architecture has to be individually constructed"""

    # HYPERPARAMETERS
    # ---------------------------
    ch_i = 2            # Number of channels in input layer (image, reward)
    ch_q = config.ch_q  # Channels in q layer (~actions) 
    imsize = config.imsize
    num_actions = config.num_actions
    # regularizer = tf.contrib.l2_regularizer(scale=0.001)

    final_shape = [config.batchsize, imsize, imsize, ch_q]
    if imsize == 8:
        first = conv_layer(X,[1,1,ch_i,ch_q],'conv_0',pad='SAME')

        # Second conv (3x3)
        conv = conv_layer(X,[3,3,ch_i,ch_i],'conv1',strides=[1,3,3,1],pad='VALID')
        second = convt_layer(conv,[3,3,ch_q,ch_i],'convt1',
            final_shape,strides=[1,3,3,1],pad='VALID',activation=None)

        # Third conv (3x3)
        conv = conv_layer(conv, [2,2,ch_i,ch_i], 'conv2',strides=[1,1,1,1],pad='SAME')
        twopta = convt_layer(conv, [2,2,ch_q,ch_i], 'convt2a',
            [config.batchsize,imsize//2,imsize//2,ch_q],strides=(1,2,2,1),pad='VALID',activation=None)
        third = convt_layer(twopta, [2,2,ch_q,ch_q], 'convt2b',
            final_shape,strides=(1,2,2,1),pad='VALID',activation=None)
    elif imsize == 14:
        # in_layer = 
        first = conv_layer(X, [1,1,ch_i,ch_q], 'conv_0',pad='SAME')

        # 14x14x2 --> 6x6x2
        conv = conv_layer(X, [3,3,ch_i,ch_i], 'conv1', strides=[1,2,2,1],pad='VALID')
        #   6x6x2 --> 14x14x2
        second = convt_layer(conv, [3,3,ch_q,ch_i],'convt1a',final_shape,strides=[1,2,2,1],pad='VALID',activation=None)

        # 6x6x2 --> 2x2x2
        conv = conv_layer(conv, [3,3,ch_i,ch_i], 'conv2', strides=[1,2,2,1],pad='VALID')
        #   2x2x2 --> 7x7x2
        intermed = convt_layer(conv, [4,4,ch_i,ch_i],'convt2a',
            [config.batchsize,7,7,ch_i],strides=[1,3,3,1],pad='VALID',activation=None)
        #   7x7x2 --> 14x14x6
        third = convt_layer(intermed, [2,2,ch_q,ch_i],'convt2b',final_shape,strides=[1,2,2,1],pad='VALID',activation=None)
        
        # # 2x2x2 --> 8
        # flattened = tf.reshape(conv, shape=[-1, 8])
        # #   8 --> 54 --> 3x3x6
        # fc_w = tf.Variable(tf.truncated_normal((8,54)),name='fc3_w')
        # fc_b = tf.Variable(tf.truncated_normal((54,)),name='fc3_b')
        # fc_middle = tf.nn.relu(tf.matmul(flattened,fc_w) + fc_b, name='fc3')
        # fc_shaped = tf.reshape(fc_middle,shape=([config.batchsize,3,3,ch_q]))
        # #   3x3x6 --> 7x7x6
        # intermed = convt_layer(fc_shaped, [3,3,ch_q,ch_q], 'convt3a',
        #     [config.batchsize,7,7,ch_q],strides=[1,2,2,1],pad='VALID',activation=None)
        # #   7x7x6 --> 14x14x6
        # fourth = convt_layer(intermed, [2,2,ch_q,ch_q], 'convt3b',
        #     final_shape,strides=[1,2,2,1],pad='VALID',activation=None)

    else:
        raise Exception("imsize must be in {8, 14}. Other architectures not yet specified")

    comb_wts = tf.Variable(tf.truncated_normal((3,)), dtype=tf.float32, name='combination_wts')

    # Take average of the output
    X = comb_wts[0]*first + comb_wts[1]*second + comb_wts[2]*third
    X = tf.reshape(X, [-1, ch_q])
    return Model(X, tf.nn.softmax(X, name='output'))

def conv_layer(x,filter_shape,name,pad,strides=(1,1,1,1),activation=tf.nn.relu):
    w, b = weight_and_bias(filter_shape,name)
    logit = tf.nn.conv2d(x,w,strides=strides,name='conv',padding=pad)+b
    return activation(logit, name='out') if activation else logit

def convt_layer(x,filter_shape,name,output_shape,pad,strides=(1,1,1,1),activation=tf.nn.relu):
    w, b = weight_and_bias(filter_shape,name)
    logit = tf.nn.conv2d_transpose(x,w,strides=strides,name='conv',output_shape=output_shape,padding=pad)+b
    return activation(logit, name='out') if activation else logit

def weight_and_bias(filter_shape,name):
    with tf.variable_scope(name):
        w = tf.get_variable(name='filter',initializer=tf.truncated_normal(filter_shape)) 
        b = tf.get_variable(name='bias',initializer=tf.truncated_normal([1,1,1,1]))
    return w,b

def conv2d(x, k, name=None, strides=(1,1,1,1),pad='SAME'):
    return tf.nn.conv2d(x, k, name=name, strides=strides, padding=pad)

def VI_Block(X, config):
    k    = config.k    # Number of value iterations performed
    ch_i = 2           # Channels in input layer, hardcoded to 2 for now
    ch_h = config.ch_h # Channels in initial hidden layer
    ch_p = config.ch_p # Channels in proxy reward layer
    ch_q = config.ch_q # Channels in q layer (~actions)
    num_actions = config.num_actions
    regularizer = tf.contrib.layers.l2_regularizer(scale=config.vin_regularizer_C)

    with tf.variable_scope('VIN', regularizer=regularizer, dtype=tf.float32):
        bias = tf.get_variable(
            name='bias',
            initializer=tf.truncated_normal([1, 1, 1, ch_h], stddev=0.01))
        # weights from inputs to q layer (~reward in Bellman equation)
        w0 = tf.get_variable(
            name='w0',
            initializer=tf.truncated_normal([3, 3, ch_i, ch_h], stddev=0.01))
        w1 = tf.get_variable(
            name='w1',
            initializer=tf.truncated_normal([1, 1, ch_h, ch_p], stddev=0.01))
        w = tf.get_variable(
            name='w',
            initializer=tf.truncated_normal([3, 3, ch_p, ch_q], stddev=0.01))
        # feedback weights from v layer into q layer
        # (Similar to the transition probabilities in Bellman equation)
        w_fb = tf.get_variable(
            name='w_fb',
            initializer=tf.truncated_normal([3, 3, 1, ch_q], stddev=0.01))
        w_o = tf.get_variable(
            name='w_o',
            initializer=tf.truncated_normal([ch_q, num_actions], stddev=0.01))

    # initial conv layer over image+reward prior
    h = tf.nn.relu(conv2d(X, w0, name="h0") + bias)

    r = conv2d(h, w1, name="r")
    q = conv2d(r, w, name="q")
    v = tf.reduce_max(q, axis=3, keep_dims=True, name="v")

    wwfb = tf.concat([w, w_fb], 2)

    for i in range(0, k-1):
        rv = tf.concat([r, v], 3)
        q = conv2d(rv, wwfb, name="q")
        v = tf.reduce_max(q, axis=3, keep_dims=True, name="v")

    # do one last convolution
    q = conv2d(tf.concat([r, v], 3), wwfb, name="q")
    q_out = tf.reshape(q, [-1, ch_q])

    # add logits
    logits = tf.matmul(q_out, w_o, name="logits")

    # softmax output weights
    output = tf.nn.softmax(logits, name="output")
    return Model(logits, output)

def calculate_action_distribution(nn, bsize, ch_q, name=None):
    """Adds TF code to calculate the % distributions of actions predicted
    nn: any tensor that represents the q-values for the grid
    returns: tensor of [bsize, ch_q] where each column is % of that action predicted
            for a given batch"""

    # nn is of shape [bsize, imsize, imsize, ch_q]
    predictions = tf.argmax(nn, axis=-1,name='predict_table',output_type=tf.int32)
    predictions = tf.reshape(predictions, [bsize, -1])
    distributions = []

    for i in range(ch_q):
        index = tf.constant(i,dtype=tf.int32)
        # [bsize, ch_q] --> [bsize]
        tensor2sum = tf.cast(tf.equal(predictions,index),dtype=tf.int32)
        distributions.append(tf.reduce_sum(tensor2sum,name='indexdist_{}'.format(i)))

    if not name:
        name = 'action_distributions'
    distributions = tf.stack([distributions], axis=-1, name=name)
    return distributions

def tf_value_iter(X, config):
    return tf_value_iter_no_config(X, config.ch_q, config.imsize, config.batchsize, config.num_iters, config.gamma, config.noise)

# Helper Functions for tf_value_iter
def tf_value_iter_no_config(X, ch_q, imsize, bsize, num_iters, discount, noise, vi_beta=1):
    """
    Note: this algorithm may need additional attention to the way rewards are inferred
            Meaning, that batch updates may be especially important, or simultaneous updates
    Currently: [image, reward] --> VI -->
                Qvals (image_dim, image_dim, num_actions) -->
                softmax on state to predict action
    Also: this algorithm needs the transition probabilities of an MDP
        but it can use walls to tell if pass/no pass, and reward arr

    :param X: Stacked channel-wise array of image and reward tensors
    :param config: Tensorflow config flags
    :return: Q-value table
    """
    print("VI is being performed with {} iterations.".format(num_iters))
    print("VI is assuming there are 5 actions: up, down, left, right, stay (ordered)")

    # TODO(rohinmshah): Make the living reward a flag
    living_reward = -0.01
    p1, p2 = 1 - noise, noise / 2
    # Unpack X tensor
    reward_for_stay = X[:,1:-1,1:-1,1]
    reward = reward_for_stay + living_reward
    reward = tf.stack([reward, reward, reward, reward, reward_for_stay], axis=-1)
    walls = tf.cast(X[:,:,:,0], tf.bool)
    values = tf.zeros([bsize, imsize, imsize])
    for _ in range(num_iters + 1):
        vals = discount * values
        stay_vals = vals[:, 1:-1, 1:-1]
        n_vals = tf.where(walls[:,:-2,1:-1], stay_vals, vals[:,:-2,1:-1])
        s_vals = tf.where(walls[:,2:,1:-1], stay_vals, vals[:,2:,1:-1])
        e_vals = tf.where(walls[:,1:-1,2:], stay_vals, vals[:,1:-1,2:])
        w_vals = tf.where(walls[:,1:-1,:-2], stay_vals, vals[:,1:-1,:-2])
        n_qvals = p1 * n_vals + p2 * (e_vals + w_vals)
        s_qvals = p1 * s_vals + p2 * (e_vals + w_vals)
        e_qvals = p1 * e_vals + p2 * (n_vals + s_vals)
        w_qvals = p1 * w_vals + p2 * (n_vals + s_vals)
        qvalues = tf.stack([n_qvals, s_qvals, e_qvals, w_qvals, stay_vals], axis=-1)
        qvalues += reward
        values = tf.reduce_logsumexp(vi_beta * qvalues, axis=-1, name="v") / vi_beta
        paddings = tf.constant([[0, 0], [1, 1], [1, 1]])
        values = tf.pad(values, paddings, "CONSTANT")

    paddings = tf.constant([[0, 0], [1, 1], [1, 1], [0, 0]])
    qvalues = tf.pad(qvalues, paddings, "CONSTANT")
    qvalues = tf.reshape(qvalues, [-1,ch_q])
    return Model(qvalues, tf.nn.softmax(qvalues,name='output'))

def activation(tensor):
    return 1 - tensor

def mask(values, masking):
    return tf.multiply(values, masking)

def negative_mask_values(values, wall_mask):
    """Subtracts -1000 from where the zero mask would put 0s"""
    zero_mask = mask(values, wall_mask)

    neg_mask = -1000*activation(wall_mask)
    return zero_mask + neg_mask

def convolve(values, kernel):
    # values = tf.expand_dims(values, axis=-1)
    return tf.nn.conv2d(values, kernel, strides=(1,1,1,1), padding="SAME")
