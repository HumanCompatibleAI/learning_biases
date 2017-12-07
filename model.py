# Code taken from https://github.com/TheAbhiKumar/tensorflow-value-iteration-networks

import numpy as np
import tensorflow as tf

def simple_model(X, S1, S2, config):
    """ Creates Conv-Net to run on 2-channel Grid Input (walls, rewards)"""

    # HYPERPARAMATERS
    # ---------------------------
    ch_i = 3
    ch_h = config.ch_h  # Number of convolutions to perform
    ch_q = config.ch_q  # Channels in q layer (~actions) 
    imsize = config.imsize
    num_actions = config.num_actions
    state_batch_size = config.statebatchsize
    # regularizer = tf.contrib.l2_regularizer(scale=0.001)

    # ENCODER
    # ---------------------------
    with tf.variable_scope('CNN_ENCODER', dtype=tf.float32):
        w0 = tf.get_variable(
            name='conv_weight_0',
            initializer=tf.truncated_normal([3,3,ch_i,ch_i]))
        b0 = tf.get_variable(
            name='conv_bias_0',
            initializer=tf.truncated_normal([1,1,1,ch_i]))
        # w1 = tf.get_variable(
        #     name='conv_weight_1',
        #     initializer=tf.truncated_normal([3,3,ch_i*4,ch_i-1]))
        # b1 = tf.get_variable(
        #     name='conv_bias_1',
        #     initializer=tf.truncated_normal([1,1,1,ch_i-1]))
        w = tf.get_variable(
            name='conv_weight_final',
            initializer=tf.truncated_normal([1,1,ch_i,ch_q]))
        b = tf.get_variable(
            name='bias_final',
            initializer=tf.truncated_normal([1,1,1,ch_q]))

    # Currently performs single dot product over every channel
    X = conv2d(X,w0)+b0
    X = conv2d(X, w)+b

    # END ENCODING
    # ---------------------------
    return X, tf.nn.softmax(X, name='output')

def conv2d(x, k, name=None, pad='SAME'):
    return tf.nn.conv2d(x, k, name=name, strides=(1, 1, 1, 1), padding=pad)

def VI_Block(X, S1, S2, config):
    k    = config.k    # Number of value iterations performed
    ch_i = 2           # Channels in input layer, hardcoded to 2 for now
    ch_h = config.ch_h # Channels in initial hidden layer
    ch_q = config.ch_q # Channels in q layer (~actions)
    num_actions = config.num_actions
    state_batch_size = config.statebatchsize # k+1 state inputs for each channel
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
            initializer=tf.truncated_normal([1, 1, ch_h, 1], stddev=0.01))
        w = tf.get_variable(
            name='w',
            initializer=tf.truncated_normal([3, 3, 1, ch_q], stddev=0.01))
        # feedback weights from v layer into q layer
        # (Similar to the transition probabilities in Bellman equation)
        w_fb = tf.get_variable(
            name='w_fb',
            initializer=tf.truncated_normal([3, 3, 1, ch_q], stddev=0.01))
        w_o = tf.get_variable(
            name='w_o',
            initializer=tf.truncated_normal([ch_q, num_actions], stddev=0.01))

    # initial conv layer over image+reward prior
    h = conv2d(X, w0, name="h0") + bias

    r = conv2d(h, w1, name="r")
    q = conv2d(r, w, name="q")
    v = tf.reduce_max(q, axis=3, keep_dims=True, name="v")

    for i in range(0, k-1):
        rv = tf.concat([r, v], 3)
        wwfb = tf.concat([w, w_fb], 2)
        q = conv2d(rv, wwfb, name="q")
        v = tf.reduce_max(q, axis=3, keep_dims=True, name="v")

    # do one last convolution
    q = conv2d(tf.concat([r, v], 3), tf.concat([w, w_fb], 2), name="q")

    # CHANGE TO THEANO ORDERING
    # Since we are selecting over channels, it becomes easier to work with
    # the tensor when it is in NCHW format vs NHWC
    q = tf.transpose(q, perm=[0, 3, 1, 2])

    # Select the conv-net channels at the state position (S1,S2).
    # This intuitively corresponds to each channel representing an action, and the convnet the Q function.
    # The tricky thing is we want to select the same (S1,S2) position *for each* channel and for each sample
    # TODO: performance can be improved here by substituting expensive
    #       transpose calls with better indexing for gather_nd
    bs = tf.shape(q)[0]
    rprn = tf.reshape(tf.tile(tf.reshape(tf.range(bs), [-1, 1]), [1, state_batch_size]), [-1])
    ins1 = tf.cast(tf.reshape(S1, [-1]), tf.int32)
    ins2 = tf.cast(tf.reshape(S2, [-1]), tf.int32)
    idx_in = tf.transpose(tf.stack([ins1, ins2, rprn]), [1, 0])
    q_out = tf.gather_nd(tf.transpose(q, [2, 3, 0, 1]), idx_in, name="q_out")

    # add logits
    logits = tf.matmul(q_out, w_o)
    
    # softmax output weights
    output = tf.nn.softmax(logits, name="output")
    return logits, output
