# Code taken from https://github.com/TheAbhiKumar/tensorflow-value-iteration-networks

import numpy as np
import tensorflow as tf

def conv2d(x, k, name=None):
    return tf.nn.conv2d(x, k, name=name, strides=(1, 1, 1, 1), padding='SAME')

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
            initializer=tf.zeros([1, 1, 1, ch_h]))
            #initializer=tf.truncated_normal([1, 1, 1, ch_h], stddev=0.01))
        # weights from inputs to q layer (~reward in Bellman equation)
        w0_val = np.zeros([3, 3, ch_i, ch_h])
        # Identify positive rewards
        w0_val[1,1,:,0] = [0, 1]
        # Identify negative rewards and walls
        w0_val[1,1,:,1] = [10, -1]
        w0 = tf.get_variable(
            name='w0',
            initializer=tf.constant(w0_val, dtype=tf.float32))
            #initializer=tf.truncated_normal([3, 3, ch_i, ch_h], stddev=0.01))
        w1_val = np.zeros([1, 1, ch_h, 2])
        # Combine positive and negative rewards into a single reward function
        w1_val[0,0,0,0] = 1
        w1_val[0,0,1,0] = -1
        # Indicator that says that this position is either a reward or a wall
        w1_val[0,0,1,1] = 1
        w1 = tf.get_variable(
            name='w1',
            initializer=tf.constant(w1_val, dtype=tf.float32))
        w_val = np.zeros([3, 3, 2, ch_q])
        # If in a reward square or wall square, must choose exit
        w_val[1,1,0,4] = 1
        w_val[1,1,1,:] = [-100, -100, -100, -100, 0]
        w = tf.get_variable(
            name='w',
            initializer=tf.constant(w_val, dtype=tf.float32))
        # feedback weights from v layer into q layer
        # (Similar to the transition probabilities in Bellman equation)
        w_fb_val = np.zeros([3, 3, 1, ch_q])
        w_fb_val[0,1,0,0] = 0.999
        w_fb_val[2,1,0,1] = 0.999
        w_fb_val[1,2,0,2] = 0.999
        w_fb_val[1,0,0,3] = 0.999
        w_fb = tf.get_variable(
            name='w_fb',
            initializer=tf.constant(w_fb_val, dtype=tf.float32))
        w_o_val = np.zeros([ch_q, num_actions])
        w_o_val[0,0] = 1
        w_o_val[1,1] = 0.99999
        w_o_val[2,2] = 0.99998
        w_o_val[3,3] = 0.99997
        w_o_val[4,4] = 1
        w_o = tf.get_variable(
            name='w_o',
            initializer=tf.constant(w_o_val, dtype=tf.float32))

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
    q_all = q

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
    return logits, output, (r, q_out, q_all)
    # TODO: Put this back in
    # return logits, output
