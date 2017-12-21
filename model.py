# Code taken from https://github.com/TheAbhiKumar/tensorflow-value-iteration-networks
import keras 
from keras.layers import Conv2D, Conv2DTranspose, Input
from keras.models import Model
import numpy as np
import tensorflow as tf

def simple_model(X, config):
    """ Creates Conv-Net to run on 2-channel Grid Input (walls, rewards)
    However, to ensure that the entire grid is convolved over, each architecture has to be individually constructed"""

    # HYPERPARAMATERS
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
        print('conv1:',conv.get_shape())
        conv = conv_layer(conv, [2,2,ch_i,ch_i], 'conv2',strides=[1,1,1,1],pad='SAME')
        twopta = convt_layer(conv, [2,2,ch_q,ch_i], 'convt2a',
            [config.batchsize,imsize//2,imsize//2,ch_q],strides=(1,2,2,1),pad='VALID',activation=None)
        third = convt_layer(twopta, [2,2,ch_q,ch_q], 'convt2b',
            final_shape,strides=(1,2,2,1),pad='VALID',activation=None)
    elif imsize == 14:
        first = conv_layer(X, [1,1,ch_i,ch_q], 'conv_0',pad='SAME')

        # 14x14x2 --> 6x6x2
        conv = conv_layer(X, [3,3,ch_i,ch_i], 'conv1', strides=[1,2,2,1],pad='VALID')
        #   6x6x2 --> 14x14x2
        second = convt_layer(conv, [3,3,ch_q,ch_i],'convt1a',final_shape,strides=[1,2,2,1],pad='VALID',activation=None)

        # 6x6x2 --> 2x2x2
        conv = conv_layer(conv, [3,3,ch_i,ch_i], 'conv2', strides=[1,2,2,1],pad='VALID')
        #   2x2x2 --> 7x7x2
        intermed = convt_layer(conv, [4,4,ch_i,ch_i],'convt2a',
            [config.batchsize,6,6,ch_i],strides=[1,3,3,1],pad='VALID',activation=None)
        #   7x7x2 --> 14x14x2
        third = convt_layer(conv, [2,2,ch_q,ch_i],'convt2b',final_shape,strides=[1,2,2,1],pad='VALID',activation=None)

        # 2x2x2 --> 2x2x2
        # conv = conv_layer(conv, [2,2,ch_i,ch_i], 'conv3', strides=[1,1,1,1], pad='SAME')

    assert first.get_shape()==second.get_shape(), ...
    "first has shape:{} while second has shape: {}".format(first.get_shape(),second.get_shape())

    # Take average of the output
    X = (first+second+third)/3
    X = tf.reshape(X, [-1, ch_q])
    # return X, tf.nn.softmax(X, name='output')
    return X, X

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
    q_out = tf.reshape(q, [-1, ch_q])

    # add logits
    logits = tf.matmul(q_out, w_o)
    
    # softmax output weights
    output = tf.nn.softmax(logits, name="output")
    return logits, output
