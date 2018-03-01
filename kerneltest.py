import tensorflow as tf
import numpy as np
from model import tf_value_iter_no_config
import pdb
"""
1) activate the walls array
2) mask values with activated walls array
3) convolve over masked value array
"""

sess = tf.InteractiveSession()
# For future advanced testing
walls  = [[1, 1, 1, 1, 1],
          [1, 0, 0, 0, 1],
          [1, 0, 0, 0, 1],
          [1, 0, 0, 0, 1],
          [1, 1, 1, 1, 1]]
reward = [[0, 0, 0, 0, 0],
          [0, 0, 0, 1, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0]]


walls = np.array(walls)
reward = np.array(reward)
imsize = walls.shape[0]
discount = 0.9
num_iters = 5
# Should print something like
# [1, 1, 1,
#  2, 4, 3,
#  0, 0, 0]
# 1 - Up, 2 - East, 4 - Stay, 3 - West, 0 - North


def test_model(wall_tf, reward_tf, alg):
    return alg(wall_tf, reward_tf)

def kernel_test_hand_coded_model(wall_tf, reward_tf):
    deterministic_kernel = np.load('convkernel.npy')
    def activation(tensor):
        return 1 - tensor

    wall_mask = activation(wall_tf)

    # Testing the wall activation
    # wall_out = sess.run(activation_out, feed_dict={wall_tf: walls})

    def mask(values, masking):
        return tf.multiply(values, masking)

    masked_values = mask(reward_tf, wall_mask)

    # Testing the value masking
    masked_out = sess.run(masked_values, feed_dict={wall_tf:walls, reward_tf:reward})
    print("Masked Values")
    print(masked_out)

    def convolve(values, kernel):
        values = tf.reshape(values, shape=(1, imsize, imsize, 1))
        return tf.nn.conv2d(values, kernel, strides=(1,1,1,1), padding="SAME")

    q_vals = convolve(masked_values, deterministic_kernel)

    # Test convolution of masking
    con_out = sess.run(q_vals, feed_dict={wall_tf:walls, reward_tf:reward})
    con_out = np.reshape(con_out, (imsize*imsize,5))
    con_out = np.argmax(con_out, axis=0)
    print(con_out,'\n')
    # Should print [7,1,3,5,4]

    print("Testing 2 convolutions as 2 iterations\n")
    # Test multiple convolutions
    vals = discount * tf.reduce_max(q_vals,reduction_indices=[-1]) + masked_values
    masked_values = mask(vals, wall_mask)
    #   test q_vals --> values
    test = sess.run(masked_values, feed_dict={wall_tf:walls, reward_tf:reward})
    print("Values after 1 conv:")
    print(test, '\n')
    #    test values --> q_vals
    q_vals = convolve(vals, deterministic_kernel)
    return q_vals

def tf_value_iter_model(wall_tf, reward_tf):
    a = tf.reshape(wall_tf, [1, imsize, imsize])
    b = tf.reshape(reward_tf, [1, imsize, imsize])
    X = tf.stack([a, b],axis=-1)
    qvals = tf_value_iter_no_config(X, ch_q=5, imsize=imsize, bsize=1, num_iters=num_iters)
    return qvals.output_probs

def run_test(walls, reward):
    """walls, reward: 2d arrays
    Agent's location must be specified in this function"""
    print("Reward")
    print(reward)

    imsize = walls.shape[0]
    wall_tf = tf.placeholder(shape=(imsize,imsize),dtype=tf.float32)
    reward_tf = tf.placeholder(tf.float32, shape=(imsize, imsize))

    q_vals = test_model(wall_tf, reward_tf, tf_value_iter_model)
    con_out = sess.run(q_vals, feed_dict={wall_tf:walls, reward_tf:reward})
    con_out = np.reshape(con_out, (imsize*imsize,5))
    con_out = np.argmax(con_out, axis=1)
    print(con_out.reshape((imsize,imsize)))

run_test(walls, reward)