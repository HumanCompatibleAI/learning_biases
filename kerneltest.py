import tensorflow as tf
import numpy as np
import pdb
"""
1) activate the walls array
2) mask values with activated walls array
3) convolve over masked value array
"""

sess = tf.InteractiveSession()
# For future advanced testing
# walls  = [[1, 1, 1, 1, 1],
#           [1, 0, 0, 0, 1],
#           [1, 0, 1, 0, 1],
#           [1, 0, 0, 0, 1],
#           [1, 1, 1, 1, 1]]
# reward = [[0, 0, 0, 0, 0],
#           [0, 0, 0, 0, 0],
#           [0, 3, 0, 0, -9],
#           [0, 0, 0, 1, 0],
#           [0, 0, 0, 0, 0]]
#
# walls = np.array(walls)
# reward = np.array(reward)

deterministic_kernel = np.load("convkernel.npy")

num_actions = 5
discount = 0.9
walls  = np.zeros((3,3))
reward = np.zeros((3,3))
reward[1,1] = 1
imsize = walls.shape[0]
print("Reward")
print(reward)

imsize = walls.shape[0]
wall_tf = tf.placeholder(shape=(imsize,imsize),dtype=tf.float32)
reward_tf = tf.placeholder(tf.float32, shape=(imsize, imsize))

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
con_out = sess.run(q_vals, feed_dict={wall_tf:walls, reward_tf:reward})
con_out = np.reshape(con_out, (imsize*imsize,5))
con_out = np.argmax(con_out, axis=1)
print(con_out)
# Should print something like
# [1, 1, 1,
#  2, 4, 3,
#  0, 0, 0]
# 1 - Up, 2 - East, 4 - Stay, 3 - West, 0 - North
