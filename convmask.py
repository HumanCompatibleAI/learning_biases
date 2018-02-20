import tensorflow as tf
import numpy as np
import pdb
"""
1) activate the walls array
2) mask values with activated walls array
3) convolve over masked value array
"""

sess = tf.InteractiveSession()
walls  = [[1, 1, 1, 1, 1],
          [1, 0, 0, 0, 1],
          [1, 0, 1, 0, 1],
          [1, 0, 0, 0, 1],
          [1, 1, 1, 1, 1]]
reward = [[0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0],
          [0, 3, 0, 0, -9],
          [0, 0, 0, 1, 0],
          [0, 0, 0, 0, 0]]
og_ker = [
    [[0, 0, 0],
    [0, 0, 0],
    [0, 1, 0]],

    [[0, 1, 0],
     [0, 0, 0],
     [0, 0, 0]],

    [[0, 0, 0],
     [1, 0, 0],
     [0, 0, 0]],

    [[0, 0, 0],
     [0, 0, 1],
     [0, 0, 0]],

    [[0, 0, 0],
     [0, 1, 0],
     [0, 0, 0]]
]

# Convolutions work when end = 1 or 3, but not 2, 4, or 5
# Additionally, in order for the convolutions to work, they seem to have to be hardcoded in reverse
# This means that the convolution for NORTH looks like the convolution for SOUTH
# kernel for north = [[0, 0, 0],
#                     [0, 0, 0],
#                     [0, 1, 0]]
# And this will produce [[0, 1, 0],
#                        [0, 0, 0],
#                        [0, 0, 0]]
end = 1
og_ker = list(reversed(og_ker[:end]))
print("og kernel")
print(og_ker)

deterministic_kernel = np.array(og_ker, dtype=np.float32)
# deterministic_kernel = np.stack((ker.reshape((3,3,1)) for ker in deterministic_kernel) ,axis=-1)
# deterministic_kernel = np.reshape(deterministic_kernel, (3,3,1,1))
deterministic_kernel = deterministic_kernel.reshape((3,3,1,end))
print("Kernel")
print(deterministic_kernel)

walls = np.array(walls)
reward = np.array(reward)

walls  = np.zeros((3,3))
reward = np.zeros((3,3))
reward[1,1] = 1
imsize = 3
print("Reward")
print(reward)

imsize = walls.shape[0]
wall_tf = tf.placeholder(shape=(imsize,imsize),dtype=tf.float32)
reward_tf = tf.placeholder(tf.float32, shape=(imsize, imsize))

def activation(tensor):
    return 1 - tensor

activation_out = activation(wall_tf)

# Testing the wall activation
# wall_out = sess.run(activation_out, feed_dict={wall_tf: walls})

def mask(values, masking):
    return tf.multiply(values, masking)

masked_values = mask(reward_tf, activation_out)

# Testing the value masking
masked_out = sess.run(masked_values, feed_dict={wall_tf:walls, reward_tf:reward})
print("Masked Values")
print(masked_out)

def convolve(values, kernel):
    values = tf.reshape(values, shape=(1, imsize, imsize, 1))
    # kernel = tf.reshape(kernel, shape=(3, 3, 1, 5))
    return tf.nn.conv2d(values, kernel, strides=(1,1,1,1), padding="SAME")

convolved_out = convolve(masked_values, deterministic_kernel)

# Test convolution of masking
con_out = sess.run(convolved_out, feed_dict={wall_tf:walls, reward_tf:reward})
con_out = np.reshape(con_out, (end, imsize, imsize))

print("Conv Out")
for im in con_out:
    for row in im:
        print(row)
    print()