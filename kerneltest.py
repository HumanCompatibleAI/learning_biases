import tensorflow as tf
import numpy as np
from model import tf_value_iter_no_config
from agents import OptimalAgent
from gridworld.gridworld import GridworldMdp
sess = tf.InteractiveSession()

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
agent_start = (1, 3)
mdp = GridworldMdp.from_numpy_input(walls.astype(np.float32),reward.astype(np.float32),start_state=agent_start)
imsize = walls.shape[0]
discount = 0.9
num_iters = 50

def test_model(wall_tf, reward_tf, alg):
    return alg(wall_tf, reward_tf)

def tf_value_iter_model(wall_tf, reward_tf):
    a = tf.reshape(wall_tf, [1, imsize, imsize])
    b = tf.reshape(reward_tf, [1, imsize, imsize])
    X = tf.stack([a, b],axis=-1)
    qvals = tf_value_iter_no_config(X, ch_q=5, imsize=imsize, bsize=1, num_iters=num_iters, discount=0.9)
    return qvals.logits

def castAgentValuesToNumpy(agent_dict):
    """Tranposes the keys of the agent's value dictionary to retrieve the true array"""
    values = np.zeros((imsize,imsize))
    for key in agent_dict:
        values[key[1],key[0]] = agent_dict[key]
    return values

def run_test(walls, reward):
    """Runs test on given walls & rewards
    walls, reward: 2d numpy arrays (numbers)"""

    agent = OptimalAgent(num_iters=num_iters)
    agent.set_mdp(mdp)
    true_values = castAgentValuesToNumpy(agent.values)

    wall_tf = tf.placeholder(shape=(imsize,imsize),dtype=tf.float32)
    reward_tf = tf.placeholder(tf.float32, shape=(imsize, imsize))
    q_vals = test_model(wall_tf, reward_tf, tf_value_iter_model)
    out = sess.run(q_vals, feed_dict={wall_tf:walls, reward_tf:reward})
    out = np.reshape(out, (imsize*imsize,5))
    predicted_values = np.max(out, axis=1).reshape((imsize,imsize))


    compareValues(true_values, predicted_values)
    visualizeValueDiff(true_values,predicted_values)

def compareValues(true_values, predicted_values):
    # true_values[i,j] = 0 --> walls[i,j] = 1
    # Only compare values where
    true = true_values[true_values != 0]
    predicted = predicted_values[true_values != 0]

    tol = 1e-1
    for i in range(3):
        values_are_close = np.allclose(true, predicted,atol=tol,equal_nan=True)
        print("Predicted values are within {} of values: {}".format(tol, values_are_close))
        tol = tol/10


def visualizeValueDiff(true_values, predicted_values):
    print("-"*20)
    print("True values:")
    print(true_values)
    print("\nPredicted values:")
    print(predicted_values)
    print("\nRelative difference:")
    print(np.array_str(predicted_values-true_values,precision=2))

run_test(walls, reward)
