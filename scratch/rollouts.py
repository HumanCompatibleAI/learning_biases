# Visualizing Rollouts
# --------------------------------------------------------
# this is space to create rollouts
#
# TODO: capture sum of rewards of every rollout
# 		for agent comparison
from gridworld import GridworldMdp, Direction
from mdp_interface import Mdp
from gridworld_data import print_training_example
import numpy as np
import agents
import tensorflow as tf
# imsize = 16
# pr_wall = 0.05
# pr_reward = 0.0

grid = 		[ 'XXXXXXXXX',
              'X9XAX   X',
              'X X X   X',
              'X       X',
              'XXXXXXXXX']

preference_grid = [
 'XXXXXXXXXXXXXX',
 'XXXXXX4XXXXXXX',
 'XXXXXX XXXXXXX',
 'XXXXX     XXXX',
 'XXXXX XXX  2XX',
 'XXXXX XXX XXXX',
 'XXXX1 XXX XXXX',
 'XXXXX XXX XXXX',
 'XXXXX XXX XXXX',
 'XXXXX XXX XXXX',
 'X1        XXXX',
 'XXXXX XX1XXXXX',
 'XXXXXAXXXXXXXX',
 'XXXXXXXXXXXXXX'
 ]

mdp = GridworldMdp(preference_grid)
# mdp = GridworldMdp.generate_random(imsize, imsize, pr_wall, pr_reward)
# agent = agents.OptimalAgent()
agent = agents.SophisticatedTimeDiscountingAgent(2, 0.01)
agent.set_mdp(mdp)

env = Mdp(mdp)
trajectory = env.perform_rollout(agent,max_iter=20, print_step=1000)
print_training_example(mdp, trajectory)
print(agent.reward)


# class NeuralAgent(Agent):

# 	def __init__(self, save_dir):
# 		Agent.__init__(self)
# 		self.sess = tf.Session(graph=tf.Graph())
# 		tf.saved_model.loader.load(sess, ['train'], '/tmp/planner-vin/model/')

# 	def get_action(self, state):
# 		walls, rewards, _ = self.mdp.convert_to_numpy_input()
# 		fd = {
# 			'image:0': walls,
# 			'reward:0', rewards,
# 			'S1:0', [state[0]],
# 			'S2:0', [state[1]],
# 		}
# 		
# 		self.sess.run([], fd)

# with tf.Session(graph=tf.Graph()) as sess:
# # sess = tf.Session(graph=tf.Graph())
# 	tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], 
# 		'/tmp/planner-vin/model10/model/')
# 	action_dist = sess.graph.get_operation_by_name('output').values()

# 	walls, rewards, _ = mdp.convert_to_numpy_input()
# 	walls, rewards = np.array([list(walls)]*12), np.array([list(rewards)]*12)

# 	x,y = env.get_current_state()
# 	x,y = [[x]*10]*12, [[y]*10]*12

# 	fd = {'image:0': walls,
# 		'reward:0': rewards,
# 		'S1:0': x,
# 		'S2:0': y}
# 	out = sess.run(action_dist, feed_dict=fd)[0]
# 	action = np.argmax(out[0,:])







