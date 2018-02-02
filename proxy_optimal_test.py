# Visualizing Rollouts
# --------------------------------------------------------
# this is space to create rollouts
#
# TODO: capture sum of rewards of every rollout
# 		for agent comparison
from gridworld import GridworldMdp, GridworldEnvironment, Direction
from gridworld_data import print_training_example
import numpy as np
import agents
import tensorflow as tf
from agent_runner import run_agent_proxy

walls =              [['X', 'X', 'X', 'X', 'X'],
                      ['X', ' ', ' ', 'A', 'X'],
                      ['X', ' ', 'X', ' ', 'X'],
                      ['X', ' ', ' ', ' ', 'X'],
                      ['X', 'X', 'X', 'X', 'X']]

reward =              [[0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 3, 0, 0, 0],
                      [0, 0, 0, 1, 0],
                      [0, 0, 0, 0, 0]]

proxy =              [[0, 0, 0, 0, 0],
                      [0, 0, 0, 0, 0],
                      [0, 1, 0, 0.1, 0],
                      [0, 0, 0, 3, 0],
                      [0, 0, 0, 0, 0]]
trajectory, proxy_reward, true_reward = run_agent_proxy(walls, proxy, reward)
print(trajectory)
print("proxy:", proxy_reward, "true reward:", true_reward)

"""

grid = 	    [ 'XXXXXXXXX',
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
"""
