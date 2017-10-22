# Visualizing Rollouts
# --------------------------------------------------------
# this is space for me to create rollouts and save animation

from gridworld import GridworldMdp, GridworldEnvironment
from gridworld_data import get_random_start_state
import agents

imsize = 16
pr_wall = 0.05
pr_reward = 0.0

grid = 		[ 'XXXXXXXXX',
              'X9X X  AX',
              'X X X   X',
              'X       X',
              'XXXXXXXXX']
mdp = GridworldMdp(grid)
# mdp = GridworldMdp.generate_random(imsize, imsize, pr_wall, pr_reward)
agent = agents.OptimalAgent()
agent.set_mdp(mdp)

env = GridworldEnvironment(mdp)

def get_minibatch():
	state = get_random_start_state(mdp)
	action = agent.get_action(state)
	return state, action

def act():
	state = env.get_current_state()
	action = agent.get_action(state)
	return env.perform_action(action)

# print(env.state)
griddy, printable = env.convert_to_grid()
print(printable)
# print(len(griddy))
