# Visualizing Rollouts
# --------------------------------------------------------
# this is space to create rollouts

from gridworld import GridworldMdp, GridworldEnvironment
import agents

# imsize = 16
# pr_wall = 0.05
# pr_reward = 0.0

grid = 		[ 'XXXXXXXXX',
              'X9XAX   X',
              'X X X   X',
              'X       X',
              'XXXXXXXXX']
mdp = GridworldMdp(grid)
# mdp = GridworldMdp.generate_random(imsize, imsize, pr_wall, pr_reward)
# agent = agents.OptimalAgent()
agent = agents.MyopicAgent(10)
agent.set_mdp(mdp)

env = GridworldEnvironment(mdp)
env.perform_rollout(agent,max_iter=10, print_step=1)