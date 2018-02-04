import unittest
from agent_interface import Agent
from agent_runner import run_agent, evaluate_proxy
from gridworld import GridworldMdp, GridworldEnvironment, Direction
import numpy as np
import pdb

class DirectionalAgent(Agent):
    """An agent that goes in a specific direction or exits.

    This agent only plays grid worlds.
    """
    def __init__(self, direction, gamma=1.0):
        Agent.__init__(self, gamma)
        self.default_action = direction

    def get_action(self, state):
        if self.default_action not in self.mdp.get_actions(state):
            return Direction.EXIT
        return self.default_action

class TestAgentRunner(unittest.TestCase):
    def test_run_agent(self):
        grid = ['XXXXXXXXX',
                'X2X X   X',
                'X X X   X',
                'XA     9X',
                'XXXXXXXXX']

        # Keep going east to get the 9 reward
        mdp1 = GridworldMdp(grid, living_reward=0)
        env1 = GridworldEnvironment(mdp1)
        # Make sure that run_agent resets the environment
        env1.perform_action(Direction.NORTH)
        east = Direction.EAST
        agent1 = DirectionalAgent(east)
        agent1.set_mdp(mdp1)
        trajectory = [((x, 3), east, (x + 1, 3), 0) for x in range(1, 7)]
        trajectory.append(((7, 3), Direction.EXIT, mdp1.terminal_state, 9))
        self.assertEqual(run_agent(agent1, env1, episode_length=10), trajectory)

        # Keep going north to get the 2 reward
        mdp2 = GridworldMdp(grid, living_reward=0)
        env2 = GridworldEnvironment(mdp2)
        north = Direction.NORTH
        agent2 = DirectionalAgent(north)
        agent2.set_mdp(mdp2)
        trajectory = [((1, y), north, (1, y - 1), 0) for y in range(3, 1, -1)]
        trajectory.append(((1, 1), Direction.EXIT, mdp2.terminal_state, 2))
        self.assertEqual(run_agent(agent2, env2, episode_length=10), trajectory)

        # Keep going south, never getting a reward
        mdp3 = GridworldMdp(grid, living_reward=-0.1)
        env3 = GridworldEnvironment(mdp3)
        south = Direction.SOUTH
        agent3 = DirectionalAgent(south)
        agent3.set_mdp(mdp3)
        trajectory = [((1, 3), south, (1, 3), -0.1)] * 10
        self.assertEqual(run_agent(agent3, env3, episode_length=10), trajectory)

    def test_evaluate_proxy(self):
        walls  = [[1, 1, 1, 1, 1],
                  [1, 0, 0, 0, 1],
                  [1, 0, 1, 0, 1],
                  [1, 0, 0, 0, 1],
                  [1, 1, 1, 1, 1]]
        reward = [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 3, 0, 0, 0],
                  [0, 0, 0, 1, 0],
                  [0, 0, 0, 0, 0]]
        proxy  = [[0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0],
                  [0, 1, 0, 0.1, 0],
                  [0, 0, 0, 3, 0],
                  [0, 0, 0, 0, 0]]
        start_state = (3, 1)
        walls, reward, proxy = map(np.array, (walls, reward, proxy))
        self.assertEqual(evaluate_proxy(walls, start_state, proxy, reward),
                         0.98 / 2.97)

if __name__ == '__main__':
    unittest.main()
