import unittest
from agent_interface import Agent
from agent_runner import run_agent, evaluate_proxy
from gridworld import GridworldMdp, Direction
from mdp_interface import Mdp
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
        return self.default_action

class TestAgentRunner(unittest.TestCase):
    def test_run_agent(self):
        grid = ['XXXXXXXXX',
                'X X2X   X',
                'X X X   X',
                'X 4A   9X',
                'XXXXXXXXX']

        # Keep going east to get the 9 reward
        mdp1 = GridworldMdp(grid, living_reward=0)
        env1 = Mdp(mdp1)
        # Make sure that run_agent resets the environment
        env1.perform_action(Direction.NORTH)
        east = Direction.EAST
        agent1 = DirectionalAgent(east)
        agent1.set_mdp(mdp1)
        trajectory = [((x, 3), east, (x + 1, 3), 0) for x in range(3, 7)]
        trajectory.extend([((7, 3), east, (7, 3), 9) for i in range(6)])
        self.assertEqual(run_agent(agent1, env1, episode_length=10), trajectory)

        # Keep going north to get the 2 reward
        mdp2 = GridworldMdp(grid, living_reward=0)
        env2 = Mdp(mdp2)
        north = Direction.NORTH
        agent2 = DirectionalAgent(north)
        agent2.set_mdp(mdp2)
        trajectory = [((3, y), north, (3, y - 1), 0) for y in range(3, 1, -1)]
        trajectory.extend([((3, 1), north, (3, 1), 2) for i in range(8)])
        self.assertEqual(run_agent(agent2, env2, episode_length=10), trajectory)

        # Keep going west, getting the 4 reward and overshooting
        mdp3 = GridworldMdp(grid, living_reward=-0.1)
        env3 = Mdp(mdp3)
        west = Direction.WEST
        agent3 = DirectionalAgent(west)
        agent3.set_mdp(mdp3)
        trajectory = [((3, 3), west, (2, 3), -0.1), ((2, 3), west, (1, 3), 3.9)]
        trajectory.extend([((1, 3), west, (1, 3), -0.1) for i in range(8)])
        self.assertEqual(run_agent(agent3, env3, episode_length=10), trajectory)

        # Keep going south, never getting a reward
        mdp4 = GridworldMdp(grid, living_reward=-0.1)
        env4 = Mdp(mdp4)
        south = Direction.SOUTH
        agent4 = DirectionalAgent(south)
        agent4.set_mdp(mdp4)
        trajectory = [((3, 3), south, (3, 3), -0.1)] * 10
        self.assertEqual(run_agent(agent4, env4, episode_length=10), trajectory)

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
        pct_reward = evaluate_proxy(walls, start_state, proxy, reward, gamma=0.9, episode_length=10)

        def geometric(a, r, n):
            return a * (1 - (r ** n)) / (1 - r)
        reward_from_proxy = geometric(-0.01, 0.9, 2) + geometric(0.81, 0.9, 8)
        reward_from_truth = geometric(-0.01, 0.9, 3) + 3 * geometric(0.729, 0.9, 7)

        expected_pct_reward = reward_from_proxy / reward_from_truth
        self.assertAlmostEqual(pct_reward, expected_pct_reward, places=7)

if __name__ == '__main__':
    unittest.main()
