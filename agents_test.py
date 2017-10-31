import unittest
import numpy as np
from agent_interface import Agent
from agent_runner import run_agent
from agents import OptimalAgent, NaiveTimeDiscountingAgent, SophisticatedTimeDiscountingAgent, MyopicAgent
from gridworld import GridworldMdp, GridworldEnvironment, Direction
from utils import Distribution

class TestAgents(unittest.TestCase):
    def setUp(self):
        self.all_actions = [
            Direction.NORTH,
            Direction.SOUTH,
            Direction.EAST,
            Direction.WEST,
            Direction.EXIT
        ]

    def run_on_env(self, agent, env, gamma=1.0):
        trajectory = run_agent(agent, env)
        actions = [action for _, action, _, _ in trajectory]
        rewards = [reward for _, _, _, reward in trajectory]
        total_reward = 0.0
        for reward in rewards[::-1]:
            total_reward = reward + gamma * total_reward
        return actions, total_reward
        

    def test_optimal_agent(self):
        grid = ['XXXXXXXXX',
                'X9X6XA  X',
                'X X X XXX',
                'X      2X',
                'XXXXXXXXX']
        n, s, e, w, exit_act = self.all_actions

        mdp = GridworldMdp(grid, living_reward=-0.1)
        env = GridworldEnvironment(mdp)
        agent = OptimalAgent(num_iters=20)
        agent.set_mdp(mdp)

        # Values
        start_state = mdp.get_start_state()
        self.assertAlmostEqual(agent.values[start_state], 8.2)

        # Action distribution
        action_dist = agent.get_action_distribution(start_state)
        self.assertEqual(action_dist, Distribution({s : 1}))

        # Trajectory
        actions, reward = self.run_on_env(agent, env)
        self.assertAlmostEqual(reward, 8.2)
        self.assertEqual(actions, [s, s, w, w, w, w, n, n, exit_act])

        # Same thing, but with a discount factor
        mdp = GridworldMdp(grid, living_reward=0)
        env = GridworldEnvironment(mdp)
        agent = OptimalAgent(gamma=0.5, num_iters=20)
        agent.set_mdp(mdp)

        # Values
        self.assertAlmostEqual(agent.values[start_state], 0.125)

        # Action distribution
        action_dist = agent.get_action_distribution(start_state)
        self.assertEqual(action_dist, Distribution({s : 1}))

        # Trajectory
        actions, reward = self.run_on_env(agent, env, gamma=0.5)
        self.assertAlmostEqual(reward, 0.125)
        self.assertEqual(actions, [s, s, e, e, exit_act])

        # Same thing, but with Boltzmann rationality
        agent = OptimalAgent(beta=1, gamma=0.5, num_iters=20)
        agent.set_mdp(mdp)

        # Values
        middle_state = (2, 3)
        self.assertAlmostEqual(agent.values[start_state], 0.125)
        self.assertAlmostEqual(agent.values[middle_state], 1.125)

        # Action distribution
        dist = agent.get_action_distribution(start_state).get_dict()
        nprob, sprob, eprob, wprob = dist[n], dist[s], dist[e], dist[w]
        for p in [nprob, sprob, eprob, wprob]:
            self.assertTrue(0 < p < 1)
        self.assertEqual(nprob, wprob)
        self.assertTrue(sprob > nprob)
        self.assertTrue(nprob > eprob)
        dist = agent.get_action_distribution(middle_state).get_dict()
        nprob, sprob, eprob, wprob = dist[n], dist[s], dist[e], dist[w]
        for p in [nprob, sprob, eprob, wprob]:
            self.assertTrue(0 < p < 1)
        self.assertEqual(nprob, sprob)
        self.assertTrue(wprob > eprob)
        self.assertTrue(eprob > nprob)

    def test_time_discounting_agents(self):
        grid = [['X', 'X', 'X', 'X', 'X', 'X', 'X'],
                ['X', ' ', ' ', ' ', ' ', 'A', 'X'],
                ['X', ' ', 'X',  4 , 'X', ' ', 'X'],
                ['X', 9.5, 'X', ' ', 'X',  4 , 'X'],
                ['X', 'X', 'X', 'X', 'X', 'X', 'X']]
        n, s, e, w, exit_act = self.all_actions

        mdp = GridworldMdp(grid, living_reward=-0.001)
        env = GridworldEnvironment(mdp)

        optimal_agent = OptimalAgent(num_iters=20)
        optimal_agent.set_mdp(mdp)
        actions, reward = self.run_on_env(optimal_agent, env)
        self.assertAlmostEqual(reward, 9.494)
        self.assertEqual(actions, [w, w, w, w, s, s, exit_act])

        naive_agent = NaiveTimeDiscountingAgent(10, 1, num_iters=20)
        naive_agent.set_mdp(mdp)
        actions, reward = self.run_on_env(naive_agent, env)
        self.assertAlmostEqual(reward, 3.997)
        self.assertEqual(actions, [w, w, s, exit_act])

        soph_agent = SophisticatedTimeDiscountingAgent(10, 1, num_iters=20)
        soph_agent.set_mdp(mdp)
        actions, reward = self.run_on_env(soph_agent, env)
        self.assertAlmostEqual(reward, 3.998)
        self.assertEqual(actions, [s, s, exit_act])

        val = 10.25  # Needs to be in (10, 10.5)
        grid = [['X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X'],
                ['X', ' ', ' ', ' ', ' ', ' ', 'X', 'A', 'X'],
                ['X', '7', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
                ['X', val, ' ', 'X', 'X', 'X', 'X', ' ', 'X'],
                ['X', 'X', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
                ['X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X']]
        mdp = GridworldMdp(grid, living_reward=-0.001)
        env = GridworldEnvironment(mdp)

        optimal_agent.set_mdp(mdp)
        actions, reward = self.run_on_env(optimal_agent, env)
        self.assertAlmostEqual(reward, 10.242)
        self.assertEqual(actions, [s, w, w, w, w, w, s, w, exit_act])

        naive_agent.set_mdp(mdp)
        actions, reward = self.run_on_env(naive_agent, env)
        self.assertAlmostEqual(reward, 6.993)
        self.assertEqual(actions, [s, w, w, w, w, w, w, exit_act])

        soph_agent.set_mdp(mdp)
        actions, reward = self.run_on_env(soph_agent, env)
        self.assertAlmostEqual(reward, 10.24)
        self.assertEqual(actions, [s, s, s, w, w, w, w, w, n, w, exit_act])

    def test_myopic_agent(self):
        grid = ['XXXXXXXX',
                'XA     X',
                'X XXXX9X',
                'X      X',
                'X X2   X',
                'XXXXXXXX']
        n, s, e, w, exit_act = self.all_actions

        mdp = GridworldMdp(grid, living_reward=-0.1)
        env = GridworldEnvironment(mdp)

        optimal_agent = OptimalAgent(num_iters=20)
        optimal_agent.set_mdp(mdp)
        actions, reward = self.run_on_env(optimal_agent, env)
        self.assertAlmostEqual(reward, 8.4)
        self.assertEqual(actions, [e, e, e, e, e, s, exit_act])

        myopic_agent = MyopicAgent(6, num_iters=20)
        myopic_agent.set_mdp(mdp)
        start_mu = myopic_agent.extend_state_to_mu(mdp.get_start_state())
        self.assertAlmostEqual(myopic_agent.values[start_mu], 1.5)
        actions, reward = self.run_on_env(myopic_agent, env)
        self.assertAlmostEqual(reward, 8.2)
        self.assertEqual(actions, [s, s, e, e, e, e, e, n, exit_act])

if __name__ == '__main__':
    unittest.main()
