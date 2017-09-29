import unittest
from agent_interface import Agent
from agent_runner import run_agent
from agents import OptimalAgent, NaiveTimeDiscountingAgent, SophisticatedTimeDiscountingAgent, MyopicAgent
from gridworld import GridworldMdp, GridworldEnvironment, Direction

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
        agent = OptimalAgent(mdp, num_iters=20)
        self.assertAlmostEqual(agent.values[mdp.get_start_state()], 8.2)
        actions, reward = self.run_on_env(agent, env)
        self.assertAlmostEqual(reward, 8.2)
        self.assertEqual(actions, [s, s, w, w, w, w, n, n, exit_act])

        mdp = GridworldMdp(grid, living_reward=0)
        env = GridworldEnvironment(mdp)
        agent = OptimalAgent(mdp, gamma=0.5, num_iters=20)
        self.assertAlmostEqual(agent.values[mdp.get_start_state()], 0.125)
        actions, reward = self.run_on_env(agent, env, gamma=0.5)
        self.assertAlmostEqual(reward, 0.125)
        self.assertEqual(actions, [s, s, e, e, exit_act])

    def test_time_discounting_agents(self):
        grid = [['X', 'X', 'X', 'X', 'X', 'X', 'X'],
                ['X', ' ', ' ', ' ', ' ', 'A', 'X'],
                ['X', ' ', 'X',  4 , 'X', ' ', 'X'],
                ['X', 9.5, 'X', ' ', 'X',  4 , 'X'],
                ['X', 'X', 'X', 'X', 'X', 'X', 'X']]
        n, s, e, w, exit_act = self.all_actions

        mdp = GridworldMdp(grid, living_reward=-0.001)
        env = GridworldEnvironment(mdp)

        agent = OptimalAgent(mdp, num_iters=20)
        actions, reward = self.run_on_env(agent, env)
        self.assertAlmostEqual(reward, 9.494)
        self.assertEqual(actions, [w, w, w, w, s, s, exit_act])

        agent = NaiveTimeDiscountingAgent(mdp, 10, 1, num_iters=20)
        actions, reward = self.run_on_env(agent, env)
        self.assertAlmostEqual(reward, 3.997)
        self.assertEqual(actions, [w, w, s, exit_act])

        agent = SophisticatedTimeDiscountingAgent(mdp, 10, 1, num_iters=20)
        actions, reward = self.run_on_env(agent, env)
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

        agent = OptimalAgent(mdp, num_iters=20)
        actions, reward = self.run_on_env(agent, env)
        self.assertAlmostEqual(reward, 10.242)
        self.assertEqual(actions, [s, w, w, w, w, w, s, w, exit_act])

        agent = NaiveTimeDiscountingAgent(mdp, 10, 1, num_iters=20)
        actions, reward = self.run_on_env(agent, env)
        self.assertAlmostEqual(reward, 6.993)
        self.assertEqual(actions, [s, w, w, w, w, w, w, exit_act])

        agent = SophisticatedTimeDiscountingAgent(mdp, 10, 1, num_iters=20)
        actions, reward = self.run_on_env(agent, env)
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

        agent = OptimalAgent(mdp, num_iters=20)
        actions, reward = self.run_on_env(agent, env)
        self.assertAlmostEqual(reward, 8.4)
        self.assertEqual(actions, [e, e, e, e, e, s, exit_act])

        agent = MyopicAgent(mdp, 6, num_iters=20)
        start_mu = agent.extend_state_to_mu(mdp.get_start_state())
        self.assertAlmostEqual(agent.values[start_mu], 1.5)
        actions, reward = self.run_on_env(agent, env)
        self.assertAlmostEqual(reward, 8.2)
        self.assertEqual(actions, [s, s, e, e, e, e, e, n, exit_act])

if __name__ == '__main__':
    unittest.main()
