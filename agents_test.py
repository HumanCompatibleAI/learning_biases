import unittest
import numpy as np
import time
import agents
import fast_agents
from agent_interface import Agent
from agent_runner import run_agent, get_reward_from_trajectory
from gridworld import GridworldMdp, Direction
from mdp_interface import Mdp
from utils import Distribution, set_seeds

class TestAgents(unittest.TestCase):
    def setUp(self):
        self.all_actions = Direction.ALL_DIRECTIONS

    def run_on_env(self, agent, env, gamma=0.9, episode_length=10):
        trajectory = run_agent(agent, env, episode_length)
        actions = [action for _, action, _, _ in trajectory]
        return actions, get_reward_from_trajectory(trajectory, gamma)
        

    def optimal_agent_test(self, agent):
        grid = ['XXXXXXXXX',
                'X9X6XA  X',
                'X X X XXX',
                'X      2X',
                'XXXXXXXXX']
        n, s, e, w, stay = self.all_actions

        mdp = GridworldMdp(grid, living_reward=-0.1)
        env = Mdp(mdp)
        agent.set_mdp(mdp)
        start_state = mdp.get_start_state()

        # Action distribution
        action_dist = agent.get_action_distribution(start_state)
        self.assertEqual(action_dist, Distribution({s : 1}))

        # Trajectory
        actions, _ = self.run_on_env(agent, env, gamma=0.95, episode_length=10)
        self.assertEqual(actions, [s, s, w, w, w, w, n, n, stay, stay])

        # Same thing, but with a bigger discount
        mdp = GridworldMdp(grid, living_reward=-0.001)
        env = Mdp(mdp)
        agent = agents.OptimalAgent(gamma=0.5, num_iters=20)
        agent.set_mdp(mdp)
        start_state = mdp.get_start_state()

        # Values
        # Inaccurate because I ignore living reward and we only use 20
        # iterations of value iteration, so only check to 2 places
        self.assertAlmostEqual(agent.value(start_state), 0.25, places=2)

        # Action distribution
        action_dist = agent.get_action_distribution(start_state)
        self.assertEqual(action_dist, Distribution({s : 1}))

        # Trajectory
        actions, reward = self.run_on_env(agent, env, gamma=0.5, episode_length=10)
        # Again approximate comparison since we don't consider living rewards
        self.assertAlmostEqual(reward, (4 - 0.0625) / 16, places=2)
        self.assertEqual(actions, [s, s, e, e, stay, stay, stay, stay, stay, stay])

        # Same thing, but with Boltzmann rationality
        agent = agents.OptimalAgent(beta=1, gamma=0.5, num_iters=20)
        agent.set_mdp(mdp)

        # Action distribution
        dist = agent.get_action_distribution(start_state).get_dict()
        nprob, sprob, eprob, wprob = dist[n], dist[s], dist[e], dist[w]
        for p in [nprob, sprob, eprob, wprob]:
            self.assertTrue(0 < p < 1)
        self.assertEqual(nprob, wprob)
        self.assertTrue(sprob > nprob)
        self.assertTrue(nprob > eprob)

        middle_state = (2, 3)
        dist = agent.get_action_distribution(middle_state).get_dict()
        nprob, sprob, eprob, wprob = dist[n], dist[s], dist[e], dist[w]
        for p in [nprob, sprob, eprob, wprob]:
            self.assertTrue(0 < p < 1)
        self.assertEqual(nprob, sprob)
        self.assertTrue(wprob > eprob)
        self.assertTrue(eprob > nprob)

    def test_optimal_agent(self):
        agent = agents.OptimalAgent(gamma=0.95, num_iters=20)
        self.optimal_agent_test(agent)

    def test_gridworld_optimal_agent(self):
        agent = fast_agents.FastOptimalAgent(gamma=0.95, num_iters=20)
        self.optimal_agent_test(agent)

    def time(self, fn, message):
        start = time.time()
        fn()
        end = time.time()
        print(message + ': ' + str(end - start) + 's')

    # TODO(rohinmshah): Think through and fix this test
    """
    def test_time_discounting_agents(self):
        grid = [['X', 'X', 'X', 'X', 'X', 'X', 'X'],
                ['X', ' ', ' ', ' ', ' ', 'A', 'X'],
                ['X', ' ', 'X',  4 , 'X', ' ', 'X'],
                ['X', 9.5, 'X', ' ', 'X',  4 , 'X'],
                ['X', 'X', 'X', 'X', 'X', 'X', 'X']]
        n, s, e, w, stay = self.all_actions

        mdp = GridworldMdp(grid, living_reward=-0.001)
        env = GridworldEnvironment(mdp)

        optimal_agent = agents.OptimalAgent(gamma=0.9, num_iters=20)
        optimal_agent.set_mdp(mdp)
        actions, _ = self.run_on_env(optimal_agent, env, gamma=0.9, episode_length=7)
        self.assertEqual(actions, [w, w, w, w, s, s, stay])

        naive_agent = agents.NaiveTimeDiscountingAgent(10, 1, gamma=0.9, num_iters=20)
        naive_agent.set_mdp(mdp)
        actions, _ = self.run_on_env(naive_agent, env, gamma=0.9, episode_length=7)
        self.assertEqual(actions, [w, w, s, stay, stay, stay, stay])

        soph_agent = agents.SophisticatedTimeDiscountingAgent(10, 1, gamma=0.9, num_iters=20)
        soph_agent.set_mdp(mdp)
        actions, _ = self.run_on_env(soph_agent, env, gamma=0.9, episode_length=7)
        self.assertEqual(actions, [s, s, stay])

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
        actions, _ = self.run_on_env(optimal_agent, env, gamma=0.9, episode_length=10)
        self.assertEqual(actions, [s, w, w, w, w, w, s, w, stay, stay])

        naive_agent.set_mdp(mdp)
        actions, _ = self.run_on_env(naive_agent, env, gamma=0.9, episode_length=10)
        self.assertEqual(actions, [s, w, w, w, w, w, w, stay, stay, stay])

        soph_agent.set_mdp(mdp)
        actions, _ = self.run_on_env(soph_agent, env, gamma=0.9, episode_length=10)
        self.assertEqual(actions, [s, s, s, w, w, w, w, w, n, w, stay])
    """

    def test_myopic_agent(self):
        grid = ['XXXXXXXX',
                'XA     X',
                'X XXXX9X',
                'X      X',
                'X X2   X',
                'XXXXXXXX']
        n, s, e, w, stay = self.all_actions

        mdp = GridworldMdp(grid, living_reward=-0.1)
        env = Mdp(mdp)

        optimal_agent = agents.OptimalAgent(gamma=0.9, num_iters=20)
        optimal_agent.set_mdp(mdp)
        actions, _ = self.run_on_env(optimal_agent, env, gamma=0.9, episode_length=10)
        self.assertEqual(actions, [e, e, e, e, e, s, stay, stay, stay, stay])

        myopic_agent = agents.MyopicAgent(6, gamma=0.9, num_iters=20)
        myopic_agent.set_mdp(mdp)
        actions, _ = self.run_on_env(myopic_agent, env, gamma=0.9, episode_length=10)
        self.assertEqual(actions, [s, s, e, e, e, e, e, n, stay, stay])

    def compare_agents(self, name, agent1, agent2, print_mdp=False):
        print('Comparing {0} agents'.format(name))
        set_seeds(314159)
        mdp = GridworldMdp.generate_random_connected(16, 16, 0.8)
        if print_mdp: print(mdp)
        env = Mdp(mdp)
        self.time(lambda: agent1.set_mdp(mdp), "Python planner")
        self.time(lambda: agent2.set_mdp(mdp), "Numpy planner")
        for s in mdp.get_states():
            for a in mdp.get_actions(s):
                mu = agent1.extend_state_to_mu(s)
                qval1, qval2 = agent1.qvalue(mu, a), agent2.qvalue(mu, a)
                self.assertAlmostEqual(qval1, qval2, places=7)

    def test_compare_optimal_agents(self):
        agent1 = agents.OptimalAgent(gamma=0.95, num_iters=20)
        agent2 = fast_agents.FastOptimalAgent(gamma=0.95, num_iters=20)
        self.compare_agents('optimal', agent1, agent2, print_mdp=True)

    def test_compare_naive_agents(self):
        agent1 = agents.NaiveTimeDiscountingAgent(10, 1, gamma=0.95, num_iters=20)
        agent2 = fast_agents.FastNaiveTimeDiscountingAgent(10, 1, gamma=0.95, num_iters=20)
        self.compare_agents('naive', agent1, agent2)

    def test_compare_sophisticated_agents(self):
        agent1 = agents.SophisticatedTimeDiscountingAgent(10, 1, gamma=0.95, num_iters=20)
        agent2 = fast_agents.FastSophisticatedTimeDiscountingAgent(10, 1, gamma=0.95, num_iters=20)
        self.compare_agents('sophisticated', agent1, agent2)

    def test_compare_myopic_agents(self):
        agent1 = agents.MyopicAgent(6, gamma=0.95, num_iters=20)
        agent2 = fast_agents.FastMyopicAgent(6, gamma=0.95, num_iters=20)
        self.compare_agents('myopic', agent1, agent2)

if __name__ == '__main__':
    unittest.main()
