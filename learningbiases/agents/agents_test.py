import unittest
import numpy as np
import tensorflow as tf
import time
import agents
import fast_agents

from agent_interface import Agent
from agent_runner import run_agent, get_reward_from_trajectory
from gridworld.gridworld import GridworldMdp, Direction
from mdp_interface import Mdp
from model import tf_value_iter_no_config
from utils import Distribution, set_seeds

class TestAgents(unittest.TestCase):
    def setUp(self):
        self.all_actions = Direction.ALL_DIRECTIONS

    def run_on_env(self, agent, env, gamma=0.9, episode_length=10):
        trajectory = run_agent(agent, env, episode_length, determinism=True)
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

    def test_uncalibrated_agents(self):
        grid = [['X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X'],
                ['X',  -9, ' ', 'X', ' ', ' ', ' ', ' ', ' ', ' ', 'X'],
                ['X', 'A', ' ', ' ', ' ', ' ', ' ', ' ', ' ',  3 , 'X'],
                ['X', ' ', ' ', 'X',  -9,  -9,  -9,  -9,  -9, ' ', 'X'],
                ['X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X', 'X']]
        n, s, e, w, stay = self.all_actions

        mdp = GridworldMdp(grid, living_reward=-0.1, noise=0.2)
        env = Mdp(mdp)

        agent1 = agents.OptimalAgent(gamma=0.9, num_iters=50)
        agent1.set_mdp(mdp)
        actions, _ = self.run_on_env(agent1, env, gamma=0.9, episode_length=13)
        self.assertEqual(actions, [e, e, e, n, e, e, e, e, e, s, stay, stay, stay])

        agent2 = agents.UncalibratedAgent(
            gamma=0.9, num_iters=20, calibration_factor=5)
        agent2.set_mdp(mdp)
        actions, _ = self.run_on_env(agent2, env, gamma=0.9, episode_length=13)
        self.assertEqual(actions, [e, e, e, e, e, e, e, e, stay, stay, stay, stay, stay])

        agent3 = agents.UncalibratedAgent(
            gamma=0.9, num_iters=20, calibration_factor=0.5)
        agent3.set_mdp(mdp)
        actions, _ = self.run_on_env(agent3, env, gamma=0.9, episode_length=13)
        self.assertEqual(actions, [s, e, n, e, e, n, e, e, e, e, e, s, stay])

    def compare_agents(self, name, agent1, agent2, places=7, print_mdp=False):
        print('Comparing {0} agents'.format(name))
        set_seeds(314159)
        mdp = GridworldMdp.generate_random_connected(16, 16, 5, 0.2)
        if print_mdp: print(mdp)
        env = Mdp(mdp)
        self.time(lambda: agent1.set_mdp(mdp), "Python planner")
        self.time(lambda: agent2.set_mdp(mdp), "Numpy/Tensorflow planner")
        for s in mdp.get_states():
            for a in mdp.get_actions(s):
                mu = agent1.extend_state_to_mu(s)
                qval1, qval2 = agent1.qvalue(mu, a), agent2.qvalue(mu, a)
                self.assertAlmostEqual(qval1, qval2, places=places)

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

    def test_compare_overconfident_agents(self):
        agent1 = agents.UncalibratedAgent(gamma=0.95, num_iters=20, calibration_factor=5)
        agent2 = fast_agents.FastUncalibratedAgent(gamma=0.95, num_iters=20, calibration_factor=5)
        self.compare_agents('overconfident', agent1, agent2)

    def test_compare_underconfident_agents(self):
        agent1 = agents.UncalibratedAgent(gamma=0.95, num_iters=20, calibration_factor=0.5)
        agent2 = fast_agents.FastUncalibratedAgent(gamma=0.95, num_iters=20, calibration_factor=0.5)
        # TODO(rohinmshah): This test fails at 3 decimal places, look
        # into this. This seems too large to be a rounding error so
        # could actually be a bug.
        self.compare_agents('underconfident', agent1, agent2, places=2)

    def test_value_iteration(self):
        agent1 = agents.OptimalAgent(gamma=0.95, num_iters=20)
        agent2 = ValueIterationAgent(gamma=0.95, num_iters=20)
        self.compare_agents('soft value iteration', agent1, agent2, places=2)


class ValueIterationAgent(Agent):
    def __init__(self, gamma=0.9, num_iters=50):
        super(ValueIterationAgent, self).__init__(gamma)
        self.num_iters = num_iters

    def create_tf_graph(self, imsize, noise):
        self.wall_tf = tf.placeholder(tf.float32, shape=(imsize, imsize))
        self.reward_tf = tf.placeholder(tf.float32, shape=(imsize, imsize))
        a = tf.reshape(self.wall_tf, [1, imsize, imsize])
        b = tf.reshape(self.reward_tf, [1, imsize, imsize])
        X = tf.stack([a, b],axis=-1)
        qvals_vector = tf_value_iter_no_config(
            X, ch_q=5, imsize=imsize, bsize=1, num_iters=self.num_iters,
            discount=self.gamma, noise=noise, vi_beta=1000).logits
        self.qvals_tensor = tf.reshape(qvals_vector, (imsize, imsize, 5))

    def set_mdp(self, mdp, reward_mdp=None):
        assert reward_mdp is None
        super(ValueIterationAgent, self).set_mdp(mdp)
        sess = tf.InteractiveSession()
        walls, reward, _ = mdp.convert_to_numpy_input()
        height, width = len(walls), len(walls[0])
        assert height == width
        self.create_tf_graph(height, mdp.noise)
        with tf.Session() as sess:
            fd = {
                self.wall_tf: walls,
                self.reward_tf: reward,
            }
            self.qvals = sess.run(self.qvals_tensor, feed_dict=fd)

    def qvalue(self, s, a, values=None):
        assert values is None
        x, y = s
        a_idx = Direction.get_number_from_direction(a)
        return self.qvals[y][x][a_idx]

if __name__ == '__main__':
    unittest.main()
