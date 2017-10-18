import unittest

from gridworld import GridworldMdp, GridworldEnvironment, Direction
import random

class TestDirection(unittest.TestCase):
    def test_direction_number_conversion(self):
        all_directions = Direction.ALL_DIRECTIONS
        all_numbers = []

        for direction in Direction.ALL_DIRECTIONS:
            number = Direction.get_number_from_direction(direction)
            direction_again = Direction.get_direction_from_number(number)
            self.assertEqual(direction, direction_again)
            all_numbers.append(number)

        # Check that all directions are distinct
        num_directions = len(all_directions)
        self.assertEqual(len(set(all_directions)), num_directions)
        # Check that the numbers are 0, 1, ... num_directions - 1
        self.assertEqual(set(all_numbers), set(range(num_directions)))

class TestGridworld(unittest.TestCase):
    def setUp(self):
        self.grid1 = [['X', 'X', 'X', 'X', 'X'],
                      ['X', ' ', ' ', 'A', 'X'],
                      ['X', '3', 'X', ' ', 'X'],
                      ['X', ' ', ' ', '1', 'X'],
                      ['X', 'X', 'X', 'X', 'X']]
        self.grid2 = ['XXXXXXXXX',
                      'X9X X  AX',
                      'X X X   X',
                      'X       X',
                      'XXXXXXXXX']
        self.grid3 = [['X', 'X', 'X', 'X', 'X'],
                      ['X', 3.5, 'X', -10, 'X'],
                      ['X', ' ', '0', ' ', 'X'],
                      ['X', ' ', ' ', 'A', 'X'],
                      ['X', 'X', 'X', 'X', 'X']]

        self.mdp1 = GridworldMdp(self.grid1, living_reward=0)
        self.mdp2 = GridworldMdp(self.grid2, noise=0.2)
        self.mdp3 = GridworldMdp(self.grid3)

    def test_str(self):
        expected = '\n'.join([''.join(row) for row in self.grid1])
        self.assertEqual(str(self.mdp1), expected)
        expected = '\n'.join(self.grid2)
        self.assertEqual(str(self.mdp2), expected)
        expected = '\n'.join(['XXXXX',
                              'XRXNX',
                              'X 0 X',
                              'X  AX',
                              'XXXXX'])
        self.assertEqual(str(self.mdp3), expected)

    def test_constructor_invalid_inputs(self):
        # Height and width must be at least 2.
        with self.assertRaises(AssertionError):
            mdp = GridworldMdp(['X', 'X', 'X'])
        with self.assertRaises(AssertionError):
            mdp = GridworldMdp([['X', 'X', 'X']])

        with self.assertRaises(AssertionError):
            # Borders must be present.
            mdp = GridworldMdp(['  A',
                                '3X ',
                                '  1'])

        with self.assertRaises(AssertionError):
            # There can't be more than one agent.
            mdp = GridworldMdp(['XXXXX',
                                'XA 3X',
                                'X3 AX',
                                'XXXXX'])

        with self.assertRaises(AssertionError):
            # There must be one agent.
            mdp = GridworldMdp(['XXXXX',
                                'X  3X',
                                'X3  X',
                                'XXXXX'])

        with self.assertRaises(AssertionError):
            # There must be at least one reward.
            mdp = GridworldMdp(['XXXXX',
                                'XAX X',
                                'X   X',
                                'XXXXX'])

        with self.assertRaises(AssertionError):
            # B is not a valid element.
            mdp = GridworldMdp(['XXXXX',
                                'XB  X',
                                'X  3X',
                                'XXXXX'])

    def test_start_state(self):
        self.assertEqual(self.mdp1.get_start_state(), (3, 1))
        self.assertEqual(self.mdp2.get_start_state(), (7, 1))
        self.assertEqual(self.mdp3.get_start_state(), (3, 3))

    def test_reward_parsing(self):
        self.assertEqual(self.mdp1.rewards, {
            (1, 2): 3,
            (3, 3): 1
        })
        self.assertEqual(self.mdp2.rewards, {
            (1, 1): 9
        })
        self.assertEqual(self.mdp3.rewards, {
            (1, 1): 3.5,
            (2, 2): 0,
            (3, 1): -10
        })

    def test_actions(self):
        a = [Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST]
        all_acts = set(a)
        exit_acts = set([Direction.EXIT])
        no_acts = set([])

        self.assertEqual(set(self.mdp1.get_actions((0, 0))), no_acts)
        self.assertEqual(set(self.mdp1.get_actions((1, 1))), all_acts)
        self.assertEqual(set(self.mdp1.get_actions((1, 2))), exit_acts)
        self.assertEqual(set(self.mdp2.get_actions((6, 2))), all_acts)
        self.assertEqual(set(self.mdp2.get_actions((3, 1))), all_acts)
        self.assertEqual(set(self.mdp3.get_actions((2, 2))), exit_acts)

    def test_rewards(self):
        grid1_reward_table = {
            ((3, 3), Direction.EXIT): 1,
            ((1, 2), Direction.EXIT): 3
        }
        grid2_reward_table = {
            ((1, 1), Direction.EXIT): 9
        }
        grid3_reward_table = {
            ((1, 1), Direction.EXIT): 3.5,
            ((2, 2), Direction.EXIT): 0,
            ((3, 1), Direction.EXIT): -10
        }
        self.check_all_rewards(self.mdp1, grid1_reward_table, 0)
        self.check_all_rewards(self.mdp2, grid2_reward_table, -0.01)
        self.check_all_rewards(self.mdp3, grid3_reward_table, -0.01)

    def check_all_rewards(self, mdp, reward_lookup_table, default):
        for state in mdp.get_states():
            for action in mdp.get_actions(state):
                expected = reward_lookup_table.get((state, action), default)
                self.assertEqual(mdp.get_reward(state, action), expected)

    def test_transitions(self):
        n, s = Direction.NORTH, Direction.SOUTH
        e, w = Direction.EAST, Direction.WEST
        exit_action = Direction.EXIT

        # Grid 1: No noise
        result = self.mdp1.get_transition_states_and_probs((1, 3), n)
        self.assertEqual(set(result), set([((1, 2), 1)]))
        result = self.mdp1.get_transition_states_and_probs((1, 2), exit_action)
        self.assertEqual(set(result), set([(self.mdp1.terminal_state, 1)]))
        result = self.mdp1.get_transition_states_and_probs((1, 1), n)
        self.assertEqual(set(result), set([((1, 1), 1)]))

        # Grid 2: Noise of 0.2
        result = set(self.mdp2.get_transition_states_and_probs((1, 2), n))
        self.assertEqual(result, set([
            ((1, 1), 0.8),
            ((1, 2), 0.2)
        ]))
        result = set(self.mdp2.get_transition_states_and_probs((6, 2), w))
        self.assertEqual(result, set([
            ((5, 2), 0.8),
            ((6, 1), 0.1),
            ((6, 3), 0.1)
        ]))
        result = set(self.mdp2.get_transition_states_and_probs((7, 3), e))
        self.assertEqual(result, set([
            ((7, 3), 0.9),
            ((7, 2), 0.1)
        ]))
        result = set(self.mdp2.get_transition_states_and_probs((5, 1), s))
        self.assertEqual(result, set([
            ((5, 2), 0.8),
            ((5, 1), 0.1),
            ((6, 1), 0.1)
        ]))
        result = self.mdp2.get_transition_states_and_probs((3, 1), n)
        self.assertEqual(set(result), set([((3, 1), 1)]))
        result = self.mdp2.get_transition_states_and_probs((1, 1), exit_action)
        self.assertEqual(set(result), set([(self.mdp2.terminal_state, 1)]))

    def test_states_reachable(self):
        def check_grid(grid):
            self.assertEqual(set(grid.get_states()), self.dfs(grid))

        # Some of the states in self.mdp1 are not reachable, since the agent
        # can't move out of a state with reward in it, so don't check grid1.
        for grid in [self.mdp2, self.mdp3]:
            check_grid(grid)

    def dfs(self, grid):
        visited = set()
        def helper(state):
            if state in visited:
                return
            visited.add(state)
            for action in grid.get_actions(state):
                for next_state, _ in grid.get_transition_states_and_probs(state, action):
                    helper(next_state)

        helper(grid.get_start_state())
        return visited

    def test_environment(self):
        env = GridworldEnvironment(self.mdp3)
        self.assertEqual(env.get_current_state(), (3, 3))
        next_state, reward = env.perform_action(Direction.NORTH)
        self.assertEqual(next_state, (3, 2))
        self.assertEqual(reward, -0.01)
        self.assertEqual(env.get_current_state(), next_state)
        self.assertFalse(env.is_done())
        env.reset()
        self.assertEqual(env.get_current_state(), (3, 3))
        self.assertFalse(env.is_done())
        next_state, reward = env.perform_action(Direction.WEST)
        self.assertEqual(next_state, (2, 3))
        self.assertEqual(reward, -0.01)
        self.assertEqual(env.get_current_state(), next_state)
        self.assertFalse(env.is_done())
        next_state, reward = env.perform_action(Direction.NORTH)
        self.assertEqual(next_state, (2, 2))
        self.assertEqual(reward, -0.01)
        self.assertEqual(env.get_current_state(), next_state)
        self.assertFalse(env.is_done())
        next_state, reward = env.perform_action(Direction.EXIT)
        self.assertEqual(next_state, self.mdp3.terminal_state)
        self.assertEqual(reward, 0)
        self.assertEqual(env.get_current_state(), next_state)
        self.assertTrue(env.is_done())
        env.reset()
        self.assertFalse(env.is_done())

    def test_random_gridworld_generation(self):
        random.seed(314159)
        mdp = GridworldMdp.generate_random(8, 8, 0, 0)
        self.assertEqual(mdp.height, 8)
        self.assertEqual(mdp.width, 8)
        mdp_string = str(mdp)
        self.assertEqual(mdp_string.count('X'), 28)
        self.assertEqual(mdp_string.count(' '), 34)
        self.assertEqual(mdp_string.count('A'), 1)
        self.assertEqual(mdp_string.count('3'), 1)

if __name__ == '__main__':
    unittest.main()
