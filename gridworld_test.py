import unittest
from gridworld import GridworldMdp, Direction

class TestGridworldMdp(unittest.TestCase):
    def setUp(self):
        self.grid1 = GridworldMdp([['X', 'X', 'X', 'X', 'X'],
                                   ['X', ' ', ' ', 'A', 'X'],
                                   ['X', '3', 'X', ' ', 'X'],
                                   ['X', ' ', ' ', '1', 'X'],
                                   ['X', 'X', 'X', 'X', 'X']],
                                  living_reward=0)
        self.grid2 = GridworldMdp(['XXXXXXXXX',
                                   'X9X X  AX',
                                   'X X X   X',
                                   'X       X',
                                   'XXXXXXXXX'],
                                  noise=0.2)
        self.grid3 = GridworldMdp([['X', 'X', 'X', 'X', 'X'],
                                   ['X', 3.5, 'X', -10, 'X'],
                                   ['X', ' ', '0', ' ', 'X'],
                                   ['X', ' ', ' ', 'A', 'X'],
                                   ['X', 'X', 'X', 'X', 'X']])

    def test_constructor_invalid_inputs(self):
        with self.assertRaises(AssertionError):
            # Width must be at least 2.
            grid = GridworldMdp(['X', 'X', 'X'])
        with self.assertRaises(AssertionError):
            # Height must be at least 2.
            grid = GridworldMdp([['X', 'X', 'X']])

        with self.assertRaises(AssertionError):
            # Borders must be present.
            grid = GridworldMdp(['  A',
                                 '3X ',
                                 '  1'])

        with self.assertRaises(AssertionError):
            # There can't be more than one agent.
            grid = GridworldMdp(['XXXXX'
                                 'XA 3X'
                                 'X3 AX'
                                 'XXXXX'])

        with self.assertRaises(AssertionError):
            # There must be one agent.
            grid = GridworldMdp(['XXXXX'
                                 'X  3X'
                                 'X3  X'
                                 'XXXXX'])

        with self.assertRaises(AssertionError):
            # B is not a valid element.
            grid = GridworldMdp(['XXXXX'
                                 'XB  X'
                                 'X  3X'
                                 'XXXXX'])

    def test_start_state(self):
        self.assertEqual(self.grid1.get_start_state(), (3, 1))
        self.assertEqual(self.grid2.get_start_state(), (7, 1))
        self.assertEqual(self.grid3.get_start_state(), (3, 3))

    def test_reward_parsing(self):
        self.assertEqual(self.grid1.rewards, {
            (1, 2): 3,
            (3, 3): 1
        })
        self.assertEqual(self.grid2.rewards, {
            (1, 1): 9
        })
        self.assertEqual(self.grid3.rewards, {
            (1, 1): 3.5,
            (2, 2): 0,
            (3, 1): -10
        })

    def test_actions(self):
        a = [Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST]
        all_acts = set(a)
        no_acts = set([])

        self.assertEqual(set(self.grid1.get_actions((0, 0))), no_acts)
        self.assertEqual(set(self.grid1.get_actions((1, 1))), all_acts)
        self.assertEqual(set(self.grid1.get_actions((1, 2))), no_acts)
        self.assertEqual(set(self.grid2.get_actions((6, 2))), all_acts)
        self.assertEqual(set(self.grid2.get_actions((3, 1))), all_acts)
        self.assertEqual(set(self.grid3.get_actions((2, 2))), no_acts)

    def test_rewards(self):
        grid1_reward_table = {
            ((2, 3), Direction.EAST): 1,
            ((3, 2), Direction.SOUTH): 1,
            ((1, 3), Direction.NORTH): 3,
            ((1, 1), Direction.SOUTH): 3
        }
        grid2_reward_table = {
            ((1, 2), Direction.NORTH): 9
        }
        self.check_all_rewards(self.grid1, grid1_reward_table, 0)
        self.check_all_rewards(self.grid2, grid2_reward_table, -0.01)

    def check_all_rewards(self, mdp, reward_lookup_table, default):
        for y in range(mdp.height):
            for x in range(mdp.width):
                state = (x, y)
                # If the state is invalid (eg. a wall), it has no actions
                for action in mdp.get_actions(state):
                    expected = reward_lookup_table.get((state, action), default)
                    self.assertEqual(mdp.get_reward(state, action), expected)

    def test_transitions(self):
        n, s = Direction.NORTH, Direction.SOUTH
        e, w = Direction.EAST, Direction.WEST

        # Grid 1: No noise
        result = self.grid1.get_transition_states_and_probs((1, 3), n)
        self.assertEqual(result, [((1, 2), 1)])
        result = self.grid1.get_transition_states_and_probs((1, 1), n)
        self.assertEqual(result, [((1, 1), 1)])

        # Grid 2: Noise of 0.2
        result = set(self.grid2.get_transition_states_and_probs((1, 2), n))
        self.assertEqual(result, set([
            ((1, 1), 0.8),
            ((1, 2), 0.2)
        ]))
        result = set(self.grid2.get_transition_states_and_probs((6, 2), w))
        self.assertEqual(result, set([
            ((5, 2), 0.8),
            ((6, 1), 0.1),
            ((6, 3), 0.1)
        ]))
        result = set(self.grid2.get_transition_states_and_probs((7, 3), e))
        self.assertEqual(result, set([
            ((7, 3), 0.9),
            ((7, 2), 0.1)
        ]))
        result = set(self.grid2.get_transition_states_and_probs((5, 1), s))
        self.assertEqual(result, set([
            ((5, 2), 0.8),
            ((5, 1), 0.1),
            ((6, 1), 0.1)
        ]))

    def test_states_reachable(self):
        def check_grid(grid):
            self.assertEqual(set(grid.get_states()), self.dfs(grid))

        # Some of the states in self.grid1 are not reachable, since the agent
        # can't move out of a state with reward in it, so don't check grid1.
        for grid in [self.grid2, self.grid3]:
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

if __name__ == '__main__':
    unittest.main()
