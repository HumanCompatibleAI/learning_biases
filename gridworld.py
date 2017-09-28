from collections import defaultdict
import random

class GridworldMdp:
    """A grid world where the objective is to navigate to one of many rewards.

    Specifies all of the static information that an agent has access to when
    playing in the given grid world, including the state space, action space,
    transition probabilities, rewards, start space, etc.
    """

    def __init__(self, grid, living_reward=-0.01, noise=0):
        self.assert_valid_grid(grid)
        self.height = len(grid)
        self.width = len(grid[0])
        self.living_reward = living_reward
        self.noise = noise
        self.terminal_state = 'Terminal State'

        self.walls = [[space == 'X' for space in row] for row in grid]
        self.populate_rewards_and_start_state(grid)

    def assert_valid_grid(self, grid):
        height = len(grid)
        width = len(grid[0])

        # Make sure the grid is not ragged
        assert all(len(row) == width for row in grid), 'Ragged grid'

        # Borders must all be walls
        for y in range(height):
            assert grid[y][0] == 'X', 'Left border must be a wall'
            assert grid[y][-1] == 'X', 'Right border must be a wall'
        for x in range(width):
            assert grid[0][x] == 'X', 'Top border must be a wall'
            assert grid[-1][x] == 'X', 'Bottom border must be a wall'

        def is_float(element):
            try:
                return float(element) or True
            except ValueError:
                return False

        # An element can be 'X' (a wall), ' ' (empty element), 'A' (the agent),
        # or a value v such that float(v) succeeds and returns a float.
        def is_valid_element(element):
            return element in ['X', ' ', 'A'] or is_float(element)

        all_elements = [element for row in grid for element in row]
        assert all(is_valid_element(element) for element in all_elements), 'Invalid element: must be X, A, blank space, or a number'
        assert all_elements.count('A') == 1, "'A' must be present exactly once"

    def populate_rewards_and_start_state(self, grid):
        self.rewards = {}
        self.start_state = None
        for y in range(len(grid)):
            for x in range(len(grid[0])):
                if grid[y][x] not in ['X', ' ', 'A']:
                    self.rewards[(x, y)] = float(grid[y][x])
                elif grid[y][x] == 'A':
                    self.start_state = (x, y)

    def get_start_state(self):
        return self.start_state

    def get_states(self):
        coords = [(x, y) for x in range(self.width) for y in range(self.height)]
        all_states = [(x, y) for x, y in coords if not self.walls[y][x]]
        all_states.append(self.terminal_state)
        return all_states

    def get_actions(self, state):
        """Returns list of valid actions for 'state'.

        Note that you can request moves into walls.
        """
        if self.is_terminal(state):
            return []
        x, y = state
        if self.walls[y][x]:
            return []
        if state in self.rewards:
            return [Direction.EXIT]
        act = [Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST]
        return act

    def get_reward(self, state, action):
        """Get reward for state, action transition."""
        if state in self.rewards and action == Direction.EXIT:
            return self.rewards[state]
        return self.living_reward

    def is_terminal(self, state):
        return state == self.terminal_state

    def get_transition_states_and_probs(self, state, action):
        """Information about possible transitions for the action.

        Returns list of (next_state, prob) pairs representing the states
        reachable from 'state' by taking 'action' along with their transition
        probabilities.
        """
        if action not in self.get_actions(state):
            raise ValueError("Illegal action %s in state %s" % (action, state))

        if action == Direction.EXIT:
            return [(self.terminal_state, 1.0)]

        next_state = self.attempt_to_move_in_direction(state, action)
        if self.noise == 0.0:
            return [(next_state, 1.0)]

        successors = defaultdict(float)
        successors[next_state] += 1.0 - self.noise
        for direction in Direction.get_adjacent_directions(action):
            next_state = self.attempt_to_move_in_direction(state, direction)
            successors[next_state] += (self.noise / 2.0)

        return successors.items()

    def attempt_to_move_in_direction(self, state, action):
        """Return the new state an agent would be in if it took the action.

        Requires: action is in self.get_actions(state).
        """
        x, y = state
        newx, newy = Direction.move_in_direction(state, action)
        return state if self.walls[newy][newx] else (newx, newy)

class GridworldEnvironment(object):

    def __init__(self, gridworld):
        self.gridworld = gridworld
        self.reset()

    def get_current_state(self):
        return self.state

    def get_actions(self, state):
        return self.gridworld.get_actions(state)

    def perform_action(self, action):
        state = self.get_current_state()
        next_state, reward = self.get_random_next_state(state, action)
        self.state = next_state
        return (next_state, reward)

    def get_random_next_state(self, state, action):
        rand = random.random()
        sum = 0.0
        results = self.gridworld.get_transition_states_and_probs(state, action)
        for next_state, prob in results:
            sum += prob
            if sum > 1.0:
                raise ValueError('Total transition probability more than one.')
            if rand < sum:
                reward = self.gridworld.get_reward(state, action)
                return (next_state, reward)
        raise ValueError('Total transition probability less than one.')

    def reset(self):
        self.state = self.gridworld.get_start_state()

    def is_done(self):
        return self.gridworld.is_terminal(self.get_current_state())

class Direction:
    NORTH = (0, -1)
    SOUTH = (0, 1)
    EAST  = (1, 0)
    WEST  = (-1, 0)
    # This is hacky, but we do want to ensure that EXIT is distinct from the
    # other actions, and so we define it here instead of in an Action class.
    EXIT = 0

    @staticmethod
    def move_in_direction(point, direction):
        x, y = point
        dx, dy = direction
        return (x + dx, y + dy)

    @staticmethod
    def get_adjacent_directions(direction):
        if direction in [Direction.NORTH, Direction.SOUTH]:
            return [Direction.EAST, Direction.WEST]
        elif direction in [Direction.EAST, Direction.WEST]:
            return [Direction.NORTH, Direction.SOUTH]
        raise ValueError('Invalid direction: %s' % direction)
