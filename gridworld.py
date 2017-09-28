from collections import defaultdict

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
        return [(x, y) for x, y in coords if not self.walls[y][x]]

    def get_actions(self, state):
        """Returns list of valid actions for 'state'.

        Note that you can request moves into walls.
        """
        x, y = state
        if state in self.rewards or self.walls[y][x]:
            return []
        act = [Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST]
        return act

    def get_reward(self, state, action):
        """Get reward for state, action transition."""
        # TODO(rohinmshah): This would be harder for a neural net to understand
        # compared to a reward function that only looked at the current state.
        next_state = Direction.move_in_direction(state, action)
        if next_state in self.rewards:
            return self.rewards[next_state]
        return self.living_reward

    def is_terminal(self, state):
        return state in self.rewards

    def get_transition_states_and_probs(self, state, action):
        """Information about possible transitions for the action.

        Returns list of (next_state, prob) pairs representing the states
        reachable from 'state' by taking 'action' along with their transition
        probabilities.
        """
        if action not in self.get_actions(state):
            raise ValueError("Illegal action %s in state %s" % (action, state))

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
        x, y = state
        newx, newy = Direction.move_in_direction(state, action)
        return state if self.walls[newy][newx] else (newx, newy)

class Direction:
    NORTH = (0, -1)
    SOUTH = (0, 1)
    EAST  = (1, 0)
    WEST  = (-1, 0)

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
