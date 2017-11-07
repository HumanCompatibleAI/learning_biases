from collections import defaultdict
from disjoint_sets import DisjointSets
import numpy as np
import random

class GridworldMdp(object):
    """A grid world where the objective is to navigate to one of many rewards.

    Specifies all of the static information that an agent has access to when
    playing in the given grid world, including the state space, action space,
    transition probabilities, rewards, start space, etc.

    Once an agent arrives at a state with a reward, the agent must take the EXIT
    action which will give it the reward. In any other state, the agent can take
    any of the four cardinal directions as an action, getting a living reward
    (typically negative in order to incentivize shorter paths).
    """

    def __init__(self, grid, living_reward=-0.01, noise=0):
        """Initializes the MDP.

        grid: A sequence of sequences of spaces, representing a grid of a
        certain height and width. See assert_valid_grid for details on the grid
        format.
        living_reward: The reward obtained when taking any action besides EXIT.
        noise: Probability that when the agent takes a non-EXIT action (that is,
        a cardinal direction), it instead moves in one of the two adjacent
        cardinal directions.

        Raises: AssertionError if the grid is invalid.
        """
        self.assert_valid_grid(grid)
        self.height = len(grid)
        self.width = len(grid[0])
        self.living_reward = living_reward
        self.noise = noise
        self.terminal_state = 'Terminal State'

        self.walls = [[space == 'X' for space in row] for row in grid]
        self.populate_rewards_and_start_state(grid)

    def assert_valid_grid(self, grid):
        """Raises an AssertionError if the grid is invalid.

        grid:  A sequence of sequences of spaces, representing a grid of a
        certain height and width. grid[y][x] is the space at row y and column
        x. A space must be either 'X' (representing a wall), ' ' (representing
        an empty space), 'A' (representing the start state), or a value v so
        that float(v) succeeds (representing a reward).

        Often, grid will be a list of strings, in which case the rewards must be
        single digit positive rewards.
        """
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
        floats = [element for element in all_elements if is_float(element)]
        assert len(floats) >= 1, 'There must at least one reward square'

    def populate_rewards_and_start_state(self, grid):
        """Sets self.rewards and self.start_state based on grid.

        Assumes that grid is a valid grid.

        grid: A sequence of sequences of spaces, representing a grid of a
        certain height and width. See assert_valid_grid for details on the grid
        format.
        """
        self.rewards = {}
        self.start_state = None
        for y in range(len(grid)):
            for x in range(len(grid[0])):
                if grid[y][x] not in ['X', ' ', 'A']:
                    self.rewards[(x, y)] = float(grid[y][x])
                elif grid[y][x] == 'A':
                    self.start_state = (x, y)

    def get_random_start_state(self):
        """Returns a state that would be a legal start state for an agent.

        Avoids walls and reward/exit states.

        Returns: Randomly chosen state (x, y).
        """
        y = random.randint(1, self.height - 2)
        x = random.randint(1, self.width - 2)
        while self.walls[y][x] or (x, y) in self.rewards:
            y = random.randint(1, self.height - 2)
            x = random.randint(1, self.width - 2)
        return (x, y)

    def convert_to_numpy_input(self):
        """Encodes this MDP in a format well-suited for deep models.

        Returns three things -- a grid of indicators for whether or not a wall
        is present, a grid of reward values (not including living reward), and
        the start state (a tuple in the format x, y).
        """
        walls = np.array(self.walls, dtype=int)
        start_state = self.start_state
        rewards = np.zeros([self.height, self.width], dtype=float)
        for x, y in self.rewards:
            rewards[y, x] = self.rewards[(x, y)]
        return walls, rewards, start_state


    @staticmethod
    def generate_random(height, width, pr_wall, pr_reward):
        """Generates a random instance of a Gridworld.

        Note that based on the generated walls and start position, it may be
        impossible for the agent to ever reach a reward.
        """
        grid = [['X'] * width for _ in range(height)]
        for y in range(1, height - 1):
            for x in range(1, width - 1):
                if random.random() < pr_reward:
                    grid[y][x] = random.randint(-9, 9)
                    # Don't allow 0 rewards
                    while grid[y][x] == 0:
                        grid[y][x] = random.randint(-9, 9)
                elif random.random() >= pr_wall:
                    grid[y][x] = ' '

        def set_random_position_to(token):
            current_val = None
            while current_val not in ['X', ' ']:
                y = random.randint(1, height - 2)
                x = random.randint(1, width - 2)
                current_val = grid[y][x]
            grid[y][x] = token

        set_random_position_to(3)
        set_random_position_to('A')
        return GridworldMdp(grid)

    @staticmethod
    def generate_random_connected(height, width, pr_reward):
        """Generates a random instance of a Gridworld.

        Unlike with generate_random, it is guaranteed that the agent
        can reach a reward. However, that reward might be negative.
        """
        directions = [
            Direction.NORTH, Direction.SOUTH, Direction.EAST, Direction.WEST]
        grid = [['X'] * width for _ in range(height)]
        walls = [(x, y) for x in range(1, width-1) for y in range(1, height-1)]
        random.shuffle(walls)
        min_free_spots = len(walls) / 2
        dsets = DisjointSets([])
        while dsets.get_num_elements() < min_free_spots or not dsets.is_connected():
            x, y = walls.pop()
            grid[y][x] = ' '
            dsets.add_singleton((x, y))
            for direction in directions:
                newx, newy = Direction.move_in_direction((x, y), direction)
                if dsets.contains((newx, newy)):
                    dsets.union((x, y), (newx, newy))

        def set_random_position_to(token):
            current_val = None
            while current_val not in ['X', ' ']:
                y = random.randint(1, height - 2)
                x = random.randint(1, width - 2)
                current_val = grid[y][x]
            grid[y][x] = token

        # Makes sure there is one reward
        set_random_position_to(3)
        # Sets random starting point for agent
        set_random_position_to('A')
        while random.random() < pr_reward:
            reward = random.randint(-9, 9)
            # Don't allow 0 rewards
            while reward == 0:
                reward = random.randint(-9, 9)
            set_random_position_to(reward)

        return GridworldMdp(grid)

    def get_start_state(self):
        """Returns the start state."""
        return self.start_state

    def get_states(self):
        """Returns a list of all possible states the agent can be in.

        Note it is not guaranteed that the agent can reach all of these states.
        """
        coords = [(x, y) for x in range(self.width) for y in range(self.height)]
        all_states = [(x, y) for x, y in coords if not self.walls[y][x]]
        all_states.append(self.terminal_state)
        return all_states

    def get_actions(self, state):
        """Returns the list of valid actions for 'state'.

        Note that you can request moves into walls. The order in which actions
        are returned is guaranteed to be deterministic, in order to allow agents
        to implement deterministic behavior.
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
        """Get reward for state, action transition.

        This is the living reward, except when we take EXIT, in which case we
        return the reward for the current state.
        """
        if state in self.rewards and action == Direction.EXIT:
            return self.rewards[state]
        return self.living_reward

    def is_terminal(self, state):
        """Returns True if the current state is terminal, False otherwise.

        A state is terminal if there are no actions available from it (which
        means that the episode is over).
        """
        return state == self.terminal_state

    def get_transition_states_and_probs(self, state, action):
        """Gets information about possible transitions for the action.

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

    def __str__(self):
        """Returns a string representation of this grid world.

        The returned string has a line for every row, and each space is exactly
        one character. These are encoded in the same way as the grid input to
        the constructor -- walls are 'X', empty spaces are ' ', the start state
        is 'A', and rewards are their own values. However, rewards like 3.5 or
        -9 cannot be represented with a single character. Such rewards are
        encoded as 'R' (if positive) or 'N' (if negative).
        """
        def get_char(x, y):
            if self.walls[y][x]:
                return 'X'
            elif (x, y) in self.rewards:
                reward = self.rewards[(x, y)]
                # Convert to an int if it would not lose information
                reward = int(reward) if int(reward) == reward else reward
                posneg_char = 'R' if reward >= 0 else 'N'
                reward_str = str(reward)
                return reward_str if len(reward_str) == 1 else posneg_char
            elif (x, y) == self.get_start_state():
                return 'A'
            else:
                return ' '

        def get_row_str(y):
            return ''.join([get_char(x, y) for x in range(self.width)])

        return '\n'.join([get_row_str(y) for y in range(self.height)])

# TODO(rohinmshah): This is a generic MDP environment, it isn't specific to
# Gridworlds. Put it in its own file and rename the gridworld field to mdp.
class GridworldEnvironment(object):
    """An environment containing a single agent that can take actions.

    The environment keeps track of the current state of the agent, and updates
    it as the agent takes actions, and provides rewards to the agent.
    """

    def __init__(self, gridworld):
        self.gridworld = gridworld
        self.reset()

    def get_current_state(self):
        return self.state

    def get_actions(self, state):
        return self.gridworld.get_actions(state)

    def perform_action(self, action):
        """Performs the action, updating the state and providing a reward."""
        state = self.get_current_state()
        next_state, reward = self.get_random_next_state(state, action)
        self.state = next_state
        return (next_state, reward)

    def get_random_next_state(self, state, action):
        """Chooses the next state according to T(state, action)."""
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
        """Resets the environment. Does NOT reset the agent."""
        self.state = self.gridworld.get_start_state()

    def is_done(self):
        """Returns True if the episode is over and the agent cannot act."""
        return self.gridworld.is_terminal(self.get_current_state())

class Direction(object):
    """A class that contains the five actions available in Gridworlds.

    Includes definitions of the actions as well as utility functions for
    manipulating them or applying them.
    """
    NORTH = (0, -1)
    SOUTH = (0, 1)
    EAST  = (1, 0)
    WEST  = (-1, 0)
    # This is hacky, but we do want to ensure that EXIT is distinct from the
    # other actions, and so we define it here instead of in an Action class.
    EXIT = 4
    INDEX_TO_DIRECTION = [NORTH, SOUTH, EAST, WEST, EXIT]
    DIRECTION_TO_INDEX = { a:i for i, a in enumerate(INDEX_TO_DIRECTION) }
    ALL_DIRECTIONS = INDEX_TO_DIRECTION

    @staticmethod
    def move_in_direction(point, direction):
        """Takes a step in the given direction and returns the new point.

        point: Tuple (x, y) representing a point in the x-y plane.
        direction: One of the Directions, except not Direction.EXIT.
        """
        x, y = point
        dx, dy = direction
        return (x + dx, y + dy)

    @staticmethod
    def get_adjacent_directions(direction):
        """Returns the directions within 90 degrees of the given direction.

        direction: One of the Directions, except not Direction.EXIT.
        """
        if direction in [Direction.NORTH, Direction.SOUTH]:
            return [Direction.EAST, Direction.WEST]
        elif direction in [Direction.EAST, Direction.WEST]:
            return [Direction.NORTH, Direction.SOUTH]
        raise ValueError('Invalid direction: %s' % direction)

    @staticmethod
    def get_number_from_direction(direction):
        return Direction.DIRECTION_TO_INDEX[direction]

    @staticmethod
    def get_direction_from_number(number):
        return Direction.INDEX_TO_DIRECTION[number]
