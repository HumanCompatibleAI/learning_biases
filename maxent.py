"""
Source @Adam's population-irl repo
Implements a tabular version of maximum entropy inverse reinforcement learning
(Ziebart et al, 2008). There are two key differences from the version
described in the paper:
  - We do not make use of features, which are not needed in the tabular setting.
  - We use Adam rather than exponentiated gradient descent.


Note: all coordinates are expected to be (y,x) because that's how the Gridworld
code is indexed
"""

import numpy as np
np.set_printoptions(2)
import tabular_maxent
from tabular_maxent import irl, expected_counts
from gridworld import GridworldMdpNoR


def irl_with_config(image, action_dists, start, config, verbose=False):
    return irl_wrapper(image, action_dists, start, config.eval_horizon, config.gamma, verbose)


def irl_wrapper(image, action_dists, start, horizon, discount, verbose=False):
    """Takes in input that works with our codebase, and harnesses @AdamGleave's MaxEnt
    implementation to do MaxCausalEnt IRL. Baseline algorithm."""

    imsize = len(image)
    transition = GridworldMdpNoR(image, start).get_transition_matrix()
    policy = action_dists
    # Flatten policy
    policy = flatten_policy(policy)

    # Initialize uniform over all non-wall states
    # initial_states = np.ones(image.shape) - image
    # initial_states = initial_states / np.sum(initial_states)

    # Initialize the initial state prob distribution to only be the state
    # that the MDP starts at.
    initial_states = np.zeros(image.shape)
    initial_states[start[1], start[0]] = 1
    # Flatten initial_states array
    initial_states = np.reshape(initial_states, -1)

    demo_counts = expected_counts(policy, transition, initial_states, horizon, discount)

    xplatform = {'transition': transition, 'initial_states': initial_states}

    flat_inferred = irl(xplatform, None, discount, demo_counts, horizon)
    inferred_reward = np.reshape(flat_inferred, (imsize, imsize))
    return inferred_reward


def flatten_policy(policy):
    """Reshapes the policy
    (imsize, imsize, num_actions)
    to
    (imsize**2, num_actions)
    """
    init = policy
    # policy = np.transpose(policy, (1, 0, 2))
    policy = policy.reshape(len(policy)**2, -1)
    assert (policy.reshape(init.shape) == init).all(), "reshaping not consistent"
    assert (np.isclose(np.sum(policy, axis=-1), 1)).all(), "error while reshaping"
    return policy


def testTransition():
    """tests creation of the mdpNoR.get_transition_matrix() method, visually :)"""
    from gridworld import Direction
    walls = [[1,1,1,1],
             [1,0,0,1],
             [1,0,0,1],
             [1,1,1,1]]
    walls = np.array(walls)
    start = (3,1)
    mdp = GridworldMdpNoR(walls, start)
    trans = mdp.get_transition_matrix()
    walk = Direction.EAST
    start_grid = np.copy(walls)
    start_grid[start] = 9
    print("Original start grid: Moving East...")
    print(start_grid)

    # Should be
    # walls = [[1,1,1,1],
    #          [1,0,A,1],
    #          [1,0,0,1],
    #          [1,1,1,1]]
    #
    # walls = [[1,1,1,1],
    #          [1,A,0,1],
    #          [1,0,0,1],
    #          [1,1,1,1]]
    # And if the transition matrix works accurately for this, then it needs
    # to be transposed to be consistent with the grid returned by generate_example
    #
    # vecend should be as marked above (2, 1)
    assert (np.array(start) == recover(flatten_position(start, len(walls)), walls)).all(), "my methods to flatten/reshape broken"
    vecend = recover(recoverDirectionTrans(start, walk, trans, len(walls)), walls)

    print("Moving East took us:")
    print(vecend)
    end_grid = np.copy(walls)
    end_grid[int(vecend[0]), int(vecend[1])] = 9
    print(end_grid)

    if end_grid[2, 1] == 9:
        print("==> Transition matrix using y,x coords (gridworld)")
    elif end_grid[1,2] == 9:
        print("==> Transition matrix using x,y coords (non-gridworld)")


def recover(position, arr):
    size = len(arr)
    i = position // size
    j = position % size
    return (i, j)


def recoverDirectionTrans(position, direction, trans, size):
    from gridworld import Direction
    """Position is 2d, index into shape image grid
    Direction of form Direction.EAST

    Returns 1d index == state you end up in"""

    start = flatten_position(position, size)
    finish = flatten_position(np.array(position) + np.array(direction), size)
    # print(trans[start, Direction.get_number_from_direction(direction)])
    end_positions = trans[start, Direction.get_number_from_direction(direction)]
    return np.argmax(end_positions)


def flatten_position(position, size):
    return position[0] * size + position[1]


def test_irl(grid, agent):
    from gridworld import GridworldMdp, Direction
    from utils import Distribution

    num_actions = len(Direction.ALL_DIRECTIONS)

    mdp = GridworldMdp(grid=grid)
    agent.set_mdp(mdp)

    def dist_to_numpy(dist):
        return dist.as_numpy_array(Direction.get_number_from_direction, num_actions)

    def action(state):
        # Walls are invalid states and the MDP will refuse to give an action for
        # them. However, the VIN's architecture requires it to provide an action
        # distribution for walls too, so hardcode it to always be STAY.
        x, y = state
        if mdp.walls[y][x]:
            return dist_to_numpy(Distribution({Direction.STAY : 1}))
        return dist_to_numpy(agent.get_action_distribution(state))

    imsize = len(grid)

    # I think it's this line that's wrong. Writing as (y, x) gives expected SVF vec
    action_dists = [[action((x, y)) for y in range(imsize)] for x in range(imsize)]
    action_dists = np.array(action_dists)

    walls, rewards, start_state = mdp.convert_to_numpy_input()

    # print("Start state for given mdp:", start_state)
    inferred = irl_wrapper(walls, action_dists, start_state, 20, 0.9)
    # print("---true below---")
    # print(rewards)

    return walls, start_state, inferred, rewards


def test_visitations(grid, agent):
    """Tests the expected_counts calculation--might be einsum error"""
    # print("Testing expected_counts")
    from gridworld import GridworldMdp, Direction
    from utils import Distribution

    num_actions = len(Direction.ALL_DIRECTIONS)

    mdp = GridworldMdp(grid=grid)
    agent.set_mdp(mdp)

    def dist_to_numpy(dist):
        return dist.as_numpy_array(Direction.get_number_from_direction, num_actions)

    def action(state):
        # Walls are invalid states and the MDP will refuse to give an action for
        # them. However, the VIN's architecture requires it to provide an action
        # distribution for walls too, so hardcode it to always be STAY.
        x, y = state
        if mdp.walls[y][x]:
            return dist_to_numpy(Distribution({Direction.STAY: 1}))
        return dist_to_numpy(agent.get_action_distribution(state))

    imsize = len(grid)

    action_dists = [[action((x, y)) for y in range(imsize)] for x in range(imsize)]
    action_dists = np.array(action_dists)

    walls, rewards, start_state = mdp.convert_to_numpy_input()

    # print("Start state for given mdp:", start_state)

    start = start_state
    trans = mdp.get_transition_matrix()
    initial_states = np.zeros((len(grid), len(grid)))
    initial_states[start[1]][start[0]] = 1
    initial_states = initial_states.reshape(-1)
    policy = flatten_policy(action_dists)

    demo_counts = expected_counts(policy, trans, initial_states,
                                  20, 0.9)

    import matplotlib.pyplot as plt
    plt.imsave("democounts",demo_counts.reshape((len(grid), len(grid))))
    # print("demo counts:", demo_counts.reshape((len(grid), len(grid))))


def test_coherence(grid, agent):
    """Test that these arrays perform as expected under np.einsum"""
    from gridworld import GridworldMdp, Direction
    from utils import Distribution

    num_actions = len(Direction.ALL_DIRECTIONS)

    mdp = GridworldMdp(grid=grid)
    agent.set_mdp(mdp)

    def dist_to_numpy(dist):
        return dist.as_numpy_array(Direction.get_number_from_direction, num_actions)

    def action(state):
        # Walls are invalid states and the MDP will refuse to give an action for
        # them. However, the VIN's architecture requires it to provide an action
        # distribution for walls too, so hardcode it to always be STAY.
        x, y = state
        if mdp.walls[y][x]:
            return dist_to_numpy(Distribution({Direction.STAY: 1}))
        return dist_to_numpy(agent.get_action_distribution(state))

    imsize = len(grid)

    action_dists = [[action((x, y)) for y in range(imsize)] for x in range(imsize)]
    action_dists = np.array(action_dists)

    walls, rewards, start_state = mdp.convert_to_numpy_input()

    print("Start state for given mdp:", start_state)
    # inferred = _irl_wrapper(walls, action_dists, start_state, 20, 1.0)

    start = start_state
    trans = mdp.get_transition_matrix()
    initial_states = np.zeros((len(grid), len(grid)))
    initial_states[start[1]][start[0]] = 1
    initial_states = initial_states.reshape(-1)
    policy = flatten_policy(action_dists)

    gshape = (len(grid), len(grid))
    print("initial states")
    print('-'*20)
    print(initial_states.reshape(gshape))
    next_states = np.einsum("i,ij,ijk -> k", initial_states, policy, trans)
    # next_states = (next_states.reshape(gshape).T).reshape(-1)
    print("first expected counts")
    print('-'*20)
    print(next_states.reshape(gshape))
    next_states = np.einsum("i,ij,ijk -> k", next_states, policy, trans)
    print("second expected counts")
    print('-'*20)
    print(next_states.reshape(gshape))

    next_states = np.einsum("i,ij,ijk -> k", next_states, policy, trans)
    # next_states = (next_states.reshape(gshape).T).reshape(-1)
    print("third expected counts")
    print('-'*20)
    print(next_states.reshape(gshape))


    # for i in range(5):
    #     next_states = np.einsum("i,ij,ijk -> k", next_states, policy, trans)
    #     # next_states = (next_states.reshape(gshape).T).reshape(-1)
    #     print("{}th expected counts".format(4+i))
    #     print('-'*20)
    #     print(next_states.reshape(gshape))
    return next_states.reshape((len(grid), len(grid)))


if __name__ == '__main__':
    from agents import OptimalAgent
    from agent_runner import evaluate_proxy
    import copy

    # testTransition()
    # grid = [['X','X','X','X'],
    #         ['X',  1,'A','X'],
    #         ['X',' ',' ','X'],
    #         ['X','X','X','X']]
    # trans =[['X','X','X','X'],
    #         ['X',  1,' ','X'],
    #         ['X','A',' ','X'],
    #         ['X','X','X','X']]
    # base = [['X','X','X','X','X','X'],
    #         ['X',' ',' ',' ',' ','X'],
    #         ['X',' ','X','X','X','X'],
    #         ['X',' ','X','X',' ','X'],
    #         ['X',' ',' ',' ',  1,'X'],
    #         ['X','X','X','X','X','X']]
    base = [['X','X','X','X','X','X'],
            ['X',' ',' ','X','X','X'],
            ['X',' ',' ',' ','X','X'],
            ['X',  3,' ',' ',' ','X'],
            ['X',' ',' ',' ',' ','X'],
            ['X','X','X','X','X','X']]
    # base = [['X','X','X','X','X','X'],
    #         ['X',' ','X',' ',' ','X'],
    #         ['X',' ','X','X',' ','X'],
    #         ['X',' ',' ',' ',' ','X'],
    #         ['X',' ',' ',' ',  1,'X'],
    #         ['X','X','X','X','X','X']]
    # base = [['X','X','X','X'],
    #         ['X',' ',' ','X'],
    #         ['X','1','X','X'],
    #         ['X','X','X','X']]
    grid = copy.deepcopy(base)
    grid[3][4]='A'
    # grid[1][4] = 'A'
    # grid[1][2] = 'A'
    # grid[1][4] = 'A'
    walls, start_state, inferred, rs = test_irl(grid, OptimalAgent(beta=1.0))

    print("inferred:\n",inferred)
    almostregret = evaluate_proxy(walls, start_state, inferred, rs, episode_length=20)
    print('Percent return:', almostregret)

    # print("")
    # walls, start_state, inferred, rs = test_irl(trans, OptimalAgent(beta=1.0))
    # print("inferred:\n",inferred)
    # almostregret = evaluate_proxy(walls,start_state,inferred,rs,episode_length=20)
    # print('Percent return:', almostregret)

    # test_visitations(grid, agent=OptimalAgent(beta=1.0))

    # test_coherence(grid, agent=OptimalAgent(beta=1.0))
