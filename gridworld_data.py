import argparse
import numpy as np
import random

import agents
from agent_runner import run_agent
from gridworld import GridworldMdp, GridworldEnvironment, Direction

# Currently unused, but may be useful later
def print_training_example(mdp, trajectory):
    """Prints the gridworld with the trajectory overlaid on top of it.

    mdp: A Gridworld MDP (not a generic MDP).
    trajectory: The trajectory of the agent in the MDP.
    """
    # Drop the last two next states, which are the terminal state and the state
    # with the reward. Note this does a reasonable thing even when the agent
    # never got to the reward.
    states_to_mark = [next_state for _, _, next_state, _ in trajectory[:-2]]
    mdp_grid = [[c for c in row] for row in str(mdp).split('\n')]
    for x, y in states_to_mark:
        mdp_grid[y][x] = '.'
    mdp_string_with_trajectory = '\n'.join([''.join(row) for row in mdp_grid])
    print(mdp_string_with_trajectory)

# TODO(rohinmshah): This really belongs in gridworld.py
def get_random_start_state(mdp):
    """Returns a state in mdp that would be a legal start state for an agent.

    mdp: A Gridworld MDP (not a generic MDP).

    Returns: Randomly chosen state (x, y).
    """
    y = random.randint(1, mdp.height - 2)
    x = random.randint(1, mdp.width - 2)
    while mdp.walls[y][x] or (x, y) in mdp.rewards:
        y = random.randint(1, mdp.height - 2)
        x = random.randint(1, mdp.width - 2)
    return (x, y)

def generate_example(agent, config):
    """Generates an example Gridworld and corresponding agent actions.

    agent: The agent that acts in the generated MDP.
    config: Configuration parameters.

    Returns: A tuple of four items:
      image: Numpy array of size imsize x imsize x 2 (walls and rewards)
      y_coords: Numpy array of integers representing y coordinates (rows).
      x_coords: Numpy array of integers representing x coordinates (columns).
      action_labels: Numpy array of integers representing agent actions.

    y_coords, x_coords and action_labels all have the same length, given by
    config.statebatchsize. For every i < L, the action taken by the agent in
    state (x_coords[i], y_coords[i]) is action_labels[i]. This can be used to
    train a planning module to recreate the actions of the agent.
    """
    expected_length, imsize = config.statebatchsize, config.imsize
    mdp = GridworldMdp.generate_random(imsize, imsize)
    agent.set_mdp(mdp)

    def get_minibatch():
        state = get_random_start_state(mdp)
        action = agent.get_action(state)
        return state, action
    minibatches = [get_minibatch() for _ in range(expected_length)]

    walls, rewards, _ = mdp.convert_to_numpy_input()
    image = np.stack([walls, rewards], axis=-1)
    y_coords = np.array([y for (x, y), _ in minibatches])
    x_coords = np.array([x for (x, y), _ in minibatches])
    action_labels = np.array(
        [Direction.get_number_from_direction(a) for _, a in minibatches])
    return image, y_coords, x_coords, action_labels

def generate_n_examples(n, agent, config):
    """Calls generate_example n times to create a dataset of examples of size n.

    Returns the same four Numpy arrays as generate_example, except that they now
    have shape (n, *previous_shape).
    """
    data = [generate_example(agent, config) for _ in range(n)]
    image, S1, S2, labels = zip(*data)
    return np.array(image), np.array(S1), np.array(S2), np.array(labels)

def generate_gridworld_data(config, num_train=1000, num_test=1000):
    """Generates training and test data for Gridworld data."""
    size = config.statebatchsize
    agent = create_agent(config)
    print('Generating %d training examples' % num_train)
    Xtrain, S1train, S2train, ytrain = generate_n_examples(num_train, agent, config)
    print('Generating %d test examples' % num_test)
    Xtest, S1test, S2test, ytest = generate_n_examples(num_test, agent, config)
    return Xtrain, S1train, S2train, ytrain, Xtest, S1test, S2test, ytest

def create_agent(config):
    """Creates the agent specified in config."""
    if config.agent == 'optimal':
        return agents.OptimalAgent(
            gamma=config.gamma,
            beta=config.beta,
            num_iters=config.num_iters)
    elif config.agent == 'naive':
        return agents.NaiveTimeDiscountingAgent(
            config.max_delay,
            config.hyperbolic_constant,
            gamma=config.gamma,
            beta=config.beta,
            num_iters=config.num_iters)
    elif config.agent == 'sophisticated':
        return agents.SophisticatedTimeDiscountingAgent(
            config.max_delay,
            config.hyperbolic_constant,
            gamma=config.gamma,
            beta=config.beta,
            num_iters=config.num_iters)
    elif config.agent == 'myopic':
        return agents.MyopicAgent(
            config.max_delay,
            gamma=config.gamma,
            beta=config.beta,
            num_iters=config.num_iters)
    raise ValueError('Invalid agent: ' + config.agent)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    args = parser.parse_args()
    if args.seed is None:
        args.seed = int(random.random() * 100000)
    print('Using seed ' + str(args.seed))
    random.seed(args.seed)
    image, y, x, labels = generate_example(8, 8)
    print('Walls:')
    print(image[:,:,0])
    print('Rewards:')
    print(image[:,:,1])
    print('Y coords:')
    print(y)
    print('X coords:')
    print(x)
    print('Optimal actions:')
    print(labels)
