import argparse
import numpy as np
import random

import agents
from agent_runner import run_agent
from gridworld import GridworldMdp, GridworldEnvironment, Direction

# TODO(rohinmshah): This really belongs in gridworld.py
def get_random_start_state(mdp):
    """Returns a state in mdp that would be a legal start state for an agent.
    Avoids walls and reward/exit states.
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
      image: Numpy array of size imsize x imsize, each element is 1 if there is
             a wall at that location, 0 otherwise.
      rewards: Numpy array of size imsize x imsize, each element is the reward
               obtained at that state. (Most will be zero.)
      y_coords: Numpy array of integers representing y coordinates (rows).
      x_coords: Numpy array of integers representing x coordinates (columns).
      action_labels: Numpy array of integers representing agent actions.

    y_coords, x_coords and action_labels all have the same length, given by
    config.statebatchsize. For every i < L, the action taken by the agent in
    state (x_coords[i], y_coords[i]) is action_labels[i]. This can be used to
    train a planning module to recreate the actions of the agent.
    """
    expected_length, imsize = config.statebatchsize, config.imsize
    pr_wall, pr_reward = config.wall_prob, config.reward_prob
    mdp = GridworldMdp.generate_random(imsize, imsize, pr_wall, pr_reward)
    agent.set_mdp(mdp)

    def get_minibatch():
        state = get_random_start_state(mdp)
        action = agent.get_action(state)
        return state, action
    minibatches = [get_minibatch() for _ in range(expected_length)]

    walls, rewards, _ = mdp.convert_to_numpy_input()
    y_coords = np.array([y for (x, y), _ in minibatches])
    x_coords = np.array([x for (x, y), _ in minibatches])
    action_labels = np.array(
        [Direction.get_number_from_direction(a) for _, a in minibatches])
    return walls, rewards, y_coords, x_coords, action_labels

def generate_n_examples(n, agent, config):
    """Calls generate_example n times to create a dataset of examples of size n.

    Returns the same five Numpy arrays as generate_example, except that they now
    have shape (n, *previous_shape).
    """
    data = [generate_example(agent, config) for _ in range(n)]
    walls, rewards, S1, S2, labels = map(np.array, zip(*data))
    return walls, rewards, S1, S2, labels

def generate_gridworld_data(agent, config, num_train=1000, num_test=100):
    """Generates training and test data for Gridworld data."""
    size = config.statebatchsize
    print('Generating %d training examples' % num_train)
    imagetrain, rewardtrain, S1train, S2train, ytrain = generate_n_examples(num_train, agent, config)
    print('Generating %d test examples' % num_test)
    imagetest, rewardtest, S1test, S2test, ytest = generate_n_examples(num_test, agent, config)
    return imagetrain, rewardtrain, S1train, S2train, ytrain, imagetest, rewardtest, S1test, S2test, ytest

def generate_gridworld_irl(config, num_train=1000, num_test=100, num_mdps=10):
    """Generates an IRL problem for Gridworlds.

    Returns 15 Numpy arrays, from 3 calls to generate_n_examples, corresponding
    to train data, test data for step 1, and test data for step 2.
    """
    agent = create_agent(config)
    step1_data = generate_gridworld_data(agent, config, num_train, num_test)
    print('Generating %d unknown reward examples' % num_mdps)
    step2_data = generate_n_examples(num_mdps, agent, config)
    return step1_data + step2_data

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
    parser.add_argument('--imsize', type=int, default=8)
    parser.add_argument('--wall_prob', type=float, default=0.05)
    parser.add_argument('--reward_prob', type=float, default=0)
    parser.add_argument('--statebatchsize', type=int, default=10)
    args = parser.parse_args()
    if args.seed is None:
        args.seed = int(random.random() * 100000)
    print('Using seed ' + str(args.seed))
    random.seed(args.seed)
    walls, rewards, y, x, labels = generate_example(agents.OptimalAgent(), args)
    print('Walls:')
    print(walls)
    print('Rewards:')
    print(rewards)
    print('Y coords:')
    print(y)
    print('X coords:')
    print(x)
    print('Optimal actions:')
    print(labels)
