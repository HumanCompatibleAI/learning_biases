import argparse
import numpy as np
import random
import csv
import os

import agents
import fast_agents
from agent_runner import run_agent
from gridworld import GridworldMdp, Direction
from mdp_interface import Mdp
from utils import Distribution, init_flags

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

def generate_example(agent, config, other_agents=[]):
    """Generates an example Gridworld and corresponding agent actions.

    agent: The agent that acts in the generated MDP.
    config: Configuration parameters.
    other_agents: List of Agents that we wish to distinguish `agent` from. In
      particular, for every other agent, for our randomly chosen training
      examples, we report the number of examples (states) on which `agent` and
      the other agent would choose different actions.

    Returns: A tuple of five items:
      image: Numpy array of size imsize x imsize, each element is 1 if there is
             a wall at that location, 0 otherwise.
      rewards: Numpy array of size imsize x imsize, each element is the reward
               obtained at that state. (Most will be zero.)
      start_state: The starting state for the gridworld (a tuple (x, y)).
      action_dists: Numpy array of size imsize x imsize x num_actions. The
                    probability distributions over actions for each state.
      num_different: Numpy array of size `len(other_agents)`. `num_different[i]`
                     is the number of states where `other_agents[i]` would
                     choose a different action compared to `agent`.
    
    For every i < L, the action taken by the agent in state (x, y) is drawn from
    the distribution action_dists[x, y, :]. This can be used to train a planning
    module to recreate the actions of the agent.
    """
    imsize = config.imsize
    num_actions = config.num_actions
    pr_wall, pr_reward = config.wall_prob, config.reward_prob
    if config.simple_mdp:
        mdp = GridworldMdp.generate_random(imsize, imsize, pr_wall, pr_reward)
    else:
        mdp = GridworldMdp.generate_random_connected(imsize, imsize, pr_reward)

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

    agent.set_mdp(mdp)
    action_dists = [[action((x, y)) for x in range(imsize)] for y in range(imsize)]
    action_dists = np.array(action_dists)

    def calculate_different(other_agent):
        """
        Return the number of states in minibatches on which the action chosen by
        `agent` is different from the action chosen by `other_agent`.
        """
        other_agent.set_mdp(mdp)
        def differs(s):
            x, y = s
            action_dist = action_dists[y][x]
            dist = dist_to_numpy(other_agent.get_action_distribution(s))
            # Two action distributions are "different" if they are sufficiently
            # far away from each other according to some distance metric.
            # TODO(rohinmshah): L2 norm is not the right distance metric for
            # probability distributions, maybe use something else?
            # Not KL divergence, since it may be undefined
            return np.linalg.norm(action_dist - dist) > config.action_distance_threshold
        return sum([sum([(1 if differs((x, y)) else 0) for x in range(imsize)]) for y in range(imsize)])

    num_different = np.array([calculate_different(o) for o in other_agents])
    walls, rewards, start_state = mdp.convert_to_numpy_input()
    return walls, rewards, start_state, action_dists, num_different

def get_filename(n, agent, config, seed):
    pattern = 'gridworlds-v1-seed-{0}-num-{1}-agent-{2}-imsize-{3.imsize}-wallprob-{3.wall_prob}-rewardprob-{3.reward_prob}-simplemdp-{3.simple_mdp}.npz'
    return pattern.format(seed, n, agent, config)

def save_dataset(filename, dataset):
    np.savez(filename, *dataset)

def load_dataset(filename):
    """ Load dataset unpacks the numpy array with all the gridworld files"""
    data = np.load(filename)
    return tuple([data['arr_{}'.format(i)] for i in range(4)])

def generate_n_examples(n, agent, config, seed=0, other_agents=[], folder='datasets/'):
    """Calls generate_example n times to create a dataset of examples of size n.

    Returns the same four Numpy arrays as generate_example, except that they
    now have shape (n, *previous_shape). (The last Numpy array from
    generate_example is analyzed and printed out, and so is not returned.)
    """
    filename = folder + get_filename(n, agent, config, seed)
    if os.path.exists(filename):
        dataset = load_dataset(filename)
        print('Reusing existing dataset')
        return dataset

    print('Could not find ' + filename)
    print('Generating {} examples'.format(n))
    np.random.seed(seed)
    random.seed(seed)
    data = [generate_example(agent, config, other_agents) for _ in range(n)]
    walls, rewards, start_states, labels, num_different = map(np.array, zip(*data))
    if other_agents:
        num_different = np.array(num_different)
        num_states = (n * config.imsize * config.imsize)
        fraction_different = float(np.sum(num_different, axis=0)) / num_states
        print('Fraction of states where agents choose different actions: '
              + str(fraction_different))

    dataset = walls, rewards, start_states, labels
    save_dataset(filename, dataset)
    return dataset

def generate_data_for_planner(agent, config, other_agents=[]):
    """Generates training and test data for Gridworld data.

    Returns a tuple of two elements, each of which is the return value of a call
    to generate_n_examples)."""
    train_data = generate_n_examples(
        config.num_train, agent, config, config.seeds.pop(0), other_agents)
    test_data = generate_n_examples(
        config.num_test, agent, config, config.seeds.pop(0), other_agents)
    return train_data, test_data

def generate_data_for_reward(agent, config, other_agents=[]):
    """Generates an IRL problem for Gridworlds.

    Returns 12 Numpy arrays, from 3 calls to generate_n_examples, corresponding
    to train data, test data for step 1, and test data for step 2.
    """
    return generate_n_examples(config.num_mdps, agent, config, config.seeds.pop(0), other_agents)

def create_agents_from_config(config):
    agent = create_agent(
        config.agent, config.gamma, config.beta,
        config.num_iters, config.max_delay,
        config.hyperbolic_constant)
    other_agents = []
    if config.other_agent is not None:
        other_agent = create_agent(
            config.other_agent, config.other_gamma, config.other_beta,
            config.other_num_iters, config.other_max_delay,
            config.other_hyperbolic_constant)
        other_agents.append(other_agent)

    return agent, other_agents

def create_agent(agent, gamma, beta, num_iters, max_delay, hyperbolic_constant):
    """Creates the agent specified in config."""
    if agent == 'optimal':
        return fast_agents.FastOptimalAgent(
            gamma=gamma,
            beta=beta,
            num_iters=num_iters)
    elif agent == 'naive':
        return fast_agents.FastNaiveTimeDiscountingAgent(
            max_delay,
            hyperbolic_constant,
            gamma=gamma,
            beta=beta,
            num_iters=num_iters)
    elif agent == 'sophisticated':
        return agents.SophisticatedTimeDiscountingAgent(
            max_delay,
            hyperbolic_constant,
            gamma=gamma,
            beta=beta,
            num_iters=num_iters)
    elif agent == 'myopic':
        return agents.MyopicAgent(
            max_delay,
            gamma=gamma,
            beta=beta,
            num_iters=num_iters)
    raise ValueError('Invalid agent: ' + agent)


if __name__ == '__main__':
    config = init_flags()
    agent, other_agents = create_agents_from_config(config)
    generate_data_for_reward(agent, config, other_agents)
