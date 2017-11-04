import argparse
import numpy as np
import random
import csv
import os.path as path

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

def generate_example(expected_length, agent, config, other_agents=[]):
    """Generates an example Gridworld and corresponding agent actions.

    expected_length: The number of state/action pairs to generate.
    agent: The agent that acts in the generated MDP.
    config: Configuration parameters.
    other_agents: List of Agents that we wish to distinguish `agent` from. In
      particular, for every other agent, for our randomly chosen training
      examples, we report the number of examples (states) on which `agent` and
      the other agent would choose different actions.

    Returns: A tuple of six items:
      image: Numpy array of size imsize x imsize, each element is 1 if there is
             a wall at that location, 0 otherwise.
      rewards: Numpy array of size imsize x imsize, each element is the reward
               obtained at that state. (Most will be zero.)
      y_coords: Numpy array of integers representing y coordinates (rows).
      x_coords: Numpy array of integers representing x coordinates (columns).
      action_labels: Numpy array of size len(x_coords) x 5. The probability
                     distributions over actions for each state.
      num_different: Numpy array of size `len(other_agents)`. `num_different[i]`
                     is the number of states where `other_agents[i]` would
                     choose a different action compared to `agent`.

    y_coords, x_coords and action_labels all have the same length, given by
    config.statebatchsize. For every i < L, the action taken by the agent in
    state (x_coords[i], y_coords[i]) is action_labels[i]. This can be used to
    train a planning module to recreate the actions of the agent.
    """
    assert len(other_agents) <= expected_length
    imsize = config.imsize
    num_actions = config.num_actions
    pr_wall, pr_reward = config.wall_prob, config.reward_prob
    if config.simple_mdp:
        mdp = GridworldMdp.generate_random(imsize, imsize, pr_wall, pr_reward)
    else:
        mdp = GridworldMdp.generate_random_connected(imsize, imsize, pr_reward)

    def dist_to_numpy(dist):
        return dist.as_numpy_array(Direction.get_number_from_direction, num_actions)

    def get_minibatch():
        state = mdp.get_random_start_state()
        action_dist = dist_to_numpy(agent.get_action_distribution(state))
        return state, action_dist

    agent.set_mdp(mdp)
    minibatches = [get_minibatch() for _ in range(expected_length)]

    threshold = config.action_distance_threshold
    def calculate_different(other_agent):
        """
        Return the number of states in minibatches on which the action chosen by
        `agent` is different from the action chosen by `other_agent`.
        """
        other_agent.set_mdp(mdp)
        def differs(s, action_dist):
            dist = dist_to_numpy(other_agent.get_action_distribution(s))
            # Two action distributions are "different" if they are sufficiently
            # far away from each other according to some distance metric.
            # TODO(rohinmshah): L2 norm is not the right distance metric for
            # probability distributions, maybe use something else?
            # Not KL divergence, since it may be undefined
            return np.linalg.norm(action_dist - dist) > threshold
        return sum([(1 if differs(s, a) else 0) for s, a in minibatches])

    num_different = np.array([calculate_different(o) for o in other_agents])
    walls, rewards, _ = mdp.convert_to_numpy_input()
    y_coords = np.array([y for (x, y), _ in minibatches])
    x_coords = np.array([x for (x, y), _ in minibatches])
    action_labels = np.array([action_dist for _, action_dist in minibatches])
    return walls, rewards, y_coords, x_coords, action_labels, num_different

def generate_n_examples(n, agent, config, other_agents=[]):
    """Calls generate_example n times to create a dataset of examples of size n.

    Returns the same five Numpy arrays as generate_example, except that they now
    have shape (n, *previous_shape). (The last Numpy array from generate_example
    is analyzed and printed out, and so is not returned.)
    """
    size = config.statebatchsize
    data = [generate_example(size, agent, config, other_agents) for _ in range(n)]
    walls, rewards, S1, S2, labels, num_different = map(np.array, zip(*data))
    num_different = np.array(num_different)
    fraction_different = np.sum(num_different, axis=0) * 1.0 / (n * size)
    print('Fraction of states where agents choose different actions:')
    print(fraction_different)
    return walls, rewards, S1, S2, labels

def generate_gridworld_data(agent, config, other_agents=[]):
    """Generates training and test data for Gridworld data."""
    size = config.statebatchsize
    print('Generating %d training examples' % config.num_train)
    imagetrain, rewardtrain, S1train, S2train, ytrain = generate_n_examples(config.num_train, agent, config, other_agents)
    print('Generating %d test examples' % config.num_test)
    imagetest, rewardtest, S1test, S2test, ytest = generate_n_examples(config.num_test, agent, config, other_agents)
    return imagetrain, rewardtrain, S1train, S2train, ytrain, \
           imagetest, rewardtest, S1test, S2test, ytest

def generate_gridworld_irl(config):
    """Generates an IRL problem for Gridworlds.

    Returns 15 Numpy arrays, from 3 calls to generate_n_examples, corresponding
    to train data, test data for step 1, and test data for step 2.
    """
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

    step1_data = generate_gridworld_data(agent, config, other_agents)
    num_mdps = config.num_mdps
    print('Generating %d unknown reward examples' % num_mdps)
    step2_data = generate_n_examples(num_mdps, agent, config, other_agents)
    return step1_data + step2_data

def create_agent(agent, gamma, beta, num_iters, max_delay, hyperbolic_constant):
    """Creates the agent specified in config."""
    if agent == 'optimal':
        return agents.OptimalAgent(
            gamma=gamma,
            beta=beta,
            num_iters=num_iters)
    elif agent == 'naive':
        return agents.NaiveTimeDiscountingAgent(
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

def save_dataset(config, filename):
    np.savez(filename, generate_gridworld_irl(config))

def load_dataset(filename):
    # imagetrain, rewardtrain, S1train, S2train, ytrain, \
    # imagetest1, rewardtest1, S1test1, S2test1, ytest1, \
    # imagetest2, rewardtest2, S1test2, S2test2, ytest2 = np.load(filename)
    return np.load(filename)

def create_dataset_repo(foldername, config, imsizes, rewardprobs, seeds=[32,1729,7,4],agents=['optimal', 'naive', 'sophisticated', 'myopic'], epochs=[100], num_train=[5000], num_test=[1000], batchsize=[32]):
    """Store dataset files into folder of numpy arrays.
    Create index csv that tells us what each numpy array contains.
    """
    with open(path.join(foldername,"index.csv"), 'w', newline='') as indexfile:
        csvfile = csv.writer(indexfile)
        i = 0
        trainsize = max(num_train)
        testsize = max(num_test)
        for imsize in imsizes:
            for rewardp in rewardprobs:
                for seed in seeds:
                    for agent in agents:
                        i+=1
                        config.imsize=imsize
                        config.seed = seed
                        config.reward_prob = rewardp
                        config.agent = agent
                        config.epochs = epochs[0]
                        config.num_train = trainsize
                        config.num_test = testsize
                        save_dataset(config, path.join(foldername, "{}.npz".format(i)))
                        csvfile.writerow([i, imsize, wallp, rewardp, agent, epoch, num_train, num_test])

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--imsize', type=int, default=8)
    parser.add_argument('--wall_prob', type=float, default=0.05)
    parser.add_argument('--reward_prob', type=float, default=0.05)
    parser.add_argument('--statebatchsize', type=int, default=10)
    args = parser.parse_args()
    if args.seed is None:
        args.seed = int(random.random() * 100000)
    print('Using seed ' + str(args.seed))
    random.seed(args.seed)
    agent = agents.OptimalAgent()
    other_agent = agents.NaiveTimeDiscountingAgent(20, 1)
    walls, rewards, y, x, labels, num_different = generate_example(10, agent, args, [other_agent])
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
    print('Fraction of actions that are different across agents: ' + str(num_different[0] / 10.0))
