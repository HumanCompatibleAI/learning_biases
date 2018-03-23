import tensorflow as tf
import numpy as np
import random

from fast_agents import FastOptimalAgent
from gridworld import GridworldMdpNoR, GridworldMdp, Direction
from gridworld_data import load_dataset
from utils import plot_reward, set_seeds


class GridworldMdpLearnableR(GridworldMdpNoR):
    """A gridworld in whith the reward is not known and is to be learned."""

    def __init__(self, walls, start_state, living_reward=-0.01, noise=0):
        GridworldMdpNoR.__init__(self, walls, start_state, noise)
        self.living_reward = living_reward
        self.weights = None

    def get_reward(self, state, action):
        if action != Direction.EXIT:
            return self.living_reward
        x, y = state
        state_num = y * self.width + x
        return self.weights[state_num]

    def policy_log_likelihood(self, policy, weights, beta):
        self.weights = weights
        agent = FastOptimalAgent()
        agent.set_mdp(self)

        def qval(state):
            if self.is_terminal(state):
                return 0
            x, y = state
            action = Direction.get_direction_from_number(policy[y][x])
            return agent.qvalue(state, action)

        result = beta * sum([qval(s) for s in self.get_states()])
        self.weights = None
        return result

    @staticmethod
    def from_full_mdp(mdp):
        return GridworldMdpLearnableR(
            mdp.walls, mdp.start_state, mdp.living_reward, mdp.noise)


def gaussian_prior(prior_weight=100):
    def log_likelihood(weights):
        return -prior_weight * weights.dot(weights) / 2
    return log_likelihood


def birl(mdp, policy, beta, prior=gaussian_prior(), num_samples=1000, num_burn_in=100,
         sigma_sq=0.1, display_step=100):
    """Runs Bayesian Inverse Reinforcement Learning to infer the reward function.

    mdp: A GridworldMdp whose reward is to be inferred.
    policy: A tensor of shape mdp.height x mdp.width x num_actions. Specifies a
            probability distribution over actions for each state.
    beta: Rationality parameter.
    prior: A function that maps weights defining a reward function to the prior
           log probability of that reward function. (The log probabilities are
           allowed to be scaled by an arbitrary constant.)
    num_samples: Number of samples to take.
    num_burn_in: Number of samples in the burn in phase (not used when computing
                 the return value).
    sigma_sq: The variance to use when computing a neighbor of the current
              reward function.

    Returns: An estimate of the reward (the mean of the posterior)
    """
    policy = np.argmax(policy, axis=2)

    def get_log_likelihood(weights):
        policy_ll = mdp.policy_log_likelihood(policy, weights, beta)
        prior_ll = prior(weights)
        return prior_ll + policy_ll

    def sample(curr_weights, curr_ll):
        new_weights = np.random.multivariate_normal(
            curr_weights, sigma_sq * np.eye(len(curr_weights)))
        new_ll = get_log_likelihood(new_weights)
        accept_prob = np.exp(np.min(new_ll - curr_ll, 0))
        if np.random.rand() < accept_prob:
            return new_weights, new_ll, True
        return curr_weights, curr_ll, False

    samples, log_likelihoods, accept_flags = [], [], []
    weights = np.random.rand(mdp.height * mdp.width)
    log_likelihood = get_log_likelihood(weights)

    for i in range(num_burn_in):
        if i % display_step == 0:
            print('Burn in iteration ' + str(i))
        weights, log_likelihood, _ = sample(weights, log_likelihood)
        log_likelihoods.append(log_likelihood)

    for i in range(num_samples):
        if i % display_step == 0:
            print('Sampling iteration ' + str(i))
        weights, log_likelihood, accept = sample(weights, log_likelihood)
        samples.append(weights)
        log_likelihoods.append(log_likelihood)
        accept_flags.append(accept)

    return np.mean(np.array(samples), axis=0)


def init_birl_flags():
    tf.app.flags.DEFINE_string('datafile', None, 'Where to get data from')
    tf.app.flags.DEFINE_float('beta', 1, 'Rationality parameter')
    tf.app.flags.DEFINE_integer(
        'seed', 0, 'Random seed for both numpy and random')
    tf.app.flags.DEFINE_integer(
        'num_burn_in', 100,
        'Number of samples to take during the burn in period')
    tf.app.flags.DEFINE_integer(
        'num_samples', 1000,
        'Number of samples to take while collecting statistics')
    tf.app.flags.DEFINE_integer(
        'display_step', 100, 'Print summary output every n epochs')
    return tf.app.flags.FLAGS

if __name__=='__main__':
    # get flags || Data
    config = init_birl_flags()
    if config.datafile is None:
        print('--datafile option is required')
        exit()

    # seed random generators
    set_seeds(config.seed)

    imagetest, rewardtest, ytest = load_dataset(config.datafile)[-3:]
    for image, reward, policy in zip(imagetest, rewardtest, ytest):
        mdp = GridworldMdp.from_numpy_input(image, reward)
        mdp = GridworldMdpLearnableR.from_full_mdp(mdp)
        inferred_reward = birl(
            mdp, policy, config.beta, num_burn_in=config.num_burn_in,
            num_samples=config.num_samples, display_step=config.display_step)

        print('The first set of walls is:')
        print(image)
        print('The first reward should be:')
        print(reward)
        inferred_reward = inferred_reward / inferred_reward.max()
        inferred_reward = np.reshape(inferred_reward, image.shape)
        print('The inferred reward is:')
        print(inferred_reward)

        plot_reward(reward, inferred_reward)
        break
