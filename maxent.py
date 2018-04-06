"""
Source @Adam's population-irl repo
Implements a tabular version of maximum entropy inverse reinforcement learning
(Ziebart et al, 2008). There are two key differences from the version
described in the paper:
  - We do not make use of features, which are not needed in the tabular setting.
  - We use Adam rather than exponentiated gradient descent.
"""

import functools

import numpy as np
from scipy.special import logsumexp as sp_lse
from time import time
import torch
from torch.autograd import Variable

from gridworld import GridworldMdpNoR
from gridworld_data import generate_example
#TODO: fully torchize?

def max_ent_policy(transition, reward, horizon, discount):
    """Backward pass of algorithm 1 of Ziebart (2008).
       This corresponds to maximum entropy.
       WARNING: You probably want to use max_causal_ent_policy instead.
       See discussion in section 6.2.2 of Ziebart's PhD thesis (2010)."""
    nS = transition.shape[0]
    logsc = np.zeros(nS)  # TODO: terminal states only?
    with np.warnings.catch_warnings():
        np.warnings.filterwarnings('ignore', 'divide by zero encountered in log')
        logt = np.nan_to_num(np.log(transition))
    reward = reward.reshape(nS, 1, 1)
    for i in range(horizon):
        # Ziebart (2008) never describes how to handle discounting. This is a
        # backward pass: so on the i'th iteration, we are computing the
        # frequency a state/action is visited at the (horizon-i-1)'th position.
        # So we should multiply reward by discount ** (horizon - i - 1).
        cur_discount = discount ** (horizon - i - 1)
        x = logt + (cur_discount * reward) + logsc.reshape(1, 1, nS)
        logac = sp_lse(x, axis=2)
        logsc = sp_lse(logac, axis=1)
    return np.exp(logac - logsc.reshape(nS, 1))

def max_causal_ent_policy(transition, reward, horizon, discount):
    """Soft Q-iteration, theorem 6.8 of Ziebart's PhD thesis (2010)."""
    nS, nA, _ = transition.shape
    V = np.zeros(nS)
    for i in range(horizon):
        Q = reward.reshape(nS, 1) + discount * (transition * V).sum(2)
        V = sp_lse(Q, axis=1)
    return np.exp(Q - V.reshape(nS, 1))

def expected_counts(policy, transition, initial_states, horizon, discount):
    """Forward pass of algorithm 1 of Ziebart (2008).
    policy(array): 3D matrix of 2D grid, last channel is action prob channel
    transition(array): 2D array (number states, prob(s' | a) for all a)
    initial_states(array): 1D array of probability distribution over the initial states
    """
    sumtest = np.sum(initial_states)
    assert np.isclose(sumtest, 1), "Initial states is a pdf over states. Should sum to 1. Currently: {}".format(sumtest)

    nS = transition.shape[0]
    counts = np.zeros((nS, horizon + 1))
    counts[:, 0] = initial_states
    for i in range(1, horizon + 1):
        counts[:, i] = np.einsum('i,ij,ijk->k', counts[:, i-1],
                                 policy, transition) * discount
    if discount == 1:
        renorm = horizon + 1
    else:
        renorm = (1 - discount ** (horizon + 1)) / (1 - discount)
    return np.sum(counts, axis=1) / renorm

def policy_loss(policy, trajectories):
    loss = 0
    log_policy = np.log(policy)
    for states, actions in trajectories:
        loss += np.sum(log_policy[states, actions])
    return loss

default_optimizer = functools.partial(torch.optim.Adam, lr=1e-1)
default_scheduler = {
    max_ent_policy: functools.partial(
        torch.optim.lr_scheduler.ExponentialLR, gamma=1.0
    ),
    max_causal_ent_policy: functools.partial(
        torch.optim.lr_scheduler.ExponentialLR, gamma=0.999
    ),
}


def _irl(transition, policy, initial_states, horizon, discount, start_state,
        planner=max_causal_ent_policy, optimizer=None, scheduler=None,
        num_iter=5000, log_every=100, verbose=False):
    """
    Args:
        - mdp(TabularMdpEnv): MDP trajectories were drawn from.
        - trajectories(list): expert trajectories; exclusive with demo_counts.
            List containing one (states, actions) pair for each trajectory,
            where states and actions are lists containing all visited
            states/actions in that trajectory.
        - discount(float): between 0 and 1.
            Should match that of the agent generating the trajectories.
        - demo_counts(array): expert visitation frequency; exclusive with trajectories.
            The expected visitation frequency of the optimal policy.
            Must supply horizon with this argument.
        - horizon(int): optional, must be supplied if demo_counts used.
        - planner(callable): max_ent_policy or max_causal_ent_policy.
        - optimizer(callable): a callable returning a torch.optim object.
            The callable is called with an iterable of parameters to optimize.
        - scheduler(callable): a callable returning a torch.optim.lr_scheduler.
            The callable is called with a torch.optim optimizer object.
        - learning_rate(float): for Adam optimizer.
        - num_iter(int): number of iterations of optimization process.
    Returns (reward, info) where:
        reward(list): estimated reward for each state in the MDP.
        info(dict): log of extra info.
    """
    assert len(start_state) == 2, "Only support start_states with len 2, of form [x, y]"
    # Assuming policy is of shape [imsize, imsize, num_actions]
    start_idx = start_state[0]*len(policy) + start_state[1]
    nS, _, _ = transition.shape

    policy = flatten_policy(policy)
    demo_counts = expected_counts(policy, transition, initial_states, horizon, discount)

    reward = Variable(torch.zeros(nS), requires_grad=True)
    if optimizer is None:
        optimizer = default_optimizer
    if scheduler is None:
        scheduler = default_scheduler[planner]
    optimizer = optimizer([reward])
    scheduler = scheduler(optimizer)

    start = time()
    for i in range(num_iter):
        pol = planner(transition, reward.data.numpy(), horizon, discount)
        ec = expected_counts(pol, transition, initial_states, horizon, discount)
        optimizer.zero_grad()
        reward.grad = Variable(torch.Tensor(ec - demo_counts))
        optimizer.step()
        scheduler.step()

        if i % log_every == 0 and verbose:
            end = time()
            print("Time elapsed from trials {} to {}: {}".format(i, i+log_every, end-start))
            start = time()

    return reward.data.numpy()


def irl_wrapper(image, action_dists, start, config, verbose=False):
    """Generate max_causal_ent wrapper for generate_example"""
    horizon = config.horizon
    discount = config.gamma

    return _irl_wrapper(image, action_dists, start, horizon, discount, verbose=verbose)

def _irl_wrapper(image, action_dists, start, horizon, discount, verbose=False):
    imsize = len(image)
    transition = GridworldMdpNoR(image, [1, 1]).get_transition_matrix()
    policy = action_dists

    # initial_states = np.ones(image.shape) - image
    # initial_states = initial_states / np.sum(initial_states)
    initial_states = np.zeros(image.shape)
    initial_states[start[1], start[0]] = 1
    initial_states = np.reshape(initial_states, -1)
    flat_inferred = _irl(transition, policy, initial_states, horizon, discount,
                         start_state=start, verbose=verbose)
    inferred_reward = np.reshape(flat_inferred, (imsize, imsize)).T
    return inferred_reward


def flatten_policy(policy):
    """Reshapes the policy
    (imsize, imsize, num_actions)
    to
    (imsize**2, num_actions)
    """

    policy = policy.reshape(len(policy)**2, -1)
    # print(np.sum(policy, axis=-1))
    assert (np.isclose(np.sum(policy, axis=-1), 1)).all(), "error while reshaping"
    return policy


def test():
    """tests creation of the mdpnor.get_transition_matrix() method, visually :)"""
    walls = [[1,1,1,1],
             [1,0,0,1],
             [1,0,0,1],
             [1,1,1,1]]
    walls = np.array(walls)
    start = (1,1)
    mdp = GridworldMdpNoR(walls, start)
    print(mdp.get_transition_matrix()[4:])

def test_irl():
    from agents import OptimalAgent
    from gridworld import GridworldMdp, Direction
    from utils import Distribution

    num_actions = len(Direction.ALL_DIRECTIONS)

    agent = OptimalAgent(beta=10.0)
    grid = [['X','X','X','X'],
             ['X','A',1,'X'],
             ['X',' ',' ','X'],
             ['X','X','X','X']]
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

    action_dists = [[action((x, y)) for y in range(imsize)] for x in range(imsize)]
    action_dists = np.array(action_dists)

    walls, rewards, start_state = mdp.convert_to_numpy_input()

    inferred = _irl_wrapper(walls, action_dists, start_state, 20, 1.0)
    print(inferred)
    print("---true below---")
    print(rewards)



if __name__ == '__main__':
    # test()
    test_irl()
