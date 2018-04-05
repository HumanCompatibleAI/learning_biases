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
    initial_states(array): no clue
    """
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

def irl(mdp, agent, discount, horizon,
        planner=max_causal_ent_policy, optimizer=None, scheduler=None,
        num_iter=5000, log_every=100):
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
    transition = mdp.get_transition_matrix()
    initial_states = np.zeros(len(transition))
    nS, _, _ = transition.shape

    demo_counts = get_demo_counts(agent)

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

        if i % log_every == 0:
            end = time()
            print("Time elapsed from trials {} to {}: {}".format(i*log_every, (i+1)*log_every, end-start))

    return reward.data.numpy()

def irl_on_grid(grid, agent, discount, horizon,
        planner=max_causal_ent_policy, optimizer=None, scheduler=None,
        num_iter=5000, log_every=1000):
    """Wrapper for irl method"""
    start = (1,1)
    mdp = GridworldMdpNoR(grid=grid, start_state=start)
    return irl(mdp, agent, discount, horizon, planner, optimizer, scheduler, num_iter, log_every)

def get_demo_counts(agent):
    # TODO: compute the agent's values, rewrite irl to accept MDPReward & Agents which use that MDP to plan
    pass

def test():
    walls = [[1,1,1,1],
             [1,0,0,1],
             [1,0,0,1],
             [1,1,1,1]]
    walls = np.array(walls)
    start = (1,1)
    mdp = GridworldMdpNoR(walls, start)
    print(mdp.get_transition_matrix()[4:])

if __name__ == '__main__':
    test()
