from agent_interface import Agent
from collections import defaultdict
from utils import Distribution
import numpy as np
import random
import pdb

class ValueIterationLikeAgent(Agent):
    """An agent that chooses actions using something similar to value iteration.

    Instead of working directly on states from the mdp, we perform value
    iteration on generalized states (called mus), following the formalism in
    "Learning the Preferences of Bounded Agents" from a NIPS 2015 workshop.

    In the default case, a single MDP provides all of the necessary
    information. However, to support evaluation of reward learning, you can
    optionally specify a reward_mdp in the set_mdp method, in which case all
    reward evaluations will be done by the reward_mdp (while everything else
    such as transition probabilities will still use the original MDP).

    The algorithm in this class is simply standard value iteration, but
    subclasses can easily change the behavior while reusing most of the code by
    overriding hooks into the algorithm.
    """

    def __init__(self, gamma=0.9, beta=None, num_iters=50):
        """Initializes the agent, setting any relevant hyperparameters.

        gamma: Discount factor.
        beta: Noise parameter when choosing actions. beta=None implies that
        there is no noise, otherwise actions are chosen with probability
        proportional to exp(beta * value).
        num_iters: The maximum number of iterations of value iteration to run.
        """
        super(ValueIterationLikeAgent, self).__init__(gamma)
        self.beta = beta
        self.num_iters = num_iters
        self.policy = None

    def set_mdp(self, mdp, reward_mdp=None):
        super(ValueIterationLikeAgent, self).set_mdp(mdp)
        self.reward_mdp = reward_mdp if reward_mdp is not None else mdp
        self.compute_values()

    def compute_values(self):
        """Computes the values for self.mdp using value iteration.

        Populates an object self.values, such that self.values[mu] is the value
        (a float) of the generalized state mu.
        """
        values = defaultdict(float)
        for iter in range(self.num_iters):
            new_values = defaultdict(float)
            for mu in self.get_mus():
                actions = self.get_actions(mu)
                if not actions:
                    continue
                new_mu = self.get_mu_for_planning(mu)  # Typically new_mu == mu
                qvalues = [(self.qvalue(new_mu, a, values), a) for a in actions]
                _, chosen_action = max(qvalues)
                new_values[mu] = self.qvalue(mu, chosen_action, values)

            if self.converged(values, new_values):
                self.values = new_values
                return

            values = new_values

        self.values = values

    def converged(self, values, new_values, tolerance=1e-3):
        """Returns True if value iteration has converged.

        Value iteration has converged if no value has changed by more than tolerance.

        values: The values from the previous iteration of value iteration.
        new_values: The new value computed during this iteration.
        """
        for mu in new_values.keys():
            if abs(values[mu] - new_values[mu]) > tolerance:
                return False
        return True

    def value(self, mu):
        """Computes V(mu).

        mu: Generalized state
        """
        return self.values[mu]

    def qvalue(self, mu, a, values=None):
        """Computes Q(mu, a) from the values table.

        mu: Generalized state
        a: Action
        values: Dictionary such that values[mu] is the value of generalized
        state mu. If None, then self.values is used instead.
        """
        if values is None:
            values = self.values
        r = self.get_reward(mu, a)
        transitions = self.get_transition_mus_and_probs(mu, a)
        return r + self.gamma * sum([p * values[mu2] for mu2, p in transitions])

    def get_action_distribution(self, s):
        """Returns a Distribution over actions.

        Note that this is a normal state s, not a generalized state mu.
        """
        mu = self.extend_state_to_mu(s)
        actions = self.mdp.get_actions(s)
        if self.beta is not None:
            q_vals = np.array([self.qvalue(mu, a) for a in actions])
            q_vals = q_vals - np.mean(q_vals)  # To prevent overflow in exp
            action_dist = np.exp(self.beta * q_vals)
            return Distribution(dict(zip(actions, action_dist)))

        best_value, best_actions = float("-inf"), []
        for a in actions:
            action_value = self.qvalue(mu, a)
            if action_value > best_value:
                best_value, best_actions = action_value, [a]
            elif action_value == best_value:
                best_actions.append(a)
        return Distribution({a : 1 for a in best_actions})
        # For more determinism, you can break ties deterministically:
        # return Distribution({best_actions[0] : 1})

    def get_mus(self):
        """Returns all possible generalized states the agent could be in.

        This is the equivalent of self.mdp.get_states() for generalized states.
        """
        return self.mdp.get_states()

    def get_actions(self, mu):
        """Returns all actions the agent could take from generalized state mu.

        This is the equivalent of self.mdp.get_actions() for generalized states.
        """
        s = self.extract_state_from_mu(mu)
        return self.mdp.get_actions(s)

    def get_reward(self, mu, a):
        """Returns the reward for taking action a from generalized state mu.

        This is the equivalent of self.mdp.get_reward() for generalized states.
        """
        s = self.extract_state_from_mu(mu)
        return self.reward_mdp.get_reward(s, a)

    def get_transition_mus_and_probs(self, mu, a):
        """Gets information about possible transitions for the action.

        This is the equivalent of self.mdp.get_transition_states_and_probs() for
        generalized states. So, it returns a list of (next_mu, prob) pairs,
        where next_mu must be a generalized state.
        """
        s = self.extract_state_from_mu(mu)
        return self.mdp.get_transition_states_and_probs(s, a)

    def get_mu_for_planning(self, mu):
        """Returns the generalized state that an agent uses for planning.

        Specifically, the returned state is used when looking forward to find
        the expected value of a future state.
        """
        return mu

    def extend_state_to_mu(self, state):
        """Converts a normal state to a generalized state."""
        return state

    def extract_state_from_mu(self, mu):
        """Converts a generalized state to a normal state."""
        return mu

class OptimalAgent(ValueIterationLikeAgent):
    """An agent that implements regular value iteration."""
    def __str__(self):
        pattern = 'Optimal-gamma-{0.gamma}-beta-{0.beta}-numiters-{0.num_iters}'
        return pattern.format(self)

class DelayDependentAgent(ValueIterationLikeAgent):
    """An agent that plans differently as it looks further in the future.

    Delay dependent agents calculate values differently as they look further
    into the future. They extend the state with the delay d. Intuitively, the
    generalized state (s, d) stands for "the agent is looking d steps into the
    future at which point is in state s".

    This class is not a full agent. It simply overrides the necessary methods in
    order to support generalized states containing the delay. Subclasses must
    override other methods in order to actually use the delay to change the
    value iteration algorithm in some way.
    """

    def __init__(self, max_delay=None, gamma=0.9, beta=None, num_iters=50):
        """Initializes the agent, setting any relevant hyperparameters.

        max_delay: Integer specifying the maximum value of d to consider during
        planning. If None, then max_delay is set equal to num_iters, which will
        ensure that the values for generalized states of the form (s, 0) are not
        affected by the max_delay. Note that large values of max_delay can cause
        a significant performance overhead.
        """
        super(DelayDependentAgent, self).__init__(gamma, beta, num_iters)
        self.max_delay = max_delay if max_delay is not None else num_iters

    def get_mus(self):
        """Override to handle states with delays."""
        states = self.mdp.get_states()
        return [s + (d,) for s in states for d in range(self.max_delay + 1)]

    def get_transition_mus_and_probs(self, mu, a):
        """Override to handle states with delays."""
        x, y, d = mu
        transitions = self.mdp.get_transition_states_and_probs((x, y), a)
        newd = min(d + 1, self.max_delay)
        return [((x2, y2, newd), p) for (x2, y2), p in transitions]

    def extend_state_to_mu(self, state):
        """Override to handle states with delays."""
        return state + (0,)

    def extract_state_from_mu(self, mu):
        """Override to handle states with delays."""
        return (mu[0], mu[1])

class TimeDiscountingAgent(DelayDependentAgent):
    """A hyperbolic time discounting agent.

    Such an agent discounts future rewards in a time inconsistent way. If they
    would get future reward R, they instead plan as though they would get future
    reward R/(1 + kd). See the paper "Learning the Preferences of Ignorant,
    Inconsistent Agents" for more details.
    """

    def __init__(self, max_delay, discount_constant,
                 gamma=0.9, beta=None, num_iters=50):
        """Initializes the agent, setting any relevant hyperparameters.

        discount_constant: Float. The parameter k in R/(1 + kd) (see above).
        """
        super(TimeDiscountingAgent, self).__init__(
            max_delay, gamma, beta, num_iters)
        self.discount_constant = discount_constant

    def get_reward(self, mu, a):
        """Override to apply hyperbolic time discounting."""
        x, y, d = mu
        discount = (1.0 / (1.0 + self.discount_constant * d))
        return discount * self.mdp.get_reward((x, y), a)

class NaiveTimeDiscountingAgent(TimeDiscountingAgent):
    """The naive time discounting agent.

    See the paper "Learning the Preferences of Ignorant, Inconsistent Agents"
    for more details.
    """
    def __str__(self):
        pattern = 'Naive-maxdelay-{0.max_delay}-discountconst-{0.discount_constant}-gamma-{0.gamma}-beta-{0.beta}-numiters-{0.num_iters}'
        return pattern.format(self)

class SophisticatedTimeDiscountingAgent(TimeDiscountingAgent):
    """The sophisticated time discounting agent.

    See the paper "Learning the Preferences of Ignorant, Inconsistent Agents"
    for more details.
    """
    def get_mu_for_planning(self, mu):
        """Override to implement sophisticated time-inconsistent behavior."""
        x, y, d = mu
        return (x, y, 0)

    def __str__(self):
        pattern = 'Sophisticated-maxdelay-{0.max_delay}-discountconst-{0.discount_constant}-gamma-{0.gamma}-beta-{0.beta}-numiters-{0.num_iters}'
        return pattern.format(self)

class MyopicAgent(DelayDependentAgent):
    """An agent that only looks forward for a fixed horizon."""

    def __init__(self, horizon, gamma=0.9, beta=None, num_iters=50):
        """Initializes the agent, setting any relevant hyperparameters.

        horizon: Integer, the number of steps forward that the agent looks while
        planning. This must also be used as the max_delay -- if the max_delay
        was lower, the agent would no longer have a finite horizon, and if the
        max_delay was higher, we would do extra computation that is never used.
        """
        # The maximum delay should be the horizon.
        super(MyopicAgent, self).__init__(horizon, gamma, beta, num_iters)
        self.horizon = horizon

    def get_reward(self, mu, a):
        """Override to ignore rewards after the horizon."""
        x, y, d = mu
        if d >= self.horizon:
            return 0
        return super(MyopicAgent, self).get_reward(mu, a)

    def __str__(self):
        pattern = 'Myopic-horizon-{0.horizon}-gamma-{0.gamma}-beta-{0.beta}-numiters-{0.num_iters}'
        return pattern.format(self)


class UncalibratedAgent(ValueIterationLikeAgent):
    """An agent that expects the most likely action to happen more or less often
    than it actually does."""

    def __init__(self, gamma=0.9, beta=None, num_iters=50, calibration_factor=5):
        """Initializes the agent, setting any relevant hyperparameters.

        calibration_factor: The odds by which the most likely state is
        upweighted. If this is 1, then we get the optimal agent. If it is >1, we
        get an overconfident agent, that is too sure that the most likely state
        will happen. If it is <1, we get an underconfident agent.
        """
        super(UncalibratedAgent, self).__init__(gamma, beta, num_iters)
        self.calibration_factor = calibration_factor

    def get_transition_mus_and_probs(self, mu, a):
        """Gets information about possible transitions for the action.

        This is the equivalent of self.mdp.get_transition_states_and_probs() for
        generalized states. So, it returns a list of (next_mu, prob) pairs,
        where next_mu must be a generalized state.
        """
        s = self.extract_state_from_mu(mu)
        base_result = self.mdp.get_transition_states_and_probs(s, a)
        most_likely_state, _ = max(base_result, key=lambda tup: tup[1])
        dist = Distribution(dict(base_result))
        dist.factor(most_likely_state, self.calibration_factor)
        return list(dist.get_dict().items())

    def __str__(self):
        pattern = 'Uncalibrated-calibration-{0.calibration_factor}-gamma-{0.gamma}-beta-{0.beta}-numiters-{0.num_iters}'
        return pattern.format(self)
