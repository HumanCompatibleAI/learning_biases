from agent_interface import Agent
from collections import defaultdict
import random

class ValueIterationLikeAgent(Agent):
    def __init__(self, gamma=1.0, beta=None, num_iters=100):
        super(ValueIterationLikeAgent, self).__init__(gamma)
        self.beta = beta
        self.num_iters = num_iters

    def set_mdp(self, mdp):
        super(ValueIterationLikeAgent, self).set_mdp(mdp)
        self.compute_values()

    def compute_values(self):
        values = defaultdict(float)
        for _ in range(self.num_iters):
            new_values = defaultdict(float)
            for mu in self.get_mus():
                actions = self.get_actions(mu)
                if not actions:
                    continue
                new_mu = self.get_mu_for_planning(mu)
                qvalues = [(self.qvalue(new_mu, a, values), a) for a in actions]
                _, chosen_action = max(qvalues)
                new_values[mu] = self.qvalue(mu, chosen_action, values)
            values = new_values
        self.values = values

    def qvalue(self, mu, a, values=None):
        if values is None:
            values = self.values
        r = self.get_reward(mu, a)
        transitions = self.get_transition_mus_and_probs(mu, a)
        return r + self.gamma * sum([p * values[mu2] for mu2, p in transitions])

    def get_action(self, s):
        # TODO(rohinmshah): Take beta into account
        if self.beta is not None:
            print(self.beta)
            raise NotImplementedError("Noisy action choice not implemented!")
        mu = self.extend_state_to_mu(s)
        best_value, best_actions = float("-inf"), []
        for a in self.mdp.get_actions(s):
            action_value = self.qvalue(mu, a)
            if action_value > best_value:
                best_value, best_actions = action_value, [a]
            elif action_value == best_value:
                best_actions.append(a)
        return random.choice(best_actions)

    def get_mus(self):
        return self.mdp.get_states()

    def get_actions(self, mu):
        s = self.extract_state_from_mu(mu)
        return self.mdp.get_actions(s)

    def get_reward(self, mu, a):
        s = self.extract_state_from_mu(mu)
        return self.mdp.get_reward(s, a)

    def get_transition_mus_and_probs(self, mu, a):
        s = self.extract_state_from_mu(mu)
        return self.mdp.get_transition_states_and_probs(s, a)

    def get_mu_for_planning(self, mu):
        return mu

    def extend_state_to_mu(self, state):
        return state

    def extract_state_from_mu(self, mu):
        return mu

class OptimalAgent(ValueIterationLikeAgent):
    pass

class DelayDependentAgent(ValueIterationLikeAgent):
    def __init__(self, max_delay, gamma=1.0, beta=None, num_iters=100):
        super(DelayDependentAgent, self).__init__(gamma, beta, num_iters)
        self.max_delay = max_delay

    def get_mus(self):
        states = self.mdp.get_states()
        return [(s, d) for s in states for d in range(self.max_delay + 1)]

    def get_transition_mus_and_probs(self, mu, a):
        s, d = mu
        transitions = self.mdp.get_transition_states_and_probs(s, a)
        newd = min(d + 1, self.max_delay)
        return [((s2, newd), p) for s2, p in transitions]

    def extend_state_to_mu(self, state):
        return (state, 0)

    def extract_state_from_mu(self, mu):
        return mu[0]

class TimeDiscountingAgent(DelayDependentAgent):
    def __init__(self, max_delay, discount_constant,
                 gamma=1.0, beta=None, num_iters=100):
        super(TimeDiscountingAgent, self).__init__(
            max_delay, gamma, beta, num_iters)
        self.discount_constant = discount_constant

    def get_reward(self, mu, a):
        s, d = mu
        discount = (1.0 / (1.0 + self.discount_constant * d))
        return discount * self.mdp.get_reward(s, a)

class NaiveTimeDiscountingAgent(TimeDiscountingAgent):
    pass

class SophisticatedTimeDiscountingAgent(TimeDiscountingAgent):
    def get_mu_for_planning(self, mu):
        s, d = mu
        return (s, 0)

class MyopicAgent(DelayDependentAgent):
    def __init__(self, horizon, gamma=1.0, beta=None, num_iters=100):
        # The maximum delay should be the horizon.
        super(MyopicAgent, self).__init__(horizon, gamma, beta, num_iters)
        self.horizon = horizon

    def get_reward(self, mu, a):
        s, d = mu
        if d >= self.horizon:
            return 0
        return super(MyopicAgent, self).get_reward(mu, a)
