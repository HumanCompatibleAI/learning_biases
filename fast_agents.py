from agents import OptimalAgent
from utils import Distribution
import numpy as np

class FastOptimalAgent(OptimalAgent):
    """An agent that chooses actions for gridworlds using value iteration.

    This agent has the same interface as the OptimalAgent, but is faster than
    the OptimalAgent in agents.py, because it specializes to gridworlds and uses
    fast Numpy code instead of slow Python.

    THIS AGENT RELIES ON INTERNAL IMPLEMENTATION DETAILS OF GRIDWORLDS.
    """
    def compute_values(self):
        """Computes the values for self.mdp using value iteration.

        Populates self.values, TODO
        """
        walls, _, _ = self.mdp.convert_to_numpy_input()
        _, rewards, _ = self.reward_mdp.convert_to_numpy_input()
        height, width = len(walls), len(walls[0])
        gamma = self.gamma

        walls_without_border = walls[1:-1,1:-1]
        rewards_without_border = rewards[1:-1,1:-1]
        rewards_with_wall_without_border = rewards_without_border - 1000 * walls_without_border

        self.values = np.zeros([height, width])
        for _ in range(self.num_iters):
            # Q(s, a) = R(s, a) + gamma V(s')
            # First compute R(s, a)
            qvalues = np.stack([rewards_with_wall_without_border] * 5, axis=0)
            qvalues[:-1,:,:] += self.mdp.living_reward
            discounted_values = gamma * self.values
            qvalues[0,:,:] += discounted_values[1:-1 , :-2]
            qvalues[1,:,:] += discounted_values[1:-1 , 2:]
            qvalues[2,:,:] += discounted_values[2:   , 1:-1]
            qvalues[3,:,:] += discounted_values[:-2  , 1:-1]
            qvalues[4,:,:] += discounted_values[1:-1 , 1:-1]
            old_values = self.values
            self.values = -1000 * np.ones([height, width])
            self.values[1:-1,1:-1] = qvalues.max(axis=0)
            if self.converged(old_values, self.values):
                return

    def converged(self, values, new_values, tolerance=1e-3):
        """Returns True if value iteration has converged.

        Value iteration has converged if no value has changed by more than tolerance.

        values: The values from the previous iteration of value iteration.
        new_values: The new value computed during this iteration.
        """
        return abs(values - new_values).max() <= tolerance

    def value(self, s):
        """Computes V(s).

        s: State
        """
        x, y = s
        return self.values[y,x]

    def qvalue(self, s, a, values=None):
        """Computes Q(s, a) from the values table.

        s: State
        a: Action
        values: Numpy array of shape (height, width), mapping states to
            values. If None, then self.values is used instead.
        """
        r = self.reward_mdp.get_reward(s, a)
        transitions = self.mdp.get_transition_states_and_probs(s, a)
        return r + self.gamma * sum([p * self.value(s2) for s2, p in transitions])

    def get_action_distribution(self, s):
        """Returns a Distribution over actions."""
        actions = self.mdp.get_actions(s)
        if self.beta is not None:
            q_vals = np.array([self.qvalue(mu, a) for a in actions])
            q_vals = q_vals - np.mean(q_vals)  # To prevent overflow in exp
            action_dist = np.exp(self.beta * q_vals)
            return Distribution(dict(zip(actions, action_dist)))

        best_value, best_actions = float("-inf"), []
        for a in actions:
            action_value = self.qvalue(s, a)
            if action_value > best_value:
                best_value, best_actions = action_value, [a]
            elif action_value == best_value:
                best_actions.append(a)
        # return Distribution({a : 1 for a in best_actions})
        return Distribution({best_actions[0] : 1})

    def __str__(self):
        pattern = 'GridworldOptimal-gamma-{0.gamma}-beta-{0.beta}-numiters-{0.num_iters}'
        return pattern.format(self)

