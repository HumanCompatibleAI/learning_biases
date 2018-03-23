from utils import Distribution
import agents
import numpy as np


class FastOptimalAgent(agents.OptimalAgent):
    def compute_values(self):
        """Computes the values for self.mdp using value iteration.

        Populates self.values, which is a Numpy array of size height x width.
        self.values[x,y] is the value of the state (x, y).
        """
        walls, _, _ = self.mdp.convert_to_numpy_input()
        _, rewards, _ = self.reward_mdp.convert_to_numpy_input()
        height, width = len(walls), len(walls[0])
        gamma = self.gamma
        walls, rewards = walls.T, rewards.T

        walls, rewards = walls[1:-1,1:-1], rewards[1:-1,1:-1]  # Remove border
        rewards = rewards - 1000 * walls

        self.values = np.zeros([width, height])
        for _ in range(self.num_iters):
            # Q(s, a) = R(s, a) + gamma V(s')
            # First compute R(s, a)
            qvalues = np.stack([rewards] * 5, axis=0)
            qvalues[:-1,:,:] += self.mdp.living_reward
            discounted_values = gamma * self.values
            qvalues[0,:,:] += discounted_values[1:-1 , :-2]
            qvalues[1,:,:] += discounted_values[1:-1 , 2:]
            qvalues[2,:,:] += discounted_values[2:   , 1:-1]
            qvalues[3,:,:] += discounted_values[:-2  , 1:-1]
            qvalues[4,:,:] += discounted_values[1:-1 , 1:-1]
            old_values = self.values
            self.values = -1000 * np.ones([width, height])
            self.values[1:-1,1:-1] = qvalues.max(axis=0)
            if self.converged(old_values, self.values):
                break

    def converged(self, values, new_values, tolerance=1e-3):
        """Returns True if value iteration has converged.

        Value iteration has converged if no value has changed by more than tolerance.

        values: The values from the previous iteration of value iteration.
        new_values: The new value computed during this iteration.
        """
        return abs(values - new_values).max() <= tolerance

    def __str__(self):
        pattern = 'FastOptimal-gamma-{0.gamma}-beta-{0.beta}-numiters-{0.num_iters}'
        return pattern.format(self)



class FastNaiveTimeDiscountingAgent(agents.NaiveTimeDiscountingAgent):
    def compute_values(self):
        """Computes the values for self.mdp using value iteration.

        Populates self.values, which is a Numpy array of size height x width.
        self.values[x,y] is the value of the state (x, y).
        """
        walls, _, _ = self.mdp.convert_to_numpy_input()
        _, rewards, _ = self.reward_mdp.convert_to_numpy_input()
        height, width, max_delay = len(walls), len(walls[0]), self.max_delay
        gamma, k = self.gamma, self.discount_constant
        walls, rewards = walls.T, rewards.T

        walls, rewards = walls[1:-1,1:-1], rewards[1:-1,1:-1]  # Remove border
        rewards = rewards - 1000 * walls
        rewards = [rewards / (1.0 + k * d) for d in range(max_delay + 1)]
        # For states of the form (x, y, max_delay), the next state after a
        # timestep would be (x, y, max_delay). For every other state (x, y, d),
        # the next state is (x, y, d+1). To deal with this nicely, we will keep
        # a copy of the values for states (x, y, max_delay), so that for any
        # state (x, y, d), the next state will be present at index (x, y, d+1).
        # This applies to hyperbolic_rewards, values, and qvalues.
        rewards.append(rewards[-1])
        rewards = np.stack(rewards, axis=-1)

        self.values = np.zeros([width, height, max_delay + 2])
        for _ in range(self.num_iters):
            # Q(s, a) = R(s, a) + gamma V(s')
            # First compute R(s, a)
            qvalues = np.stack([rewards] * 5, axis=0)
            qvalues[:-1,:,:,:] += self.mdp.living_reward
            discounted_values = gamma * self.values
            qvalues[0,:,:,:-1] += discounted_values[1:-1 , :-2 , 1:]
            qvalues[1,:,:,:-1] += discounted_values[1:-1 , 2:  , 1:]
            qvalues[2,:,:,:-1] += discounted_values[2:   , 1:-1, 1:]
            qvalues[3,:,:,:-1] += discounted_values[:-2  , 1:-1, 1:]
            qvalues[4,:,:,:-1] += discounted_values[1:-1 , 1:-1, 1:]
            qvalues[:,:,:,-1] = qvalues[:,:,:,-2]
            old_values = self.values
            self.values = -1000 * np.ones([width, height, max_delay+2])
            self.values[1:-1,1:-1,:] = qvalues.max(axis=0)
            if self.converged(old_values, self.values):
                break

        # Remove the extra copy of the values
        self.values = self.values[:,:,:-1]

    def converged(self, values, new_values, tolerance=1e-3):
        """Returns True if value iteration has converged.

        Value iteration has converged if no value has changed by more than tolerance.

        values: The values from the previous iteration of value iteration.
        new_values: The new value computed during this iteration.
        """
        return abs(values - new_values).max() <= tolerance

    def __str__(self):
        pattern = 'FastNaive-maxdelay-{0.max_delay}-discountconst-{0.discount_constant}-gamma-{0.gamma}-beta-{0.beta}-numiters-{0.num_iters}'
        return pattern.format(self)
