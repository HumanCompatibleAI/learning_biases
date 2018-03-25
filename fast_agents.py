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
            if converged(old_values, self.values):
                break

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
        living_reward = self.mdp.living_reward
        walls, rewards = walls.T, rewards.T

        walls, rewards = walls[1:-1,1:-1], rewards[1:-1,1:-1]  # Remove border
        rewards = rewards - 1000 * walls
        rewards_for_stay = [rewards / (1.0 + k * d) for d in range(max_delay + 1)]
        rewards_for_moves = [(rewards + living_reward) / (1.0 + k * d) for d in range(max_delay + 1)]
        # For states of the form (x, y, max_delay), the next state after a
        # timestep would be (x, y, max_delay). For every other state (x, y, d),
        # the next state is (x, y, d+1). To deal with this nicely, we will keep
        # a copy of the values for states (x, y, max_delay), so that for any
        # state (x, y, d), the next state will be present at index (x, y, d+1).
        # This applies to hyperbolic_rewards, values, and qvalues.
        rewards_for_stay.append(rewards_for_stay[-1])
        rewards_for_moves.append(rewards_for_moves[-1])
        rewards_for_stay = np.stack(rewards_for_stay, axis=-1)
        rewards_for_moves = np.stack(rewards_for_moves, axis=-1)
        hyperbolic_rewards = np.stack([rewards_for_moves] * 4 + [rewards_for_stay], axis=0)

        self.values = np.zeros([width, height, max_delay + 2])
        for _ in range(self.num_iters):
            # Q(s, a) = R(s, a) + gamma V(s')
            qvalues = hyperbolic_rewards.copy()
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
            if converged(old_values, self.values):
                break

        # Remove the extra copy of the values
        self.values = self.values[:,:,:-1]

    def __str__(self):
        pattern = 'FastNaive-maxdelay-{0.max_delay}-discountconst-{0.discount_constant}-gamma-{0.gamma}-beta-{0.beta}-numiters-{0.num_iters}'
        return pattern.format(self)


class FastSophisticatedTimeDiscountingAgent(agents.SophisticatedTimeDiscountingAgent):
    def compute_values(self):
        """Computes the values for self.mdp using value iteration.

        Populates self.values, which is a Numpy array of size height x width.
        self.values[x,y] is the value of the state (x, y).
        """
        walls, _, _ = self.mdp.convert_to_numpy_input()
        _, rewards, _ = self.reward_mdp.convert_to_numpy_input()
        height, width, max_delay = len(walls), len(walls[0]), self.max_delay
        gamma, k = self.gamma, self.discount_constant
        living_reward = self.mdp.living_reward
        walls, rewards = walls.T, rewards.T

        walls, rewards = walls[1:-1,1:-1], rewards[1:-1,1:-1]  # Remove border
        rewards = rewards - 1000 * walls
        rewards_for_stay = [rewards / (1.0 + k * d) for d in range(max_delay + 1)]
        rewards_for_moves = [(rewards + living_reward) / (1.0 + k * d) for d in range(max_delay + 1)]
        # For states of the form (x, y, max_delay), the next state after a
        # timestep would be (x, y, max_delay). For every other state (x, y, d),
        # the next state is (x, y, d+1). To deal with this nicely, we will keep
        # a copy of the values for states (x, y, max_delay), so that for any
        # state (x, y, d), the next state will be present at index (x, y, d+1).
        # This applies to hyperbolic_rewards, values, and qvalues.
        rewards_for_stay.append(rewards_for_stay[-1])
        rewards_for_moves.append(rewards_for_moves[-1])
        rewards_for_stay = np.stack(rewards_for_stay, axis=-1)
        rewards_for_moves = np.stack(rewards_for_moves, axis=-1)
        hyperbolic_rewards = np.stack([rewards_for_moves] * 4 + [rewards_for_stay], axis=0)

        self.values = np.zeros([width, height, max_delay + 2])
        for _ in range(self.num_iters):
            # First compute Q-values for planning to choose actions
            planning_qvalues = hyperbolic_rewards[:,:,:,0].copy()
            discounted_values = gamma * self.values
            # TODO(rohinmshah): We should be using discounted_values[foo,bar,0],
            # but the agent in agents.py doesn't do this. Fix.
            planning_qvalues[0,:,:] += discounted_values[1:-1 , :-2 , 1]
            planning_qvalues[1,:,:] += discounted_values[1:-1 , 2:  , 1]
            planning_qvalues[2,:,:] += discounted_values[2:   , 1:-1, 1]
            planning_qvalues[3,:,:] += discounted_values[:-2  , 1:-1, 1]
            planning_qvalues[4,:,:] += discounted_values[1:-1 , 1:-1, 1]
            actions_chosen = planning_qvalues.argmax(axis=0)
            onehot_actions = np.eye(5, dtype=bool)[actions_chosen]
            onehot_actions = np.transpose(onehot_actions, (2, 0, 1))
            onehot_actions = np.stack([onehot_actions] * (max_delay + 2), axis=-1)

            # Q(s, a) = R(s, a) + gamma V(s')
            qvalues = hyperbolic_rewards.copy()
            qvalues[0,:,:,:-1] += discounted_values[1:-1 , :-2 , 1:]
            qvalues[1,:,:,:-1] += discounted_values[1:-1 , 2:  , 1:]
            qvalues[2,:,:,:-1] += discounted_values[2:   , 1:-1, 1:]
            qvalues[3,:,:,:-1] += discounted_values[:-2  , 1:-1, 1:]
            qvalues[4,:,:,:-1] += discounted_values[1:-1 , 1:-1, 1:]
            qvalues[:,:,:,-1] = qvalues[:,:,:,-2]

            # For value computation, use the onehot_actions to choose Q-values
            chosen_qvalues = np.select([onehot_actions], [qvalues], default=float("-inf")).max(axis=0)
            old_values = self.values
            self.values = -1000 * np.ones([width, height, max_delay+2])
            self.values[1:-1,1:-1,:] = chosen_qvalues
            if converged(old_values, self.values):
                break

        # Remove the extra copy of the values
        self.values = self.values[:,:,:-1]

    def __str__(self):
        pattern = 'FastSophisticated-maxdelay-{0.max_delay}-discountconst-{0.discount_constant}-gamma-{0.gamma}-beta-{0.beta}-numiters-{0.num_iters}'
        return pattern.format(self)


class FastMyopicAgent(agents.MyopicAgent):
    def compute_values(self):
        """Computes the values for self.mdp using value iteration.

        Populates self.values, which is a Numpy array of size height x width.
        self.values[x,y] is the value of the state (x, y).
        """
        walls, _, _ = self.mdp.convert_to_numpy_input()
        _, rewards, _ = self.reward_mdp.convert_to_numpy_input()
        height, width, max_delay = len(walls), len(walls[0]), self.max_delay
        gamma = self.gamma
        walls, rewards = walls.T, rewards.T

        walls, rewards = walls[1:-1,1:-1], rewards[1:-1,1:-1]  # Remove border
        rewards = rewards - 1000 * walls
        # For states of the form (x, y, max_delay), the next state after a
        # timestep would be (x, y, max_delay). For every other state (x, y, d),
        # the next state is (x, y, d+1). To deal with this nicely, we will keep
        # a copy of the values for states (x, y, max_delay), so that for any
        # state (x, y, d), the next state will be present at index (x, y, d+1).
        # This applies to rewards, values, and qvalues.

        self.values = np.zeros([width, height, max_delay + 1])
        for _ in range(self.num_iters):
            # Q(s, a) = R(s, a) + gamma V(s')
            # First compute R(s, a)
            qvalues = np.stack([rewards] * 5, axis=0)
            qvalues[:-1,:,:] += self.mdp.living_reward
            horizon_qs = np.zeros(qvalues.shape)
            qvalues = np.stack([qvalues] * max_delay + [horizon_qs], axis=-1)
            discounted_values = gamma * self.values
            qvalues[0,:,:,:-1] += discounted_values[1:-1 , :-2,  1:]
            qvalues[1,:,:,:-1] += discounted_values[1:-1 , 2:,   1:]
            qvalues[2,:,:,:-1] += discounted_values[2:   , 1:-1, 1:]
            qvalues[3,:,:,:-1] += discounted_values[:-2  , 1:-1, 1:]
            qvalues[4,:,:,:-1] += discounted_values[1:-1 , 1:-1, 1:]
            old_values = self.values
            self.values = -1000 * np.ones([width, height, max_delay + 1])
            self.values[1:-1,1:-1,:] = qvalues.max(axis=0)
            if converged(old_values, self.values):
                break

    def __str__(self):
        pattern = 'FastMyopic-horizon-{0.horizon}-gamma-{0.gamma}-beta-{0.beta}-numiters-{0.num_iters}'
        return pattern.format(self)


def converged(values, new_values, tolerance=1e-3):
    """Returns True if value iteration has converged.

    Value iteration has converged if no value has changed by more than tolerance.

    values: The values from the previous iteration of value iteration.
    new_values: The new value computed during this iteration.
    """
    return abs(values - new_values).max() <= tolerance
