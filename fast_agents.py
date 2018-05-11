from utils import Distribution
import agents
import numpy as np


class FastOptimalAgent(agents.OptimalAgent):
    def compute_values(self):
        """Computes the values for self.mdp using value iteration.

        Populates self.values, which is a Numpy array of size height x width.
        self.values[x,y] is the value of the state (x, y).
        """
        rewards, height, width, living_reward, gamma, noise_info, wall_info = preprocess(self)
        self.values = np.zeros([width, height])
        for _ in range(self.num_iters):
            # Q(s, a) = R(s, a) + gamma V(s')
            vals = gamma * self.values
            qvalues = get_next_state_values(vals, noise_info, wall_info)
            qvalues += rewards
            qvalues[:-1,:,:] += living_reward

            old_values = self.values
            self.values = np.zeros([width, height])
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
        k, max_delay = self.discount_constant, self.max_delay
        rewards, height, width, living_reward, gamma, noise_info, wall_info = preprocess(self, max_delay)
        stay_rewards = [rewards / (1.0 + k * d) for d in range(max_delay + 1)]
        move_rewards = [(rewards + living_reward) / (1.0 + k * d) for d in range(max_delay + 1)]
        # For states of the form (x, y, max_delay), the next state after a
        # timestep would be (x, y, max_delay). For every other state (x, y, d),
        # the next state is (x, y, d+1). To deal with this nicely, we will keep
        # a copy of the values for states (x, y, max_delay), so that for any
        # state (x, y, d), the next state will be present at index (x, y, d+1).
        # This applies to hyperbolic_rewards, values, and qvalues.
        stay_rewards.append(stay_rewards[-1])
        move_rewards.append(move_rewards[-1])
        stay_rewards = np.stack(stay_rewards, axis=-1)
        move_rewards = np.stack(move_rewards, axis=-1)
        hyperbolic_rewards = np.stack([move_rewards] * 4 + [stay_rewards], axis=0)

        self.values = np.zeros([width, height, max_delay + 2])
        for _ in range(self.num_iters):
            # Q(s, a) = R(s, a) + gamma V(s')
            vals = gamma * self.values
            qvalues = get_next_state_values(vals, noise_info, wall_info, max_delay)
            qvalues += hyperbolic_rewards

            old_values = self.values
            self.values = np.zeros([width, height, max_delay+2])
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
        k, max_delay = self.discount_constant, self.max_delay
        rewards, height, width, living_reward, gamma, delay_noise_info, delay_wall_info = preprocess(self, max_delay)
        _, _, _, _, _, noise_info, wall_info = preprocess(self)
        stay_rewards = [rewards / (1.0 + k * d) for d in range(max_delay + 1)]
        move_rewards = [(rewards + living_reward) / (1.0 + k * d) for d in range(max_delay + 1)]
        # For states of the form (x, y, max_delay), the next state after a
        # timestep would be (x, y, max_delay). For every other state (x, y, d),
        # the next state is (x, y, d+1). To deal with this nicely, we will keep
        # a copy of the values for states (x, y, max_delay), so that for any
        # state (x, y, d), the next state will be present at index (x, y, d+1).
        # This applies to hyperbolic_rewards, values, and qvalues.
        stay_rewards.append(stay_rewards[-1])
        move_rewards.append(move_rewards[-1])
        stay_rewards = np.stack(stay_rewards, axis=-1)
        move_rewards = np.stack(move_rewards, axis=-1)
        hyperbolic_rewards = np.stack([move_rewards] * 4 + [stay_rewards], axis=0)

        self.values = np.zeros([width, height, max_delay + 2])
        for _ in range(self.num_iters):
            vals = gamma * self.values

            # First compute Q-values for planning to choose actions
            # TODO(rohinmshah): Should we be using discounted_values[:,:,0]?
            planning_qvalues = get_next_state_values(vals[:,:,1], noise_info, wall_info)
            planning_qvalues += hyperbolic_rewards[:,:,:,0]
            actions_chosen = planning_qvalues.argmax(axis=0)
            onehot_actions = np.eye(5, dtype=bool)[actions_chosen]
            onehot_actions = np.transpose(onehot_actions, (2, 0, 1))
            onehot_actions = np.stack([onehot_actions] * (max_delay + 2), axis=-1)

            # Q(s, a) = R(s, a) + gamma V(s')
            qvalues = get_next_state_values(vals, delay_noise_info, delay_wall_info, max_delay)
            qvalues += hyperbolic_rewards

            # For value computation, use the onehot_actions to choose Q-values
            chosen_qvalues = np.select([onehot_actions], [qvalues], default=float("-inf")).max(axis=0)
            old_values = self.values
            self.values = np.zeros([width, height, max_delay+2])
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
        max_delay = self.max_delay
        rewards, height, width, living_reward, gamma, noise_info, wall_info = preprocess(self, max_delay)
        rewards = np.reshape(rewards, (1, width - 2, height - 2, 1))
        # For states of the form (x, y, max_delay), the next state after a
        # timestep would be (x, y, max_delay). For every other state (x, y, d),
        # the next state is (x, y, d+1). To deal with this nicely, we will keep
        # a copy of the values for states (x, y, max_delay), so that for any
        # state (x, y, d), the next state will be present at index (x, y, d+1).
        # This applies to values and qvalues.

        self.values = np.zeros([width, height, max_delay + 2])
        for _ in range(self.num_iters):
            # Q(s, a) = R(s, a) + gamma V(s')
            vals = gamma * self.values
            qvalues = get_next_state_values(vals, noise_info, wall_info, max_delay)
            qvalues[:,:,:,:-2] += rewards
            qvalues[:-1,:,:,:-2] += living_reward

            old_values = self.values
            self.values = np.zeros([width, height, max_delay + 2])
            self.values[1:-1,1:-1,:] = qvalues.max(axis=0)
            if converged(old_values, self.values):
                break

    def __str__(self):
        pattern = 'FastMyopic-horizon-{0.horizon}-gamma-{0.gamma}-beta-{0.beta}-numiters-{0.num_iters}'
        return pattern.format(self)


class FastUncalibratedAgent(agents.UncalibratedAgent):
    def compute_values(self):
        """Computes the values for self.mdp using value iteration.

        Populates self.values, which is a Numpy array of size height x width.
        self.values[x,y] is the value of the state (x, y).
        """
        rewards, height, width, living_reward, gamma, (p1, p2), wall_info = preprocess(self)
        p1 *= self.calibration_factor
        Z = p1 + 2 * p2
        noise_info = p1 / Z, p2 / Z

        self.values = np.zeros([width, height])
        for _ in range(self.num_iters):
            # Q(s, a) = R(s, a) + gamma V(s')
            vals = gamma * self.values
            qvalues = get_next_state_values(vals, noise_info, wall_info)
            qvalues += rewards
            qvalues[:-1,:,:] += living_reward

            old_values = self.values
            self.values = np.zeros([width, height])
            self.values[1:-1,1:-1] = qvalues.max(axis=0)
            if converged(old_values, self.values):
                break

    def __str__(self):
        pattern = 'FastUncalibrated-calibration-{0.calibration_factor}-gamma-{0.gamma}-beta-{0.beta}-numiters-{0.num_iters}'
        return pattern.format(self)


def converged(values, new_values, tolerance=1e-3):
    """Returns True if value iteration has converged.

    Value iteration has converged if no value has changed by more than tolerance.

    values: The values from the previous iteration of value iteration.
    new_values: The new value computed during this iteration.
    """
    return abs(values - new_values).max() <= tolerance


def preprocess_walls(walls, max_delay=None):
    is_wall, is_free = walls.astype(bool), (1 - walls).astype(bool)
    n_wall, n_free = map(shift_north, (is_wall, is_free))
    s_wall, s_free = map(shift_south, (is_wall, is_free))
    e_wall, e_free = map(shift_east, (is_wall, is_free))
    w_wall, w_free = map(shift_west, (is_wall, is_free))
    result = (n_wall, n_free, s_wall, s_free, e_wall, e_free, w_wall, w_free)
    if max_delay is not None:
        result = tuple((np.stack([x] * (max_delay+1), axis=-1) for x in result))
    return result

def preprocess(agent, max_delay=None):
    walls, _, _ = agent.mdp.convert_to_numpy_input()
    _, rewards, _ = agent.reward_mdp.convert_to_numpy_input()
    height, width = len(walls), len(walls[0])
    living_reward = agent.mdp.living_reward
    gamma, noise = agent.gamma, agent.mdp.noise
    p1, p2 = 1 - noise, noise / 2
    walls, rewards = walls.T, rewards.T
    wall_info = preprocess_walls(walls, max_delay)
    rewards = rewards[1:-1,1:-1]  # Remove border
    return rewards, height, width, living_reward, gamma, (p1, p2), wall_info


def get_next_state_values(values, noise_info, wall_info, max_delay=None):
    p1, p2 = noise_info
    n_wall, n_free, s_wall, s_free, e_wall, e_free, w_wall, w_free = wall_info
    has_delay = max_delay is not None
    stay_vals = remove_border(values, has_delay)
    n_vals = np.select([n_wall, n_free], [stay_vals, shift_north(values, has_delay)])
    s_vals = np.select([s_wall, s_free], [stay_vals, shift_south(values, has_delay)])
    e_vals = np.select([e_wall, e_free], [stay_vals, shift_east(values, has_delay)])
    w_vals = np.select([w_wall, w_free], [stay_vals, shift_west(values, has_delay)])
    n_qvals = p1 * n_vals + p2 * (e_vals + w_vals)
    s_qvals = p1 * s_vals + p2 * (e_vals + w_vals)
    e_qvals = p1 * e_vals + p2 * (n_vals + s_vals)
    w_qvals = p1 * w_vals + p2 * (n_vals + s_vals)
    if has_delay:
        # qvalues has shape (num_actions, width, height, max_delay+2)
        qvals = np.stack([n_qvals, s_qvals, e_qvals, w_qvals, stay_vals], axis=0)
        return np.concatenate((qvals, qvals[:,:,:,-1:]), axis=-1)
    else:
        return np.stack([n_qvals, s_qvals, e_qvals, w_qvals, stay_vals], axis=0)


def shift_north(arr, has_delay=False):
    return arr[1:-1, :-2, 1:] if has_delay else arr[1:-1, :-2]

def shift_south(arr, has_delay=False):
    return arr[1:-1, 2:, 1:] if has_delay else arr[1:-1, 2:]

def shift_east(arr, has_delay=False):
    return arr[2:, 1:-1, 1:] if has_delay else arr[2:, 1:-1]

def shift_west(arr, has_delay=False):
    return arr[:-2, 1:-1, 1:] if has_delay else arr[:-2, 1:-1]

def remove_border(arr, has_delay=False):
    return arr[1:-1 , 1:-1, 1:] if has_delay else arr[1:-1 , 1:-1]
