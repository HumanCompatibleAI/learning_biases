import gym

""" Task: Learn "good"* rewards from agent's trajectory that can be used
to plan perfectly by an optimal agent.

Setup: Observe a biased agent
Training: Train a NN to replicate biased agent behavior based on random rewards
Deliverable: Backprop through NN to generate true reward

Note: `biased` means the agent has a bias in its actions that prevents it from
having full control over its actions, (or perhaps inability to understand thrust controls)

Tip: It'd probably be advantageous to transfer learn weights from an NN that can solve this problem for simple
synthetic rollouts of agents with a simple known bias and simple known reward.

***CURRENT***
4/19/18
Setting up biased agents
    Finding known algorithm to solve Continuous Control tasks
    Biasing it
    Adding code to capture its trajectory

Setting up code to capture rollouts

Goal: Train DQN to solve this, then shift all its observations to left or side by varying amount
"""

class Rollout:
    """Keeps track of information returned by a gym environment.
    env(Environment): gym environment we are interacting with
    trajectory(list): list of (prev_obs, action, obs, reward) tuples
                            a.k.a. <s, a, s', r> tuples
    reward(float): scalar value of summed reward trajectory"""

    def __init__(self, env):
        self.env = env
        self.env.reset()
        self.trajectory = []
        self.reward = 0
        self.done = False
        self.prev_obs = None

    def act(self, action):
        """Accepts (int) action to perform on an environment. Returns the result of `env.step(action)`"""
        if not self.done:
            obs, reward, done, info = self.env.step(action)
            self.reward += reward
            self.done = done
            if self.prev_obs is None:
                self.prev_obs = obs
            results = (self.prev_obs, action, obs, reward)
            self.trajectory.append(results)
        else:
            print("Cannot act when the environment is done.")
            return None
        return results

    def get_num_actions(self):
        """Returns the number of possible actions to take in this space"""
        return self.env.action_space.n

    def get_obs_shape(self):
        """Returns size of input space"""
        return self.env.observation_space.shape

    def get_random_action(self):
        """Returns a random action using the `env.action_space.sample()` function"""
        return self.env.action_space.sample()

    def is_done(self):
        return self.done

class Agent:
    """Takes random moves in environment"""

    def __init__(self, env):
        self.env = env
        self.rolls = []
        self.total_reward = 0

    def rollout(self):
        roll = Rollout(self.env)
        while not roll.is_done():
            self.act(roll)
        # print("Received {} reward".format(roll.reward))
        self.total_reward += roll.reward
        self.rolls.append(roll)

    def act(self, roll):
        return roll.act(self.get_action(roll))

    def get_action(self, roll):
        return roll.get_random_action()

    def get_avg_reward(self):
        return self.total_reward / len(self.rolls)


if __name__ == '__main__':
    from time import time
    env = gym.make('LunarLander-v2')

    agent = Agent(env)
    num_rolls = 1000
    start = time()
    agent.rollout()
    print(agent.rolls[0].get_obs_shape())
    for _ in range(num_rolls):
        agent.rollout()
    print("{} rolls took {} seconds.".format(num_rolls, time() - start))



