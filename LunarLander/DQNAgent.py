from LunarLander.simulations import Rollout, Agent
import numpy as np
import os
import os.path as path
import keras
from keras.layers import InputLayer, Dense
from keras.losses import mean_squared_error
from keras.models import Sequential

class DQNAgent(Agent):
    """Trains an unbiased DQNAgent to solve LunarLander"""

    def __init__(self, env):
        super(self).__init__(env)
        # Hyper params
        self.eps = 0.1
        self.eps_decay = 0.9
        self.min_eps = 0.001
        self.gamma = 0.999
        self.batchsize = 32

        # Memory
        self.max_mem = 10000
        self.min_mem = 100
        self.memory = []

        # Create target net
        self.num_steps = 0
        self.target_step_max = 1000
        self.target_net = self.create_net()
        self.net = self.create_net()
        self.weight_dir = "./lunar/"
        self.weight_path = self.weight_dir + "dqn_weights.h5"

        # Save current state
        self.state = None

    def create_net(self):
        """Initializes model"""
        if not path.isdir(self.weight_dir):
            os.mkdir(self.weight_dir)

        roll = Rollout(self.env)
        input_shape = roll.get_obs_shape()
        output_size = roll.get_num_actions()

        activation = "relu"

        model = Sequential()
        model.add(InputLayer(input_shape=input_shape))
        model.add(Dense(40, activation=activation, kernel_initializer='he_uniform'))
        model.add(Dense(40, activation=activation, kernel_initializer='he_uniform'))
        model.add(Dense(output_size, activation=activation, kernel_initializer='he_uniform'))

        model.compile(loss=mean_squared_error, optimizer="adam")
        return model

    def act(self, roll):
        """Takes an epsilon greedy action or predicts q values to
        sample from.

        Returns reward for action taken
        """
        # epsilon greedy
        if self.state is None or np.random.random() <= self.eps:
            result = roll.act(roll.get_random_action())
        else:
            qvals = self.net.predict(self.state)
            assert len(qvals) > 1, "qvals need to be unpacked"
            action = np.argmax(qvals)
            result = roll.act(action)

        self._store(result)
        self.step()
        return result[2]

    def step(self):
        """Updates things which need to be updated after every action"""
        self.num_steps += 1
        self.decay()
        self.fix_memory()
        if self.num_steps % self.target_step_max == 0:
            self.update_target()

    def fix_memory(self):
        """Enforces self.max_mem memory size"""
        if len(self.memory) > self.max_mem:
            diff = len(self.memory) - self.max_mem
            self.memory = self.memory[diff : self.max_mem + diff]

    def update_target(self):
        """Updates target net weights by loading from model weights"""
        self.net.save_weights(self.weight_path)
        self.target_net.load_weights(self.weight_path)
        raise AttributeError("Not Yet Implemented")

    def decay(self):
        self.eps *= self.eps_decay
        if self.eps < self.min_eps:
            self.eps = self.min_eps

    @staticmethod
    def sample_from(qvals):
        """Takes random move from qvals, returns int"""
        return np.random.choice(len(qvals), 1, p=qvals)[0]

    def _store(self, result):
        """Stores result of env in replay memory, and saves
        in self.state"""
        self.memory.append(result)
        self.state = result[0]
