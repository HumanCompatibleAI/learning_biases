# Some code taken from https://github.com/TheAbhiKumar/tensorflow-value-iteration-networks

import time
import numpy as np
import random
import tensorflow as tf

import agents
import fast_agents
from gridworld_data import generate_data_for_planner, generate_data_for_reward, create_agents_from_config
from model import create_model, calculate_action_distribution
from utils import fmt_row, init_flags, plot_reward, set_seeds
from agent_runner import evaluate_proxy
import sys

class PlannerArchitecture(object):
    """Stores all of the tensors involved in the architecture.

    Includes the MDP description, the reward function, the model mapping these
    to actions, etc. Also includes methods for various kinds of training that
    can be applied to this architecture.
    """

    def __init__(self, config):
        """Create model using config flags and also create model saver.

        Note: this function does initiate variables (necessary for SavedModel api)
        """
        # Tensorflow refuses to have a Variable whose shape is not fully
        # determined. As a result, we must set the batch size to a constant
        # which cannot be changed during a particular run. (We need to use a
        # Variable for the reward so that the reward is trainable.)
        self.config = config
        batch_size = config.batchsize
        imsize = config.imsize
        num_actions = config.num_actions

        self.image = tf.placeholder(
            tf.float32, name="image", shape=[batch_size, imsize, imsize])
        self.reward_input = tf.placeholder(
            tf.float32, name="reward_input", shape=[batch_size, imsize, imsize])
        self.y  = tf.placeholder(
            tf.float32, name="y",  shape=[batch_size, imsize, imsize, num_actions])

        self.reward = tf.Variable(
            tf.zeros([batch_size, imsize, imsize]), name='reward', trainable=False)
        self.assign_reward = self.reward.assign(self.reward_input)

        if config.verbosity >= 1:
            print('Creating model: ' + config.model)
        self.model = create_model(self.image, self.reward, config)

        # Add tensors to calculate action distributions
        self.pred_dist = calculate_action_distribution(
            self.model.logits, config.batchsize, config.ch_q, name='pred_action_dist')
        self.y_dist = calculate_action_distribution(
            self.y, config.batchsize, config.ch_q, name='true_action_dist')

        # Reshape for losses
        logits = tf.reshape(self.model.logits, [-1, num_actions])
        labels = tf.reshape(self.y, [-1, num_actions])

        # Define losses
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            logits=logits, labels=labels, name='cross_entropy')
        cross_entropy_mean = tf.reduce_mean(
            cross_entropy, name='cross_entropy_mean')
        tf.add_to_collection('losses', cross_entropy_mean)

        logits_cost = tf.add_n(tf.get_collection('losses'), name='logits_loss')
    
        if config.model == 'VIN' and config.vin_regularizer_C > 0:
            # TODO(rohinmshah): This assumes that no regularization has been
            # added besides to the VIN -- very errorprone, fix. Also add
            # regularization losses to the Model instead of computing them
            # here.
            vin_regularizer_cost = tf.add_n(
                tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES),
                name='vin_loss')
            # TODO(rohinmshah): Rename step1_cost and step2_cost
            self.step1_cost = logits_cost + vin_regularizer_cost
        else:
            self.step1_cost = logits_cost

        if config.reward_regularizer_C > 0:
            l1_regularizer = tf.contrib.layers.l1_regularizer(config.reward_regularizer_C)
            reward_regularizer_cost = tf.contrib.layers.apply_regularization(
                l1_regularizer, [self.reward])
            self.step2_cost = logits_cost + reward_regularizer_cost
        else:
            self.step2_cost = logits_cost

        # Define optimizers
        if config.model != 'VI':
            planner_optimizer = tf.train.RMSPropOptimizer(
                learning_rate=config.lr, epsilon=1e-6, centered=True)
            self.planner_optimize_op = planner_optimizer.minimize(self.step1_cost)

        reward_optimizer = tf.train.RMSPropOptimizer(
            learning_rate=config.reward_lr, epsilon=1e-6, centered=True)
        self.reward_optimize_op = reward_optimizer.minimize(self.step2_cost, var_list=[self.reward])

        # Test model & calculate accuracy
        cp = tf.cast(tf.argmax(self.model.output_probs, 1), tf.int32)

        # Use the most probable action even for the gold labels
        most_likely_labels = tf.cast(tf.argmax(labels, axis=1), tf.int32)
        self.err = tf.reduce_mean(
            tf.cast(tf.not_equal(cp, most_likely_labels), dtype=tf.float32))

        # Initializing the variables
        self.initialize_op = tf.global_variables_initializer()

        # Saving model in SavedModel format
        # if config.log:
        #     self.builder = tf.saved_model.builder.SavedModelBuilder(
        #         config.logdir+'model/')
        # If process does not finish
        self.final_accuracy = "NA"

    def register_new_session(self, sess):
        # The tag on this model is to access the weights explicitly
        # I think SERVING vs TRAINING tags means you can save static & dynamic weights 4 a model
        sess.run(self.initialize_op)
        # if self.config.log:
        #     self.builder.add_meta_graph_and_variables(
        #         sess, [tf.saved_model.tag_constants.SERVING])

    def run_epoch(self, sess, data, ops_to_run, ops_to_average, distributions=[]):
        batch_size = self.config.batchsize
        imsize = self.config.imsize
        num_actions = self.config.num_actions
        tstart = time.time()
        image_data, reward_data, y_data = data
        averages = [0.0] * len(ops_to_average)
        num_batches = int(image_data.shape[0] / batch_size)
        avg_dists = [np.zeros(num_actions) for dist in distributions]

        # Loop over all batches
        for i in range(num_batches):
            start, end = i * batch_size, (i + 1) * batch_size
            fd = {
                "image:0": image_data[start:end],
                "reward:0": reward_data[start:end],
                "y:0": y_data[start:end]
            }
            results = sess.run(ops_to_run + ops_to_average+distributions, feed_dict=fd)
            num_ops_to_run = len(ops_to_run)
            num_ops_to_avg = len(ops_to_average)
            op_results = results[:num_ops_to_run]
            average_op_results = results[num_ops_to_run:num_ops_to_run+num_ops_to_avg]
            averages = [x + y for x, y in zip(averages, average_op_results)]

            dists = results[num_ops_to_run+num_ops_to_avg:]
            avg_dists = [avg + np.reshape(batch_dist,(-1,)) for avg, batch_dist in zip(avg_dists, dists)]
            
        averages = [x / num_batches for x in averages]
        elapsed = time.time() - tstart
        return op_results, averages, elapsed, avg_dists

    def train_planner(self, sess, train_data, validation_data, num_epochs):
        """Trains the planner module given MDPs with reward functions and the
        corresponding policies.

        Validation can be turned off by explicitly passing None.
        """
        if self.config.verbosity >= 3:
            print(fmt_row(10, ["Epoch", "Train Cost", "Train Err", "Valid Err", "Epoch Time"]))

        num_actions = self.config.num_actions
        action_dists = [np.zeros(num_actions), np.zeros(num_actions)]
        for epoch in range(int(num_epochs)):
            _, (avg_cost, avg_err), elapsed, epoch_dist = self.run_epoch(
                sess, train_data, [self.planner_optimize_op],
                [self.step1_cost, self.err], [self.pred_dist, self.y_dist])

            # Display logs per epoch step
            if self.config.verbosity >= 3 and epoch % self.config.display_step == 0:
                if validation_data is not None:
                    _, (validation_err,), _, _ = self.run_epoch(sess, validation_data, [], [self.err])
                else:
                    validation_err = 'N/A'
                print(fmt_row(10, [epoch, avg_cost, avg_err, validation_err, elapsed]))
            elif self.config.verbosity >= 2 and epoch % self.config.display_step == 0:
                print('Epoch {} of {}'.format(epoch, num_epochs))

            if self.config.log:
                summary = tf.Summary()
                # summary.ParseFromString(sess.run(summary_op))
                # summary.value.add(tag='Average error', simple_value=float(avg_err))
                # summary.value.add(tag='Average cost', simple_value=float(avg_cost))
                # summary_writer.add_summary(summary, epoch)

        action_dists = [d + b_d for d, b_d in zip(action_dists, epoch_dist)]
        action_dists = [d / (np.sum(d)) for d in action_dists]
        if self.config.verbosity >= 3 and action_dists:
            print("Action Distribution Comparison")
            print("------------------------------")
            print(fmt_row(10, ["Predicted"] + action_dists[0].tolist()))
            print(fmt_row(10, ["Actual"]+ action_dists[1].tolist()))

        if self.config.verbosity >= 1 and validation_data is not None:
            _, (validation_err,), _, _ = self.run_epoch(sess, validation_data, [], [self.err])
            print('Final Accuracy: ' + str(100 * (1 - validation_err)))
            self.final_accuracy = 100 * (1 - validation_err)
            print('Validation Accuracy: ' + str(100 * (1 - validation_err)))

        # Saving SavedModel instance
        # if self.config.log:
        #     savepath = self.builder.save()
        #     print("Model saved to: {}".format(savepath))

    def train_reward(self, sess, image_data, reward_data, y_data, num_epochs, print_output=True):
        """Infers the reward using backprop, holding the planner fixed.

        Due to Tensorflow constraints, image_data must contain exactly
        batch_size number of MDPs on which the reward should be inferred.

        The rewards are initialized to the values in reward_data. If reward_data
        is None, the rewards are initialized to all zeroes.
        """
        if self.config.verbosity >= 3:
            print(fmt_row(10, ["Iteration", "Train Cost", "Train Err", "Iter Time"]))
        if reward_data is None:
            reward_data = np.random.randn(image_data.shape)

        batch_size = self.config.batchsize
        num_batches = int(image_data.shape[0] / batch_size)
        for batch_num in range(num_batches):
            if self.config.verbosity >= 2 and batch_num % 10 == 0:
                print('Batch {} of {}'.format(batch_num, num_batches))
            start, end = batch_num * batch_size, (batch_num + 1) * batch_size
            # We can't feed in reward_data directly to self.reward, because then
            # it will treat it as a constant and will not be able to update it
            # with backprop. Instead, we first run an op that assigns the
            # reward, and only then do the backprop.
            fd = {
                "reward_input:0": reward_data[start:end],
            }
            sess.run([self.assign_reward.op], feed_dict=fd)

            for epoch in range(num_epochs):
                tstart = time.time()
                fd = {
                    "image:0": image_data[start:end],
                    "y:0": y_data[start:end]
                }
                _, e_, c_ = sess.run(
                    [self.reward_optimize_op, self.err, self.step2_cost],
                    feed_dict=fd)
                elapsed = time.time() - tstart
                if self.config.verbosity >= 3 and batch_num % 10 == 0:
                    print(fmt_row(10, [epoch, c_, e_, elapsed]))

            reward_data[start:end] = self.reward.eval()

        return reward_data

    def train_joint(self, sess, image_data, reward_data, y_data, num_epochs, print_output=True):
        """Trains the planner module given MDPs with reward functions and the
        corresponding policies.

        Essentially performs train_reward, but also passes planner graph

        Validation can be turned off by explicitly passing None.
        """
        if print_output:
            print(fmt_row(10, ["Iteration", "Train Cost", "Train Err", "Iter Time"]))
        if reward_data is None:
            # Initialize the reward array to random values
            reward_data = np.random.randn(*image_data.shape)

        planner_ops = [self.planner_optimize_op, self.step1_cost]
        batch_size = self.config.batchsize
        num_batches = int(image_data.shape[0] / batch_size)
        for epoch in range(num_epochs):
            # Run an epoch on each batch of MDPs to infer their rewards
            for batch_num in range(num_batches):
                start, end = batch_num * batch_size, (batch_num + 1) * batch_size
                # We can't feed in reward_data directly to self.reward, because then
                # it will treat it as a constant and will not be able to update it
                # with backprop. Instead, we first run an op that assigns the
                # reward, and only then do the backprop.
                fd = {
                    "reward_input:0": reward_data[start:end],
                }
                sess.run([self.assign_reward.op], feed_dict=fd)

                tstart = time.time()
                fd = {
                    "image:0": image_data[start:end],
                    "y:0": y_data[start:end]
                }

                # Run both step1 & step2 ops, report only error & step2 cost
                _, e_, c_, _etc, __etc = sess.run(
                    [self.reward_optimize_op, self.err, self.step2_cost] + planner_ops,
                    feed_dict=fd)

                self.final_accuracy = (1-e_)*100
                elapsed = time.time() - tstart
                if print_output and batch_num % 10 == 0:
                    print(fmt_row(10, [epoch, c_, e_, elapsed]))

                reward_data[start:end] = self.reward.eval()

        return reward_data

def run_interruptibly(fn, step_name='this step'):
    """Runs fn in a mode where KeyboardInterrupts will interrupt fn but will
    then continue with the rest of the program execution.
    """
    try:
        return fn()
    except KeyboardInterrupt:
        print('Skipping the rest of ' + step_name)

def run_inference(planner_train_data, planner_validation_data, reward_data,
                  algorithm_fn, config):
    """
    Evaluates a given algorithm_fn based upon its inferred reward

    Specifically, regret is calculated with respect to the reward_data

    :return: (final_accuracy, % reward = optimal planner)
    """
    # seed random number generators
    seed = config.seeds.pop(0)
    set_seeds(seed)
    # use flags to create model and retrieve relevant operations
    architecture = PlannerArchitecture(config)

    if planner_train_data and planner_validation_data:
        image_train, reward_train, _, y_train = planner_train_data
        image_validation, reward_validation, _, y_validation = planner_validation_data
        train_data = (image_train, reward_train, y_train)
        validation_data = (image_validation, reward_validation, y_validation)
    else:
        # This is for algorithms which do not need data to infer
        # Like vi_algorithm
        train_data = None
        validation_data = None

    image_irl, reward_irl, start_states_irl, y_irl = reward_data
    reward_data = (image_irl, y_irl)

    # Launch the graph
    gpu_config = None
    if config.use_gpu:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
        gpu_config = tf.ConfigProto(gpu_options=gpu_options)

    with tf.Session(config=gpu_config) as sess:
        if config.log:
            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(config.logdir, sess.graph)

        architecture.register_new_session(sess)

        inferred_rewards = algorithm_fn(
            architecture, sess, train_data, validation_data, reward_data, config)

        if config.verbosity >= 3:
            print('The first reward should be:')
            print(reward_irl[0])
            # normalized_inferred_reward = inferred_reward / inferred_reward.max()
            print('The inferred reward is:')
            print(inferred_rewards[0])

        reward_percents = []
        for label, reward, wall, start_state, i in zip(reward_irl, inferred_rewards, image_irl, start_states_irl, range(len(reward_irl))):
            if config.plot_rewards and i < 10:
                plot_reward(label, reward, wall, 'reward_pics/reward_{}'.format(i))
            reward_percents.append(
                evaluate_proxy(wall, start_state, reward, label, episode_length=20))

        average_percent_reward = float(sum(reward_percents)) / len(reward_percents)
        if config.verbosity >= 1:
            print(reward_percents[:10])
            print('On average planning with the inferred rewards is '
                  + str(100 * average_percent_reward)
                  + '% as good as planning with the true rewards')

        return architecture.final_accuracy, average_percent_reward

def two_phase_algorithm(architecture, sess, train_data, validation_data,
                        reward_data, config):
    config.em_iterations = 0
    return iterative_algorithm(
        architecture, sess, train_data, validation_data, reward_data, config)

def iterative_algorithm(architecture, sess, train_data, validation_data,
                        reward_data, config):
    """Iterative EM-like algorithm.

    The train and validation data are used to initialize the planner, but fine
    tuning of both the reward and the planner are done with reward_data.
    """
    image_irl, y_irl = reward_data
    run_interruptibly(
        lambda: architecture.train_planner(
            sess, train_data, validation_data, config.epochs),
        'planner training')

    rewards = architecture.train_reward(
        sess, image_irl, None, y_irl, config.reward_epochs)

    for _ in range(config.em_iterations):
        run_interruptibly(
            lambda: architecture.train_planner(
                sess, train_data, validation_data, config.epochs),
            'planner training')
        rewards = architecture.train_reward(
            sess, image_irl, None, y_irl, config.reward_epochs)

    return rewards

def joint_algorithm(architecture, sess, train_data, validation_data,
                    reward_data, config):
    """Interruptibly trains the model + planner on the trajectories.

    Then infers reward. Currently, does not pretrain the model.
    In the future, probably worth looking into how to pretrain jointly.
    """
    assert not train_data and not validation_data, "joint pretraining not yet supported"

    image_irl, y_irl = reward_data

    rewards = architecture.train_joint(
            sess, image_irl, None, y_irl, config.epochs)

    return rewards

def vi_algorithm(architecture, sess, train_data, validation_data, reward_data, config):
    """Value Iteration:

    The only variable is the reward_tensor, allow it to be trained as you try VI to predict actions.
    """
    if train_data or validation_data:
        print("You're wasting compute by generating/loading in train/validation data.")
        print("Value Iteration does not require any data to make reward inferences.")
        print("")

    image_data, y_data = reward_data
    rewards = architecture.train_reward(sess, image_data=image_data, reward_data=None, y_data=y_data,
                    num_epochs=config.reward_epochs)
    return rewards

def infer_given_some_rewards(config):
    if config.verbosity >= 2:
        print('Assumption: We have some human data where the rewards are known')
    agent, other_agents = create_agents_from_config(config)
    train_data, validation_data = generate_data_for_planner(
        agent, config, other_agents)
    reward_data = generate_data_for_reward(agent, config, other_agents)
    return run_inference(train_data, validation_data, reward_data,
                         two_phase_algorithm, config)

def infer_with_rational_planner(config, beta=None):
    if config.verbosity >= 2:
        print('Using a rational planner with beta {} to mimic normal IRL'.format(beta))
    agent, other_agents = create_agents_from_config(config)
    optimal_agent = fast_agents.FastOptimalAgent(
        gamma=config.gamma, beta=beta, num_iters=config.num_iters)
    train_data, validation_data = generate_data_for_planner(
        optimal_agent, config, other_agents)
    reward_data = generate_data_for_reward(agent, config, other_agents)
    return run_inference(train_data, validation_data, reward_data,
                         two_phase_algorithm, config)

def infer_with_no_rewards(config):
    if config.verbosity >= 2:
        print('No rewards given, using the iterative EM-like algorithm')
    agent, other_agents = create_agents_from_config(config)
    optimal_agent = fast_agents.FastOptimalAgent(
        gamma=config.gamma, beta=config.beta, num_iters=config.num_iters)
    train_data, validation_data = generate_data_for_planner(
        optimal_agent, config, other_agents)
    reward_data = generate_data_for_reward(agent, config, other_agents)
    return run_inference(train_data, validation_data, reward_data,
                         iterative_algorithm, config)

def joint_infer_with_no_rewards(config):
    """Performs inference of reward by learning a trajectory.

    Backpropagation of reward and planner module performed jointly.
    """

    print("No rewards given, updating planner and inferred reward jointly")
    agent, other_agents = create_agents_from_config(config)
    # optimal_agent = agents.OptimalAgent(
    #     gamma=config.gamma, beta=config.beta, num_iters=config.num_iters)
    # train_data, validation_data = generate_data_for_planner(
    #     optimal_agent, config, other_agents)
    reward_data = generate_data_for_reward(agent, config, other_agents)
    return run_inference(None, None, reward_data,
                         joint_algorithm, config)

def infer_with_value_iteration(config):
    """ This uses a differentiable value iteration algorithm to infer rewards.
    It's basically just the reward inference part of infer_with_some_rewards, with model=Value_Iter
    """
    print("Using Value Iteration to infer rewards")
    agent, other_agents = create_agents_from_config(config)
    # No data for planning necessary
    reward_data = generate_data_for_reward(agent, config, other_agents)
    return run_inference(None, None, reward_data, vi_algorithm, config)

def run_algorithm(config):
    if config.algorithm == 'given_rewards':
        return infer_given_some_rewards(config)
    elif config.algorithm == 'boltzmann_planner':
        beta = config.beta if config.beta is not None else 1.0
        return infer_with_rational_planner(config, beta)
    elif config.algorithm == 'optimal_planner':
        return infer_with_rational_planner(config, None)
    elif config.algorithm == 'no_rewards':
        return infer_with_no_rewards(config)
    elif config.algorithm == 'joint_no_rewards':
        return joint_infer_with_no_rewards(config)
    elif config.algorithm == 'vi_inference':
        return infer_with_value_iteration(config)
    else:
        raise ValueError('Unknown algorithm: ' + str(config.algorithm))

if __name__=='__main__':
    # get flags || Data
    config = init_flags()
    final_accuracy, performance = run_algorithm(config)
    print("<1>{}<1>".format(final_accuracy))
    print("<2>{}<2>".format(performance))
