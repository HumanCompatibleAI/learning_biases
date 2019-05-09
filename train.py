# Some code taken from https://github.com/TheAbhiKumar/tensorflow-value-iteration-networks

import time
import numpy as np
import tensorflow as tf
import hashlib
import os
import pickle

import agents
import fast_agents
from gridworld.gridworld_data import generate_data_for_planner, generate_data_for_reward, create_agents_from_config
from model import create_model, calculate_action_distribution
from utils import fmt_row, init_flags, plot_reward_and_trajectories, set_seeds, concat_folder
from agent_runner import evaluate_proxy
from utils import plot_reward

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

        self.logits_cost = tf.add_n(tf.get_collection('losses'), name='logits_loss')
    
        if config.model == 'VIN' and config.vin_regularizer_C > 0:
            # TODO(rohinmshah): This assumes that no regularization has been
            # added besides to the VIN -- very errorprone, fix. Also add
            # regularization losses to the Model instead of computing them
            # here.
            vin_regularizer_cost = tf.add_n(
                tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES),
                name='vin_loss')
            # TODO(rohinmshah): Rename step1_cost and step2_cost
            self.step1_cost = self.logits_cost + vin_regularizer_cost
        else:
            self.step1_cost = self.logits_cost

        if config.reward_regularizer_C > 0:
            l1_regularizer = tf.contrib.layers.l1_regularizer(config.reward_regularizer_C)
            reward_regularizer_cost = tf.contrib.layers.apply_regularization(
                l1_regularizer, [self.reward])
            self.step2_cost = self.logits_cost + reward_regularizer_cost
        else:
            self.step2_cost = self.logits_cost

        # Naming step2_cost op
        self.step2_cost = tf.identity(self.step2_cost, "step2cost")

        # Define optimizers
        if config.model != 'VI':
            planner_optimizer = tf.train.AdamOptimizer(config.lr)
            tf.add_to_collection("optimizers", planner_optimizer)
            self.planner_optimize_op = planner_optimizer.minimize(self.step1_cost)
            tf.add_to_collection("optimizers", planner_optimizer)
            tf.add_to_collection("optimizer ops", self.planner_optimize_op)

        reward_optimizer = tf.train.AdamOptimizer(config.reward_lr)
        tf.add_to_collection("optimizers", reward_optimizer)
        self.reward_optimize_op = reward_optimizer.minimize(self.step2_cost, var_list=[self.reward])
        tf.add_to_collection("optimizer ops", self.reward_optimize_op)

        # Test model & calculate accuracy
        cp = tf.cast(tf.argmax(self.model.output_probs, 1), tf.int32)

        # Use the most probable action even for the gold labels
        most_likely_labels = tf.cast(tf.argmax(labels, axis=1), tf.int32)
        self.err = tf.reduce_mean(
            tf.cast(tf.not_equal(cp, most_likely_labels), dtype=tf.float32))
        self.err = tf.identity(self.err, "identity")

        # Initializing the variables
        self.initialize_op = tf.global_variables_initializer()

        # Saving model in SavedModel format
        # if config.log:
        #     self.builder = tf.saved_model.builder.SavedModelBuilder(
        #         config.logdir+'model/')
        # If process does not finish
        self.accuracy = "NA"

    def register_new_session(self, sess):
        # The tag on this model is to access the weights explicitly
        # I think SERVING vs TRAINING tags means you can save static & dynamic weights 4 a model
        self.sess = sess
        sess.run(self.initialize_op)
        # if self.config.log:
        #     self.builder.add_meta_graph_and_variables(
        #         sess, [tf.saved_model.tag_constants.SERVING])

    def evaluate_loss_and_err(self, sess, image_data, reward_data, y_data, logs):
        """Infers the reward using backprop, holding the planner fixed.

        Due to Tensorflow constraints, image_data must contain exactly
        batch_size number of MDPs on which the reward should be inferred.

        The rewards are initialized to the values in reward_data. If reward_data
        is None, the rewards are initialized to all zeroes.
        """
        batch_size = self.config.batchsize
        num_batches = int(image_data.shape[0] / batch_size)
        total_loss, total_err, num_samples = 0, 0, 0
        for batch_num in range(num_batches):
            start, end = batch_num * batch_size, (batch_num + 1) * batch_size
            images, rewards = image_data[start:end], reward_data[start:end]
            fd = {
                "image:0": images,
                "reward_input:0": rewards,
                "y:0": y_data[start:end]
            }
            loss, err = sess.run([self.logits_cost, self.err], feed_dict=fd)
            total_loss += (loss * len(images))
            total_err += (err * len(images))
            num_samples += len(images)
        return total_loss / num_samples, total_err / num_samples

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

    def train_planner(self, sess, train_data, validation_data, num_epochs, logs):
        """Trains the planner module given MDPs with reward functions and the
        corresponding policies.

        Validation can be turned off by explicitly passing None.
        """
        if self.config.verbosity >= 3:
            print(fmt_row(10, ["Epoch", "Train Cost", "Train Err", "Valid Err", "Epoch Time"]))

        avg_costs, train_errs, validation_errs, times = [], [], [], []
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
                avg_costs.append(avg_cost)
                train_errs.append(avg_err)
                validation_errs.append(validation_err)
                times.append(elapsed)
            elif self.config.verbosity >= 2 and epoch % self.config.display_step == 0:
                print('Epoch {} of {}'.format(epoch, num_epochs))

            # if self.config.log:
            #     summary = tf.Summary()
            #     summary.ParseFromString(sess.run(summary_op))
            #     summary.value.add(tag='Average error', simple_value=float(avg_err))
            #     summary.value.add(tag='Average cost', simple_value=float(avg_cost))
            #     summary_writer.add_summary(summary, epoch)
        logs['train_planner_costs'].append(avg_costs)
        logs['train_planner_train_errs'].append(train_errs)
        logs['train_planner_validation_errs'].append(validation_errs)
        logs['train_planner_times'].append(times)

        # TODO(rohinmshah): This seems redundant
        num_actions = self.config.num_actions
        action_dists = [np.zeros(num_actions), np.zeros(num_actions)]
        action_dists = [d + b_d for d, b_d in zip(action_dists, epoch_dist)]
        action_dists = [d / (np.sum(d)) for d in action_dists]
        pred = action_dists[0].tolist()
        actual = action_dists[1].tolist()
        logs['train_planner_predicted_action_dists'].append(pred)
        logs['train_planner_actual_action_dists'].append(actual)
        if self.config.verbosity >= 3:
            print("Action Distribution Comparison")
            print("------------------------------")
            print(fmt_row(10, ["Predicted"] + pred))
            print(fmt_row(10, ["Actual"]+ actual))

        if validation_data is not None:
            _, (err,), _, _ = self.run_epoch(sess, validation_data, [], [self.err])
            logs['accuracy'].append(100 * (1 - err))
            if self.config.verbosity >= 1:
                print('Validation Accuracy: ' + str(100 * (1 - err)))

        # Saving SavedModel instance
        if self.config.savemodel:
            saver = tf.train.Saver()
            # This allows for the model to perform reward inference
            saver.save(sess, "model_save_sess_0/")

    def train_reward(self, sess, image_data, reward_data, y_data, num_epochs, logs):
        """Infers the reward using backprop, holding the planner fixed.

        Due to Tensorflow constraints, image_data must contain exactly
        batch_size number of MDPs on which the reward should be inferred.

        The rewards are initialized to the values in reward_data. If reward_data
        is None, the rewards are initialized to all zeroes.
        """
        if self.config.verbosity >= 3:
            print(fmt_row(10, ["Iteration", "Train Cost", "Train Err", "Iter Time"]))
        if reward_data is None:
            reward_data = np.random.randn(*image_data.shape)

        batch_size = self.config.batchsize
        num_batches = int(image_data.shape[0] / batch_size)
        costs, errs = [], []
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

            if batch_num % 10 == 0:
                costs.append([])
                errs.append([])
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
                    costs[-1].append(c_)
                    errs[-1].append(e_)

            reward_data[start:end] = self.reward.eval()

        logs['train_reward_costs'].append(costs)
        logs['train_reward_errs'].append(errs)
        return reward_data

    def train_joint(self, sess, image_data, reward_data, y_data, num_epochs, logs):
        """Trains the planner module given MDPs with reward functions and the
        corresponding policies.

        Essentially performs train_reward, but also passes planner graph

        Validation can be turned off by explicitly passing None.
        """
        if self.config.verbosity >= 3:
            print(fmt_row(10, ["Iteration", "Train Cost", "Train Err", "Iter Time"]))
        if reward_data is None:
            # Initialize the reward array to random values
            reward_data = np.random.randn(*image_data.shape)

        planner_ops = [self.planner_optimize_op, self.step1_cost]
        batch_size = self.config.batchsize
        num_batches = int(image_data.shape[0] / batch_size)
        errors, costs, times = [], [], []
        for epoch in range(num_epochs):
            tstart = time.time()
            errors_per_batch, costs_per_batch = [], []
            for batch_num in range(num_batches):
                start, end = batch_num * batch_size, (batch_num + 1) * batch_size
                # We can't feed in reward_data directly to self.reward.
                # See train_reward for the explanation.
                fd = {
                    "reward_input:0": reward_data[start:end],
                }
                sess.run([self.assign_reward.op], feed_dict=fd)

                fd = {
                    "image:0": image_data[start:end],
                    "y:0": y_data[start:end]
                }

                # Run both step1 & step2 ops, report only error & step2 cost
                _, err, cost, _, _ = sess.run(
                    [self.reward_optimize_op, self.err, self.step2_cost] + planner_ops,
                    feed_dict=fd)
                errors_per_batch.append(err)
                costs_per_batch.append(cost)

            epoch_error = sum(errors_per_batch) / len(errors_per_batch)
            errors.append(epoch_error)
            epoch_cost = sum(costs_per_batch) / len(costs_per_batch)
            costs.append(epoch_cost)
            elapsed = time.time() - tstart
            times.append(elapsed)

            if self.config.verbosity >= 3:
                print(fmt_row(10, [epoch, epoch_cost, epoch_error, elapsed]))
                reward_data[start:end] = self.reward.eval()

        logs['train_joint_errs'].append(errors)
        logs['train_joint_costs'].append(costs)
        logs['train_joint_times'].append(times)
        self.accuracy = 100 * (1 - errors[-1])
        logs['accuracy'] = self.accuracy

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
                  algorithm_fn, architecture, config):
    """
    Evaluates a given algorithm_fn based upon its inferred reward

    Specifically, regret is calculated with respect to the reward_data

    :return: logs dictionary
    """
    # seed random number generators
    seed = config.seeds.pop(0)
    set_seeds(seed)
    logs = {
        'train_planner_costs': [],
        'train_planner_train_errs': [],
        'train_planner_validation_errs': [],
        'train_planner_times': [],
        'train_planner_predicted_action_dists': [],
        'train_planner_actual_action_dists': [],
        'train_reward_costs': [],
        'train_reward_errs': [],
        'train_joint_costs': [],
        'train_joint_errs': [],
        'train_joint_times': [],
        'accuracy': [],
    }
    # use flags to create model and retrieve relevant operations

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

    image_irl, reward_irl, start_states_irl, y_irl, image_test, start_states_test, y_test = reward_data
    reward_data = (image_irl, y_irl)

    # Launch the graph
    gpu_config = None
    if config.use_gpu:
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.25)
        gpu_config = tf.ConfigProto(gpu_options=gpu_options)

    with tf.Session(config=gpu_config) as sess:
        if config.log:
            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(config.logdir, sess.graph)

        architecture.register_new_session(sess)

        inferred_rewards = algorithm_fn(
            architecture, sess, train_data, validation_data, reward_data, config, logs)

        reward_percents = []
        for label, reward, wall, start_state, i in zip(reward_irl, inferred_rewards, image_irl, start_states_irl, range(len(reward_irl))):
            if config.plot_rewards and i < 10:
                plot_reward_and_trajectories(label, reward, wall, start_state, config, 'reward_pics/reward_{}'.format(i))
            reward_percents.append(
                evaluate_proxy(wall, start_state, reward, label, episode_length=20))

        average_percent_reward = float(sum(reward_percents)) / len(reward_percents)
        logs['Average %reward'] = average_percent_reward
        logs['Average %regret'] = 1 - average_percent_reward
        logs['%rewards'] = reward_percents

        avg_loss, err = architecture.evaluate_loss_and_err(
            sess, image_test, inferred_rewards, y_test, logs)
        logs['Average loss on test walls'] = avg_loss
        logs['Error on test walls'] = err
        logs['Accuracy on test walls'] = 1 - err

        if config.verbosity >= 1:
            print(reward_percents[:10])
            print('On average planning with the inferred rewards is '
                  + str(100 * average_percent_reward)
                  + '% as good as planning with the true rewards')
            print('Average loss on test walls: {}'.format(avg_loss))
            print('Accuracy on test walls: {}%'.format(100 * (1 - err)))

    return logs

def evaluate_inferred_reward(reward_irl, inferred_rewards, image_irl, start_states_irl, horizon):
    """
    Calculates 1-%regret for each (reward, inferred_reward) pair. Regret is amt of reward
    lost by planning with the inferred reward, rather than planning with the true reward.

    reward_irl(list): list of true rewards (n, imsize, imsize)
    inferred_rewards(list): list of inferred rewards (n, imsize, imsize)
    image_irl(list): list of walls for each grid (n, imsize, imsize):
    start_states_irl(list): list of start states (n, 2) where each entry of form [x,y]
    horizon(integer): number of steps to evaluate inferred reward
    Returns 1-%regret
    """
    reward_percents = []
    for label, reward, wall, start_state, i in zip(reward_irl, inferred_rewards, image_irl, start_states_irl, range(len(reward_irl))):
        if i < 10:
            plot_reward(label, reward, wall, 'reward_pics/reward_{}'.format(i))
        percent = evaluate_proxy(wall, start_state, reward, label, episode_length=horizon)
        print("Reward had: {}".format(percent))
        reward_percents.append(percent)

    average_percent_reward = float(sum(reward_percents)) / len(reward_percents)
    print(reward_percents[:10])
    print('On average planning with the inferred rewards is '
          + str(100 * average_percent_reward)
          + '% as good as planning with the true rewards')
    return average_percent_reward


def two_phase_algorithm(architecture, sess, train_data, validation_data,
                        reward_data, config, logs):
    config.em_iterations = 0
    return iterative_algorithm(
        architecture, sess, train_data, validation_data, reward_data, config, logs)

def iterative_algorithm(architecture, sess, train_data, validation_data,
                        reward_data, config, logs):
    """Iterative EM-like algorithm.

    The train and validation data are used to initialize the planner, but fine
    tuning of both the reward and the planner are done with reward_data.
    """
    image_irl, y_irl = reward_data
    if train_data and validation_data:
        run_interruptibly(
            lambda: architecture.train_planner(
                sess, train_data, validation_data, config.epochs, logs),
            'planner training')

    rewards = architecture.train_reward(
        sess, image_irl, None, y_irl, config.reward_epochs, logs)

    for i in range(config.em_iterations):
        train_data = image_irl, rewards, y_irl
        run_interruptibly(
            lambda: architecture.train_planner(
                sess, train_data, None, config.epochs, logs),
            'planner training')
        rewards = architecture.train_reward(
            sess, image_irl, rewards, y_irl, config.reward_epochs, logs)

    return rewards

def joint_algorithm(architecture, sess, train_data, validation_data,
                    reward_data, config, logs):
    """Interruptibly trains the model + planner on the trajectories.

    Then infers reward. Currently, does not pretrain the model.
    In the future, probably worth looking into how to pretrain jointly.
    """
    image_irl, y_irl = reward_data
    rewards = None
    if train_data and validation_data:
        run_interruptibly(
            lambda: architecture.train_planner(
                sess, train_data, validation_data, config.epochs, logs),
            'planner training')
        rewards = architecture.train_reward(
            sess, image_irl, rewards, y_irl, config.reward_epochs, logs)

    rewards = architecture.train_joint(
        sess, image_irl, rewards, y_irl, config.epochs, logs)
    return rewards

def vi_algorithm(architecture, sess, train_data, validation_data, reward_data,
                 config, logs):
    """Value Iteration:

    The only variable is the reward_tensor, allow it to be trained as you try VI to predict actions.
    """
    assert not train_data and not validation_data, "No planner training for VI"
    image_data, y_data = reward_data
    rewards = architecture.train_reward(
        sess, image_data, None, y_data, config.reward_epochs, logs)
    return rewards

def infer_given_some_rewards(config):
    if config.verbosity >= 2:
        print('Assumption: We have some human data where the rewards are known')
    architecture = PlannerArchitecture(config)
    agent, other_agents = create_agents_from_config(config)
    num_traj, num_with_reward = config.num_human_trajectories, config.num_with_rewards
    num_without_reward = make_evenly_batched(num_traj - num_with_reward, config)
    num_validation = config.num_validation
    num_train = num_with_reward - num_validation

    train_data, validation_data = generate_data_for_planner(
        num_train, num_validation, agent, config, other_agents)
    reward_data = generate_data_for_reward(
        num_without_reward, agent, config, other_agents)
    return run_inference(train_data, validation_data, reward_data, 
                         two_phase_algorithm, architecture, config)

def infer_with_rational_planner(config, beta=None):
    if config.verbosity >= 2:
        print('Using a rational planner with beta {} to mimic normal IRL'.format(beta))
    architecture = PlannerArchitecture(config)

    agent, other_agents = create_agents_from_config(config)
    num_without_reward = make_evenly_batched(config.num_human_trajectories, config)
    num_simulated, num_validation = config.num_simulated, config.num_validation

    optimal_agent = fast_agents.FastOptimalAgent(
        gamma=config.gamma, beta=beta, num_iters=config.num_iters)
    train_data, validation_data = generate_data_for_planner(
        num_simulated, num_validation, optimal_agent, config, other_agents)
    reward_data = generate_data_for_reward(
        num_without_reward, agent, config, other_agents)
    return run_inference(train_data, validation_data, reward_data, 
                         two_phase_algorithm, architecture, config)

def infer_with_no_rewards(config, train_jointly, initialize):
    if config.verbosity >= 2:
        s1 = 'jointly' if train_jointly else 'iteratively'
        s2 = 'with' if initialize else 'without'
        print('No rewards given, training planner and reward {} {} initialization'.format(s1, s2))
    architecture = PlannerArchitecture(config)

    agent, other_agents = create_agents_from_config(config)
    num_simulated, num_validation = config.num_simulated, config.num_validation
    num_without_reward = make_evenly_batched(config.num_human_trajectories, config)
    reward_data = generate_data_for_reward(
        num_without_reward, agent, config, other_agents)
    alg = joint_algorithm if train_jointly else iterative_algorithm

    train_data, validation_data = None, None
    if initialize:
        optimal_agent = fast_agents.FastOptimalAgent(
            gamma=config.gamma, beta=config.beta, num_iters=config.num_iters)
        train_data, validation_data = generate_data_for_planner(
            num_simulated, num_validation, optimal_agent, config, other_agents)
    return run_inference(train_data, validation_data, reward_data, alg, architecture, config)

def infer_with_value_iteration(config):
    """ This uses a differentiable value iteration algorithm to infer rewards.
    It's basically just the reward inference part of infer_with_some_rewards, with model=Value_Iter
    """
    print("Using Value Iteration to infer rewards")
    architecture = PlannerArchitecture(config)
    agent, other_agents = create_agents_from_config(config)
    num_without_reward = make_evenly_batched(config.num_human_trajectories, config)
    reward_data = generate_data_for_reward(
        num_without_reward, agent, config, other_agents)
    return run_inference(None, None, reward_data, vi_algorithm, architecture, config)

def infer_with_max_causal_ent(config):
    """Uses Adam's code to implement Max Causal Entropy for our gridworld MDP."""
    # Importing only when used because PyTorch is dependency for maxent (as of 4/5)
    from maxent import irl_with_config

    print("Using Max Causal Entropy (source @AdamGleave)")

    agent, other_agents = create_agents_from_config(config)
    walls, rewards, starts, policies = generate_data_for_reward(agent, config, other_agents)

    inferred_rewards = []
    verbose = False
    for i, wall, pol, start in zip(range(len(walls)), walls, policies, starts):
        if i % 5 == 0:
            print("Running IRL on grid number: {} / {}".format(i, len(walls)))
            verbose = True
        inferred = irl_with_config(wall, pol, start, config, verbose=verbose)
        inferred_rewards.append(inferred)
        verbose = False

    avg_percent_reward = evaluate_inferred_reward(rewards, inferred_rewards, walls, starts, config.horizon)
    return None, avg_percent_reward



def run_algorithm(config):
    if config.algorithm == 'given_rewards':
        return infer_given_some_rewards(config)
    elif config.algorithm == 'boltzmann_planner':
        beta = config.beta if config.beta is not None else 1.0
        return infer_with_rational_planner(config, beta)
    elif config.algorithm == 'optimal_planner':
        return infer_with_rational_planner(config, None)
    elif config.algorithm == 'joint_with_init':
        return infer_with_no_rewards(config, train_jointly=True, initialize=True)
    elif config.algorithm == 'joint_without_init':
        return infer_with_no_rewards(config, train_jointly=True, initialize=False)
    elif config.algorithm == 'em_with_init':
        return infer_with_no_rewards(config, train_jointly=False, initialize=True)
    elif config.algorithm == 'em_without_init':
        return infer_with_no_rewards(config, train_jointly=False, initialize=False)
    elif config.algorithm == 'vi_inference':
        return infer_with_value_iteration(config)
    elif config.algorithm == 'max_entropy':
        return infer_with_max_causal_ent(config)
    else:
        raise ValueError('Unknown algorithm: ' + str(config.algorithm))


def make_evenly_batched(n, config):
    # It is required that the number of unknown reward functions be divisible by
    # the batch size, due to Tensorflow constraints.
    if n % config.batchsize != 0:
        n = n - (n % config.batchsize)
        print('Reducing to {} MDPs to be divisible by the batch size'.format(n))
    return n


def get_output_stuff(config, seeds):
    IGNORED_FLAGS = ['output_folder', 'seeds']
    # flags_dict = config.__dict__['__flags']  # Hacky but works
    # flags_dict = {k:v for k, v in flags_dict.items() if k not in IGNORED_FLAGS}
    flags_dict = dir(config)
    flags_dict = {k: config[k].value for k in flags_dict if k not in IGNORED_FLAGS}

    kvs = tuple(sorted(flags_dict.items()))
    kv_hash = hashlib.sha224(str(kvs).encode()).hexdigest()
    folder = concat_folder(config.output_folder, kv_hash)

    seed_str = ','.join([str(seed) for seed in seeds])
    filename = concat_folder(folder, 'seeds-{}.npz'.format(seed_str))
    return flags_dict, folder, filename

def save_results(logs, config, seeds):
    flags_dict, folder, filename = get_output_stuff(config, seeds)
    if not os.path.exists(folder):
        os.makedirs(folder)
        with open(concat_folder(folder, 'flags.pickle'), 'wb') as f:
            pickle.dump(flags_dict, f)

    if os.path.exists(filename):
        print('Warning: Overwriting existing file {}'.format(filename))
    logs = {k:np.array(v) for k, v in logs.items()}
    np.savez(filename, **logs)

def results_present(config, seeds):
    _, _, filename = get_output_stuff(config, seeds)
    return os.path.exists(filename)


def main():
    # get flags || Data
    config = init_flags()
    seeds = config.seeds[:]
    # if results_present(config, seeds):
    #     print('Results already present!')
    #     return
    logs = run_algorithm(config)
    # save_results(logs, config, seeds)
    # For bash scripts which read from stdout
    print("<1>N/A<1>")
    print("<2>{}<2>".format(logs['Average %reward']))


if __name__=='__main__':
    main()
