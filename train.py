# Some code taken from https://github.com/TheAbhiKumar/tensorflow-value-iteration-networks

import time
import numpy as np
import random
import tensorflow as tf
import pdb

import agents
from gridworld_data import generate_gridworld_irl, load_dataset
from model import create_model, calculate_action_distribution
from utils import fmt_row, init_flags, plot_reward
from agent_runner import run_agent_proxy
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
        self.reward = tf.Variable(
            tf.zeros([batch_size, imsize, imsize]), name='reward', trainable=False)
        self.y  = tf.placeholder(
            tf.float32, name="y",  shape=[batch_size, imsize, imsize, num_actions])

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
        self.builder = tf.saved_model.builder.SavedModelBuilder(
            config.logdir+'model/')

    def register_new_session(self, sess):
        # The tag on this model is to access the weights explicitly
        # I think SERVING vs TRAINING tags means you can save static & dynamic weights 4 a model
        sess.run(self.initialize_op)
        self.builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING])

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

    def train_planner(self, sess, train_data, validation_data, num_epochs, print_output=True):
        """Trains the planner module given MDPs with reward functions and the
        corresponding policies.
        """
        if print_output:
            print(fmt_row(10, ["Epoch", "Train Cost", "Train Err", "Valid Err", "Epoch Time"]))

        num_actions = self.config.num_actions
        action_dists = [np.zeros(num_actions), np.zeros(num_actions)]
        for epoch in range(int(num_epochs)):
            _, (avg_cost, avg_err), elapsed, epoch_dist = self.run_epoch(
                sess, train_data, [self.planner_optimize_op],
                [self.step1_cost, self.err], [self.pred_dist, self.y_dist])

            # Display logs per epoch step
            if print_output and epoch % self.config.display_step == 0:
                _, (validation_err,), _, _ = self.run_epoch(sess, validation_data, [], [self.err])
                print(fmt_row(10, [epoch, avg_cost, avg_err, validation_err, elapsed]))
            if self.config.log:
                summary = tf.Summary()
                summary.ParseFromString(sess.run(summary_op))
                summary.value.add(tag='Average error', simple_value=float(avg_err))
                summary.value.add(tag='Average cost', simple_value=float(avg_cost))
                summary_writer.add_summary(summary, epoch)

        action_dists = [d + b_d for d, b_d in zip(action_dists, epoch_dist)]
        action_dists = [d / (np.sum(d)) for d in action_dists]
        if print_output and action_dists:
            print("Action Distribution Comparison")
            print("------------------------------")
            print(fmt_row(10, ["Predicted"] + action_dists[0].tolist()))
            print(fmt_row(10, ["Actual"]+ action_dists[1].tolist()))

        if print_output:
            _, (validation_err,), _, _ = self.run_epoch(sess, validation_data, [], [self.err])
            # Saving SavedModel instance
            savepath = self.builder.save()
            print("Model saved to: {}".format(savepath))
            print('Final Accuracy: ' + str(100 * (1 - validation_err)))

    def train_reward(self, sess, image_data, y_data, num_epochs, print_output=True):
        """Infers the reward using backprop, holding the planner fixed.

        Due to Tensorflow constraints, image_data must contain exactly
        batch_size number of MDPs on which the reward should be inferred.
        """
        # TODO(rohinmshah): We can get this to work with arbitrary numbers of
        # MDPs by inferring the reward functions and saving the resulting
        # Tensors outside of a Tensorflow Variable, and then putting them back
        # in to Tensorflow whenever necessary (when calling sess.run).
        if print_output:
            print(fmt_row(10, ["Iteration", "Train Cost", "Train Err", "Iter Time"]))

        for epoch in range(num_epochs):
            tstart = time.time()
            fd = {
                "image:0": image_data,
                "y:0": y_data,
            }
            _, predicted_reward, e_, c_ = sess.run(
                [self.reward_optimize_op, self.reward, self.err, self.step2_cost],
                feed_dict=fd)
            elapsed = time.time() - tstart
            if print_output:
                print(fmt_row(10, [epoch, c_, e_, elapsed]))

def run_interruptibly(fn, step_name='this step'):
    """Runs fn in a mode where KeyboardInterrupts will interrupt fn but will
    then continue with the rest of the program execution.
    """
    try:
        return fn()
    except KeyboardInterrupt:
        print('Skipping the rest of ' + step_name)

def go():
    # get flags || Data
    config = init_flags()
    # seed random generators
    np.random.seed(config.seed)
    random.seed(config.seed)
    # use flags to create model and retrieve relevant operations
    architecture = PlannerArchitecture(config)

    if config.datafile:
        imagetrain, rewardtrain, ytrain, \
        imagetest1, rewardtest1, ytest1, \
        imagetest2, rewardtest2, ytest2 = load_dataset(config.datafile)
    else:
        imagetrain, rewardtrain, ytrain, \
        imagetest1, rewardtest1, ytest1, \
        imagetest2, rewardtest2, ytest2 = generate_gridworld_irl(config)

    # Launch the graph
    with tf.Session() as sess:
        if config.log:
            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(config.logdir, sess.graph)

        architecture.register_new_session(sess)
        train_data = (imagetrain, rewardtrain, ytrain)
        test1_data = (imagetest1, rewardtest1, ytest1)

        run_interruptibly(
            lambda: architecture.train_planner(
                sess, train_data, test1_data, config.epochs),
            'planner training')

        run_interruptibly(
            lambda: architecture.train_reward(
                sess, imagetest2, ytest2, config.reward_epochs),
            'reward training')

        print('The first set of walls is:')
        print(imagetest2[0])
        print('The first reward should be:')
        print(rewardtest2[0])
        inferred_reward = architecture.reward.eval()[0]
        # normalized_inferred_reward = inferred_reward / inferred_reward.max()
        print('The inferred reward is:')
        print(inferred_reward)

        for label, reward, wall, i in zip(rewardtest2, architecture.reward.eval(), imagetest2, range(len(rewardtest2))):
            plot_reward(label, reward, wall, 'reward_pics/reward_{}'.format(i))
            pdb.set_trace()
            trajectory, proxy_r, true_r = run_agent_proxy(wall, reward, label)


if __name__=='__main__':
    go()
