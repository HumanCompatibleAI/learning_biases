# Code taken from https://github.com/TheAbhiKumar/tensorflow-value-iteration-networks

import time
import numpy as np
import random
import tensorflow as tf

import agents
from gridworld_data import generate_gridworld_irl, load_dataset
from model import VI_Block, simple_model, add_distribution
from utils import fmt_row, init_flags, plot_reward

import sys

def model_declaration(config):
    """Create model using config flags and also create model saver.

    Note: this function does initiate variables (necessary for SavedModel api)
    """

    # Tensorflow refuses to have a Variable whose shape is not fully determined. As
    # a result, we must set the batch size to a constant which cannot be changed
    # during a particular run. (We need to use a Variable for the reward so that the
    # reward can be trained in step 2.)
    batch_size = config.batchsize
    imsize = config.imsize
    num_actions = config.num_actions

    image = tf.placeholder(
        tf.float32, name="image", shape=[batch_size, imsize, imsize])
    reward = tf.Variable(
        tf.zeros([batch_size, imsize, imsize]), name='reward', trainable=False)
    X  = tf.stack([image, reward], axis=-1)
    y  = tf.placeholder(
        tf.float32, name="y",  shape=[batch_size, imsize, imsize, num_actions])

    if config.model == 'VIN':
        # Construct model (Value Iteration Network)
        print("vin")
        logits, nn = VI_Block(X, config)
    elif config.model == "SIMPLE":
        # Construct model (Simple Model)
        print("simple model")
        logits, nn = simple_model(X, config)
        print("simple shape:",logits.get_shape())

    # Add tensors to calculate action distributions
    pred_dist = add_distribution(logits, config.batchsize, config.ch_q,name='pred_action_dist')
    y_dist = add_distribution(y, config.batchsize, config.ch_q,name='true_action_dist')

    # Reshape for losses
    logits = tf.reshape(logits, [-1, num_actions])
    y = tf.reshape(y, [-1, num_actions])

    # Define losses
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=y, name='cross_entropy')
    cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
    tf.add_to_collection('losses', cross_entropy_mean)

    logits_cost = tf.add_n(tf.get_collection('losses'), name='logits_loss')
    
    if config.model == 'VIN' and config.vin_regularizer_C > 0:
        vin_regularizer_cost = tf.add_n(
            tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES), name='vin_loss')
        step1_cost = logits_cost + vin_regularizer_cost
    else:
        step1_cost = logits_cost

    if config.reward_regularizer_C > 0:
        l1_regularizer = tf.contrib.layers.l1_regularizer(config.reward_regularizer_C)
        reward_regularizer_cost = tf.contrib.layers.apply_regularization(
            l1_regularizer, [reward])
        step2_cost = logits_cost + reward_regularizer_cost
    else:
        step2_cost = logits_cost

    # Define optimizers
    planner_optimizer = tf.train.RMSPropOptimizer(
        learning_rate=config.lr, epsilon=1e-6, centered=True)
    planner_optimize_op = planner_optimizer.minimize(step1_cost)
    reward_optimizer = tf.train.RMSPropOptimizer(
        learning_rate=config.reward_lr, epsilon=1e-6, centered=True)
    reward_optimize_op = reward_optimizer.minimize(step2_cost, var_list=[reward])

    # Test model & calculate accuracy
    cp = tf.cast(tf.argmax(nn, 1), tf.int32)
    # Use the most probable action even for the gold labels
    most_likely_y = tf.cast(tf.argmax(y, axis=1), tf.int32)
    err = tf.reduce_mean(tf.cast(tf.not_equal(cp, most_likely_y), dtype=tf.float32))

    # Initializing the variables
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

    # Saving model in SavedModel format
    builder = tf.saved_model.builder.SavedModelBuilder(config.logdir+'model/')

    return (builder, init, saver), (err, step1_cost, step2_cost), \
            (planner_optimize_op, reward_optimize_op), reward

if __name__=='__main__':
    # get flags || Data
    config = init_flags()
    # seed random generators
    np.random.seed(config.seed)
    random.seed(config.seed)
    # use flags to create model and retrieve relevant operations
    saver_ops, cost_and_err, optimizers, reward = model_declaration(config)
    builder, init, saver = saver_ops
    err, step1_cost, step2_cost = cost_and_err
    planner_optimize_op, reward_optimize_op = optimizers

    if config.datafile:
        imagetrain, rewardtrain, ytrain, \
        imagetest1, rewardtest1, ytest1, \
        imagetest2, rewardtest2, ytest2 = load_dataset(config.datafile)
    else:
        imagetrain, rewardtrain, ytrain, \
        imagetest1, rewardtest1, ytest1, \
        imagetest2, rewardtest2, ytest2 = generate_gridworld_irl(config)

    batch_size = config.batchsize
    imsize = config.imsize
    num_actions = config.num_actions

    # Launch the graph
    with tf.Session() as sess:
        if config.log:
            for var in tf.trainable_variables():
                tf.summary.histogram(var.op.name, var)
            summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(config.logdir, sess.graph)
        sess.run(init)

        # The tag on this model is to access the weights explicitly
        # I think SERVING vs TRAINING tags means you can save static & dynamic weights 4 a model
        builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING])

        def run_epoch(data, ops_to_run, ops_to_average):
            tstart = time.time()
            image_data, reward_data, y_data = data
            averages = [0.0] * len(ops_to_average)
            num_batches = int(image_data.shape[0] / batch_size)
            # Loop over all batches
            for i in range(num_batches):
                start, end = i * batch_size, (i + 1) * batch_size
                fd = {
                    "image:0": image_data[start:end],
                    "reward:0": reward_data[start:end],
                    "y:0": y_data[start:end]
                }
                results = sess.run(ops_to_run + ops_to_average, feed_dict=fd)
                num_ops_to_run = len(ops_to_run)
                op_results, average_op_results = results[:num_ops_to_run], results[num_ops_to_run:]
                averages = [x + y for x, y in zip(averages, average_op_results)]
            
            averages = [x / num_batches for x in averages]
            elapsed = time.time() - tstart
            return op_results, averages, elapsed

        train_data = (imagetrain, rewardtrain, ytrain)
        test1_data = (imagetest1, rewardtest1, ytest1)

        print(fmt_row(10, ["Epoch", "Train Cost", "Train Err", "Valid Err", "Epoch Time"]))
        try:
            for epoch in range(int(config.epochs)):
                _, (avg_cost, avg_err), elapsed = run_epoch(
                    train_data, [planner_optimize_op], [step1_cost, err])
                # Display logs per epoch step
                if epoch % config.display_step == 0:
                    _, (test1_err,), _ = run_epoch(test1_data, [], [err])
                    print(fmt_row(10, [epoch, avg_cost, avg_err, test1_err, elapsed]))
                if config.log:
                    summary = tf.Summary()
                    summary.ParseFromString(sess.run(summary_op))
                    summary.value.add(tag='Average error', simple_value=float(avg_err))
                    summary.value.add(tag='Average cost', simple_value=float(avg_cost))
                    summary_writer.add_summary(summary, epoch)
                    # saver.save(sess, config.logdir)
        except KeyboardInterrupt:
            print("Step 1 skipped")
            pass
      
        print("Finished training!")
        _, (test1_err,), _ = run_epoch(test1_data, [], [err])
        # saving SavedModel instance
        savepath = builder.save()
        print("model saved 2: {}".format(savepath))
        print('Final Accuracy: ' + str(100 * (1 - test1_err)))

        print('Beginning IRL inference')
        print(fmt_row(10, ["Iteration", "Train Cost", "Train Err", "Iter Time"]))
        try:
            for epoch in range(config.reward_epochs):
                tstart = time.time()
                fd = {
                    "image:0": imagetest2,
                    "y:0": ytest2,
                }
                _, predicted_reward, e_, c_ = sess.run(
                    [reward_optimize_op, reward, err, step2_cost], feed_dict=fd)
                elapsed = time.time() - tstart
                print(fmt_row(10, [epoch, c_, e_, elapsed]))
        except KeyboardInterrupt:
            print("Step 2 skipped")
            pass

        print('The first set of walls is:')
        print(imagetest2[0])
        print('The first reward should be:')
        print(rewardtest2[0])
        inferred_reward = reward.eval()[0]
        normalized_inferred_reward = inferred_reward / inferred_reward.max()
        print('The inferred reward is:')
        print(normalized_inferred_reward)

        plot_reward(rewardtest2[0], normalized_inferred_reward)


