# Code taken from https://github.com/TheAbhiKumar/tensorflow-value-iteration-networks

import time
import numpy as np
import random
import tensorflow as tf
import matplotlib
matplotlib.use("tkagg")
import matplotlib.pyplot as plt


import agents
from gridworld_data import generate_gridworld_irl
from model import VI_Block, VI_Untied_Block
from utils import fmt_row
# from tf.saved_model.tag_constants import SERVING, TRAINING

import sys
sys.path.insert(0, '../tensorflow-value-iteration-networks')

# Data
tf.app.flags.DEFINE_integer('imsize',         14,
                            'Size of input image')
tf.app.flags.DEFINE_float('wall_prob',      0.05,
                            'Probability of having a wall at any particular space in the gridworld')
tf.app.flags.DEFINE_float('reward_prob',    0,
                            'Probability of having a reward at any particular space in the gridworld')
# Parameters
tf.app.flags.DEFINE_float('lr',               0.001,
                          'Learning rate for RMSProp')
tf.app.flags.DEFINE_integer('epochs',         100,
                            'Maximum epochs to train for')
tf.app.flags.DEFINE_integer('reward_epochs',  10,
                            'Number of epochs to run when inferring reward function')
tf.app.flags.DEFINE_integer('k',              10,
                            'Number of value iterations')
tf.app.flags.DEFINE_integer('ch_h',           150,
                            'Channels in initial hidden layer')
tf.app.flags.DEFINE_integer('ch_q',           5,
                            'Channels in q layer (~actions)')
tf.app.flags.DEFINE_integer('batchsize',      12,
                            'Batch size')
tf.app.flags.DEFINE_integer('statebatchsize', 10,
                            'Number of state inputs for each sample (real number, technically is k+1)')
tf.app.flags.DEFINE_boolean('untied_weights', False,
                            'Untie weights of VI network')
# Agents
tf.app.flags.DEFINE_string('agent',           'optimal',
                           'Agent to generate training data with')
tf.app.flags.DEFINE_float('gamma',            1.0,
                          'Discount factor')
tf.app.flags.DEFINE_float('beta',             None,
                          'Noise when selecting actions')
tf.app.flags.DEFINE_integer('num_iters',      50,
                            'Number of iterations of value iteration the agent should run.')
tf.app.flags.DEFINE_integer('max_delay',      5,
                            'Maximum delay that the agent should use. Only affects naive/sophisticated and myopic agents.')
tf.app.flags.DEFINE_float('hyperbolic_constant', 1.0,
                          'Discount for the future for hyperbolic time discounters')
# Misc.
tf.app.flags.DEFINE_integer('seed',           0,
                            'Random seed for numpy')
tf.app.flags.DEFINE_integer('display_step',   1,
                            'Print summary output every n epochs')
tf.app.flags.DEFINE_boolean('log',            False,
                            'Enable for tensorboard summary')
tf.app.flags.DEFINE_string('logdir',          '/tmp/planner-vin/',
                           'Directory to store tensorboard summary')
tf.app.flags.DEFINE_integer("num_train_ex", 2000,
                            'Number of training examples')
tf.app.flags.DEFINE_integer("num_test_ex", 300,
                            'Number of testing examples')

config = tf.app.flags.FLAGS

np.random.seed(config.seed)
random.seed(config.seed)

# Tensorflow refuses to have a Variable whose shape is not fully determined. As
# a result, we must set the batch size to a constant which cannot be changed
# during a particular run. (We need to use a Variable for the reward so that the
# reward can be trained in step 2.)
batch_size, state_batch_size = config.batchsize, config.statebatchsize
imsize = config.imsize
image = tf.placeholder(
    tf.float32, name="image", shape=[batch_size, imsize, imsize])
reward = tf.Variable(
    tf.zeros([batch_size, imsize, imsize]), name='reward', trainable=False)
X  = tf.stack([image, reward], axis=-1)
# symbolic input batches of vertical positions
S1 = tf.placeholder(tf.int32, name="S1", shape=[batch_size, state_batch_size])
# symbolic input batches of horizontal positions
S2 = tf.placeholder(tf.int32, name="S2", shape=[batch_size, state_batch_size])
y  = tf.placeholder(tf.int32, name="y",  shape=[batch_size * state_batch_size])

# Construct model (Value Iteration Network)
if (config.untied_weights):
    logits, nn = VI_Untied_Block(X, S1, S2, config)
else:
    logits, nn = VI_Block(X, S1, S2, config)

# Define loss and optimizers
y_ = tf.cast(y, tf.int64)
cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=logits, labels=y_, name='cross_entropy')
cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy_mean')
tf.add_to_collection('losses', cross_entropy_mean)

cost = tf.add_n(tf.get_collection('losses'), name='total_loss')
planner_optimizer = tf.train.RMSPropOptimizer(learning_rate=config.lr, epsilon=1e-6, centered=True).minimize(cost)
reward_optimizer = tf.train.RMSPropOptimizer(learning_rate=config.lr * 100, epsilon=1e-6, centered=True).minimize(cost, var_list=[reward])

# Test model & calculate accuracy
cp = tf.cast(tf.argmax(nn, 1), tf.int32)
err = tf.reduce_mean(tf.cast(tf.not_equal(cp, y), dtype=tf.float32))

# Initializing the variables
init = tf.global_variables_initializer()
saver = tf.train.Saver()

# Saving model in SavedModel format
builder = tf.saved_model.builder.SavedModelBuilder(config.logdir+'model/')

imagetrain, rewardtrain, S1train, S2train, ytrain, \
imagetest1, rewardtest1, S1test1, S2test1, ytest1, \
imagetest2, rewardtest2, S1test2, S2test2, ytest2 = generate_gridworld_irl(
    config, config.num_train_ex, config.num_test_ex, batch_size
)
ytrain = np.reshape(ytrain, [-1])
ytest1 = np.reshape(ytest1, [-1])
ytest2 = np.reshape(ytest2, [-1])

# Launch the graph
with tf.Session() as sess:
    if config.log:
        for var in tf.trainable_variables():
            tf.summary.histogram(var.op.name, var)
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(config.logdir, sess.graph)
    sess.run(init)
    builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING])

    def run_epoch(data, ops_to_run, ops_to_average):
        tstart = time.time()
        image_data, reward_data, S1_data, S2_data, y_data = data
        averages = [0.0] * len(ops_to_average)
        num_batches = int(image_data.shape[0] / batch_size)
        # Loop over all batches
        for i in range(num_batches):
            start, end = i * batch_size, (i + 1) * batch_size
            fd = {
                image: image_data[start:end],
                reward: reward_data[start:end],
                S1: S1_data[start:end],
                S2: S2_data[start:end],
                y: y_data[start * state_batch_size:end * state_batch_size]
            }
            results = sess.run(ops_to_run + ops_to_average, feed_dict=fd)
            num_ops_to_run = len(ops_to_run)
            op_results, average_op_results = results[:num_ops_to_run], results[num_ops_to_run:]
            averages = [x + y for x, y in zip(averages, average_op_results)]
        
        averages = [x / num_batches for x in averages]
        elapsed = time.time() - tstart
        return op_results, averages, elapsed

    train_data = (imagetrain, rewardtrain, S1train, S2train, ytrain)
    test1_data = (imagetest1, rewardtest1, S1test1, S2test1, ytest1)

    print(fmt_row(10, ["Epoch", "Train Cost", "Train Err", "Valid Err", "Epoch Time"]))
    for epoch in range(int(config.epochs)):
        _, (avg_cost, avg_err), elapsed = run_epoch(
            train_data, [planner_optimizer], [cost, err])
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
  
    print("Finished training!")
    _, (test1_err,), _ = run_epoch(test1_data, [], [err])
    # saving SavedModel instance
    savepath = builder.save()
    print("model saved 2: {}".format(savepath))
    print('Final Accuracy: ' + str(100 * (1 - test1_err)))

    print('Beginning IRL inference')
    # It is required that the number of unknown reward functions be equal to the
    # batch size. If we tried to train multiple batches, then they would all be
    # modifying the same reward function, which would be bad.
    print(fmt_row(10, ["Iteration", "Train Cost", "Train Err", "Iter Time"]))
    for epoch in range(config.reward_epochs):
        tstart = time.time()
        fd = {
            image: imagetest2,
            S1: S1test2,
            S2: S2test2,
            y: ytest2,
        }
        _, predicted_reward, e_, c_ = sess.run(
            [reward_optimizer, reward, err, cost], feed_dict=fd)
        elapsed = time.time() - tstart
        print(fmt_row(10, [epoch, c_, e_, elapsed]))

    # this saves reward
    fig, axes = plt.subplots(1,2)
    print('The first reward should be:')
    print(rewardtest2[0])
    print('The inferred reward is:')
    print(reward.eval()[0])
    true = axes[0].imshow(rewardtest2[0],cmap='hot',interpolation='nearest')
    axes[0].set_title("Truth")
    cbaxes = fig.add_axes([0.02, 0.1, 0.02, 0.8])
    cb = plt.colorbar(true, cax=cbaxes)
    tensor = axes[1].imshow(reward.eval()[0],cmap='hot',interpolation='nearest')
    axes[1].set_title("Predicted")
    cbaxes2 = fig.add_axes([0.925, 0.1, 0.02, 0.8])
    plt.colorbar(tensor, cax=cbaxes2)
    # plt.colorbar.make_axes(axes[1], location='left')
    fig.suptitle("Comparison of Reward Functions")
    fig.savefig("predictioneval")

