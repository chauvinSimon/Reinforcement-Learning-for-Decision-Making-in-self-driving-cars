"""
This part of code is the Deep Q Network (DQN) brain.

view the tensorboard picture about this DQN structure on:
https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/4-3-DQN3/#modification

View more on my tutorial page: https://morvanzhou.github.io/tutorials/

Observation:
- a big loss (>0 at each last position) - because of r=40 (vs r=-2 or -40)
- copy of net work
- in learning, it modifies almost all the table (especially all the values for action a)
- in learning, it does not modify the values of other actions at the state s
- best so far: with default parameters. But learn every 5 steps (counter = 0 at each start of episode)
- despite clipping and despite normalization function, q goes high (max = 800, min = 70)
- normalizing the reward did not bring anything (although in the initial code, r=+1; 0 or +1)
- -67 and -156 are common returns

To do:
- random choice if several action candidates with the same value
- decay of epsilon

Using:
Tensorflow: r1.2
"""

import numpy as np
import tensorflow as tf
import os
import sys

if "../" not in sys.path:
    sys.path.append("../")

np.random.seed(1)
tf.set_random_seed(1)

# Create a global step variable
global_step = tf.Variable(0, name='global_step', trainable=False)


def normalization_function(targets_batch_to_normalize):
    mini_target_value = -100
    maxi_target_value = +100
    res = (targets_batch_to_normalize - mini_target_value) / (maxi_target_value - mini_target_value)

    return res


def inverse_normalization_function(targets_batch_to_inverse_normalize):
    mini_target_value = -10000
    maxi_target_value = +10000
    res = targets_batch_to_inverse_normalize * (maxi_target_value - mini_target_value) + mini_target_value

    return res


def normalize_reward(r):
    # Parameters to rescale the rewards, but don’t shift mean, as that affects agent’s will to live
    mini_reward = -50
    maxi_reward = 40
    reward = r / (maxi_reward - mini_reward)
    return reward


# Deep Q Network off-policy
class DeepQNetwork:
    def __init__(
            self,
            actions,
            state,
            learning_rate=0.01,
            reward_decay=0.9,
            # e_greedy=0.9,
            replace_target_iter=300,
            memory_size=50,
            batch_size=32,
            e_greedy_increment=None,
            summaries_dir=None,  # log-dir
            saver_dir=None
    ):

        self.actions_list = actions
        self.action_taken = None
        self.state_features_list = state

        self.n_actions = len(actions)
        self.n_features = len(state)
        self.lr = learning_rate
        self.gamma = reward_decay
        # self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        # self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        # total learning step
        self.learning_counter = 0

        # initialize zero memory [s, a, r, s_]
        # each row is like [state, action, reward, state] = ['0' '3' 'slow_down_down' '-7' '1' '1']
        # self.memory = np.zeros((self.memory_size, self.n_features * 2 + 2))
        self.memory = np.zeros((self.memory_size, self.n_features * 2 + 2))

        # consist of [target_net, evaluate_net]
        self._build_net()

        # get all the parameters from the two nets
        # t_params and e_params are lists of values in the collection with the given name
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_net')
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='eval_net')

        # replacement
        with tf.variable_scope('soft_replacement'):
            self.target_replace_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())
        self.cost_his = []
        self.memory_counter = 0

        if summaries_dir:
            summary_dir = os.path.join(summaries_dir, "summaries_DQN_{}".format("eval_net"))
            if not os.path.exists(summary_dir):
                os.makedirs(summary_dir)  # create folder

            # Create the log writer object (= FileWriter)
            # = create an event file in a given directory and add summaries and events to it
            self.summary_writer = tf.summary.FileWriter(
                summary_dir,  # log directory
                graph=self.sess.graph  # Adds a Graph to the event file.
            )

        self.saver_dir = saver_dir
        if saver_dir:
            self.saver = tf.train.Saver()

            # Restore variables from disk.
            self.saver.restore(self.sess, self.saver_dir)
            print("Model restored.")
            # Check the values of the variables
            # print("v1 : %s" % eval_net.eval())
            # print("v2 : %s" % .eval())

    def _build_net(self):
        # ------------------ all inputs ------------------------
        # Like q-learning: S.A.R.S.
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input State
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')  # input Next State
        self.r = tf.placeholder(tf.float32, [None, ], name='r')  # input Reward
        self.a = tf.placeholder(tf.int32, [None, ], name='a')  # input Action

        # Initializer that generates tensors with constant/random values.
        w_initializer = tf.random_normal_initializer(0., 0.3)
        b_initializer = tf.constant_initializer(0.1)

        # ------------------ build evaluate_net ------------------
        # Tensorboard: To clean up the visualization of our model in tensorboard we need to add the scope of our
        # variables and a name for our placeholders and variables.

        with tf.variable_scope('eval_net'):
            # FC with activation ReLu: outputs (=e1) = activation(inputs.kernel + bias)
            # units: Integer or Long, dimensionality of the output space = 20 hidden layers
            e1 = tf.layers.dense(
                inputs=self.s,
                units=128,  # 20 hidden layers
                activation=tf.nn.relu,  # tf.contrib.fully_connected has relu as it's default activation,
                # while tf.layers.dense is a linear activation by default.
                kernel_initializer=w_initializer,
                bias_initializer=b_initializer,
                name='e1'
            )

            self.q_eval = tf.layers.dense(
                inputs=e1,
                units=self.n_actions,  # output is an action
                kernel_initializer=w_initializer,
                bias_initializer=b_initializer,
                name='q'
            )

        # ------------------ build target_net ------------------
        with tf.variable_scope('target_net'):
            t1 = tf.layers.dense(
                inputs=self.s_,
                units=128,  # also 20 hidden layers
                activation=tf.nn.relu,
                kernel_initializer=w_initializer,
                bias_initializer=b_initializer,
                name='t1'
            )

            # it should be renamed q_target ?
            self.q_next = tf.layers.dense(
                inputs=t1,
                units=self.n_actions,
                kernel_initializer=w_initializer,
                bias_initializer=b_initializer,
                name='t2'
            )

        with tf.variable_scope('q_target'):
            # Computes the maximum of elements across dimensions of tensor "self.q_next"
            q_target = normalization_function(
                self.r + self.gamma * tf.reduce_max(
                    input_tensor=inverse_normalization_function(self.q_next),
                    axis=1,
                    name='Qmax_s_'
                )
            )  # shape=(None, )

            # q_target = self.r + self.gamma * tf.reduce_max(
            #     input_tensor=self.q_next,
            #     axis=1,
            #     name='Qmax_s_'
            # )  # shape=(None, )

            # clip
            tf.clip_by_value(q_target, 0, 1)

            # Stops gradient computation - I don't want to improve the target net now
            self.q_target = tf.stop_gradient(
                input=q_target
            )

        with tf.variable_scope('q_eval'):
            # To estimate Q w.r.t. action a
            # Stacks a list of rank-R tensors into one rank-(R+1) tensor.
            a_indices = tf.stack(
                values=[tf.range(tf.shape(self.a)[0], dtype=tf.int32), self.a],
                axis=1
            )

            # Gather slices from params into a Tensor with shape specified by indices
            self.q_eval_wrt_a = tf.gather_nd(
                params=self.q_eval,
                indices=a_indices
            )    # shape=(None, )

        with tf.variable_scope('loss'):
            # Computes the mean of elements across dimensions of the TD error tensor
            self.losses = tf.squared_difference(
                self.q_target,
                self.q_eval_wrt_a,
                name='TD_error'
            )

            self.loss = tf.reduce_mean(
                input_tensor=self.losses
            )

        with tf.variable_scope('train'):
            # Optimizer that implements the RMSProp algorithm
            # self._train_op = tf.train.RMSPropOptimizer(
            #     learning_rate=self.lr
            # ).minimize(
            #     loss=self.loss
            # )

            # Optimizer Parameters from original paper
            self.optimizer = tf.train.RMSPropOptimizer(  # Optimizer that implements the RMSProp algorithm
                learning_rate=self.lr,
                decay=0.99,
                momentum=0.0,
                epsilon=1e-6
            )
            self._train_op = self.optimizer.minimize(
                loss=self.loss,
                global_step=tf.contrib.framework.get_global_step()
                # increment by one after the variables have been updated
            )

        # create and merge all summaries into a single "operation" which we can execute in a session
        self.summaries_op = tf.summary.merge([
            tf.summary.scalar("loSSS", self.loss),
            tf.summary.histogram("loss_hist", self.losses),
            tf.summary.histogram("q_values_hist", self.q_eval),
            tf.summary.scalar("max_q_value", tf.reduce_max(self.q_eval)),
            tf.summary.scalar("min_q_value", tf.reduce_min(self.q_eval)),
            tf.summary.scalar("q_target", self.q_target[0]),  # printing the order of magnitude of the output
            tf.summary.scalar("q_eval_wrt_a", self.q_eval_wrt_a[0])
        ])

    def store_transition(self, s, a, r, s_):
        """
        Store transition SARS' in memory D
        :param s:
        :param a:
        :param r:
        :param s_:
        :return:
        """
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0

        a_id = self.actions_list.index(a)
        # print("action = %s and id = %s" % (a, a_id))
        # r = normalize_reward(r)
        transition = np.hstack((s, [a_id, r], s_))
        # replace the old memory with new memory
        # print("let's store memory. transition = %s" % transition)
        index = self.memory_counter % self.memory_size
        # print(self.memory)
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation, masked_actions_list, greedy_epsilon):
        """

        :param observation:
        :param masked_actions_list:
        :param greedy_epsilon: [float] probability of random choice for epsilon-greedy action selection
        :return:
        """
        possible_actions = [action for action in self.actions_list if action not in masked_actions_list]
        # print("possible_actions = %s" % possible_actions)

        if not possible_actions:
            print("!!!!! No possible_action !!!!!")

        # print("observation to choose action = %s" % observation)

        # to have batch dimension when feed into tf placeholder
        observation = np.asarray(observation)  # convert to numpy
        observation = observation[np.newaxis, :]  # [0, 3] becomes [[0, 3]]

        if np.random.uniform() > greedy_epsilon:
            # forward feed the observation and get q value for every actions
            # Runs operations and evaluates tensors in fetches
            actions_value = self.sess.run(
                fetches=self.q_eval,  # the output of the evaluation net
                feed_dict={self.s: observation}  # A dictionary that maps graph elements to values
            )
            # filter the possible actions:
            possible_id = [self.actions_list.index(a) for a in possible_actions]
            # print("possible_id = %s" % possible_id)
            # chose the max at the output
            # print("actions_value = %s" % actions_value)
            ranked_id_actions_value = actions_value.argsort()[0][::-1]
            id_candidates = [elem for elem in ranked_id_actions_value if elem in possible_id]
            # print("ranked_id_actions_value = %s" % ranked_id_actions_value)
            # print("id_candidates = %s" % id_candidates)
            id_action_to_do = id_candidates[0]
            action_to_do = self.actions_list[id_action_to_do]
        else:
            # action = np.random.randint(0, self.n_actions)
            action_to_do = np.random.choice(possible_actions)
        self.action_taken = action_to_do

        # print("action_to_do: %s" % action_to_do)
        # id_action_to_do = self.actions_list.index(action_to_do)
        # print("id_action_to_do: %s" % id_action_to_do)

        # return id_action_to_do
        return action_to_do

    def learn(self):
        """

        :return:
        """

        if self.saver_dir:
            # Save the current checkpoint
            self.saver.save(tf.get_default_session(), self.saver_dir + 'model.ckpt')

        # check if I should replace the target net with the evaluation net
        # the evaluation net is updated at each step
        # the target net is updated every X  steps

        # Every T steps (a hyper-parameter) the parameters from the Q network are copied to the target network.
        if self.learning_counter % self.replace_target_iter == 0:
            # print("learn step counter = %s" % self.learning_counter)
            # print("replace_target_iter = %s" % self.replace_target_iter)

            self.sess.run(
                fetches=self.target_replace_op  # replace the parameters in the target net
            )
            print('\nTarget_params_replaced\n')

        # sample random mini-batch from memory D
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)

        # the sample:
        batch_memory = self.memory[sample_index, :]
        # print("sample_index = {}".format(sample_index))
        # print("memory = {}".format(self.memory))
        # print("the sample = {}".format(batch_memory))

        # print("s like: {}".format(batch_memory[:, :self.n_features]))
        # print("a like: {}".format(batch_memory[:, self.n_features]))
        # print("r like: {}".format(batch_memory[:, self.n_features + 1]))
        # print("s_ like: {}".format(batch_memory[:, -self.n_features:]))
        feed_dict = {
            self.s: batch_memory[:, :self.n_features],
            self.a: batch_memory[:, self.n_features],
            self.r: batch_memory[:, self.n_features + 1],
            self.s_: batch_memory[:, -self.n_features:]
        }

        # test if same nets before the update. Evaluation on the same state = s
        # q_stat, q_dyn, q_eval_wrt_a = self.sess.run(
        #     [self.q_next, self.q_eval, self.q_eval_wrt_a],
        #     feed_dict={
        #         self.s: batch_memory[:, :self.n_features],
        #         self.s_: batch_memory[:, :self.n_features],
        #         self.a: batch_memory[:, self.n_features],
        #     }
        # )
        # print("q_stat={}".format(q_stat))
        # print("q_dyn={}".format(q_dyn))
        # print("q_eval_wrt_a={}".format(q_eval_wrt_a))
        # print("nets are the same after transfer={}".format(q_dyn == q_stat))

        # print(" ------------ BEFORE update -------------")  # they are identical
        # for pos in range(20):
        #     for vel in range(5):
        #         state = [[pos, vel]]
        #         q_stat, q_dyn = self.sess.run(
        #             [self.q_next, self.q_eval],
        #             feed_dict={
        #                 self.s: state,
        #                 self.s_: state
        #             }
        #         )
        #         print("{} - before update {} ".format(q_dyn == q_stat, state))
        #         print("{} - q_dyn initial".format(q_dyn))

        # perform a gradient descent step on the evaluation net (the only to be trained)
        summaries, global_step_update, _, cost, q_eval_wrt_a, q_target = self.sess.run(
            [self.summaries_op, tf.contrib.framework.get_global_step(), self._train_op, self.loss, self.q_eval_wrt_a,
             self.q_target],
            feed_dict=feed_dict
        )
        # print("q_target at update={}".format(q_target))
        # print("q_eval_wrt_a at update={}".format(q_eval_wrt_a))

        # See if the two nets still say the same - evaluation on the same state = s
        # print(" ------------ AFTER update -------------")
        # for pos in range(20):
        #     for vel in range(5):
        #         state = [[pos, vel]]
        #         q_stat, q_dyn = self.sess.run(
        #             [self.q_next, self.q_eval],
        #             feed_dict={
        #                 self.s: state,
        #                 self.s_: state
        #             }
        #         )
        #         print("{} - after update {} ".format(q_dyn == q_stat, state))
        #         print("{} - q_dyn changed".format(q_dyn))
        #         print("{} - q_stat initial".format(q_stat))

        # print("q_stat changed={}".format(q_stat))
        # print("q_dyn changed={}".format(q_dyn))
        # print("nets are the same after transfer={}".format(q_dyn == q_stat))

        # See the change in TD target and prediction
        # q_stat, q_dyn, q_eval_wrt_a, q_target = self.sess.run(
        #     [self.q_next, self.q_eval, self.q_eval_wrt_a, self.q_target],
        #     feed_dict=feed_dict
        # )
        # print("q_target changed after update={}".format(q_target))
        # print("q_eval_wrt_a changed after update={}".format(q_eval_wrt_a))

        print("loss ={}".format(cost))
        if cost > 1000:
            print("Big loss - state ={}".format(cost))
        print("Big loss at ={}".format(batch_memory[:, :self.n_features]))
        # print("loss ={}".format(global_step_update))

        if self.summary_writer:
            # Adds a Summary protocol buffer to the event file.
            # = write log
            self.summary_writer.add_summary(
                summaries,  # A Summary protocol buffer, optionally serialized as a string
                global_step=global_step_update  # Number. Optional global step value to record with the summary.
            )

        if self.saver_dir:
            # Save the variables to disk.
            save_path = self.saver.save(self.sess, self.saver_dir)
            print("Model saved in path: %s" % save_path)

        self.cost_his.append(cost)
        # print(self.cost_his)

        # increasing epsilon
        # if self.epsilon < self.epsilon_max:
        #     self.epsilon = self.epsilon + self.epsilon_increment
        # else:
        #     self.epsilon = self.epsilon_max

        self.learning_counter += 1
        # print("learn counter = %s" % self.learning_counter)
        # print("epsilon = %s" % self.epsilon)

    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_his)), self.cost_his)
        plt.ylabel('Cost')
        plt.xlabel('training steps')
        plt.show()


if __name__ == '__main__':
    actions_list = ["no_change", "speed_up", "speed_up_up", "slow_down", "slow_down_down"]
    state_features_list = ["position", "velocity"]
    DQN = DeepQNetwork(actions_list, state_features_list)
    print("cmd: >> tensorboard --logdir=C:\tmp\tensorflow_logs\RL")
