"""
This part of code defines the brain of the agent.
- decisions are made here
- the q-table is updated here

Three TD-based model-free control algorithms inherit from the parent Agent. Only the learn() method differs:
-Q-learning
-SARSA
-SARSA-lambda

TD-based = Temporal Difference = all make Sample Back-Up (as opposed to DP = dynamic programming)

On-policy SARSA learns action values relative to the policy it follows
While off-policy Q-Learning does it relative to the greedy policy.
|             | SARSA | Q-learning |
|:-----------:|:-----:|:----------:|
| Choosing a_ |   π   |      π     |
| Updating Q  |   π   |      μ     |

In other words, Q-learning is trying to evaluate π while following another policy μ, so it's an off-policy algorithm.

Q-Learning tends to converge a little slower, but has the capability to continue learning while changing policies.
Also, Q-Learning is not guaranteed to converge when combined with linear approximation.

All are model-free
- Ask yourself this question:
- After learning, can the agent make predictions about next state and reward before it takes each action?
    -- If it can, then it’s a model-based RL algorithm.
    -- If it cannot, it’s a model-free algorithm.

Tabular representation of the discrete (action/state) pairs and the associated q-value:
- I use pd.DataFrame
- one could use collections.defaultdict for structure as well
- np.array should be possible as well

Structure of the object named "q_table":
[id][-------------------------actions---------------------------] [--state features--]
    no_change   speed_up  speed_up_up  slow_down  slow_down_down  position  velocity
0      -4.500  -4.500000       3.1441  -3.434166       -3.177462       0.0       0.0
1      -1.260  -1.260000       9.0490   0.000000        0.000000       2.0       2.0
2       0.396   0.000000       0.0000   0.000000        0.000000       4.0       2.0
3       2.178   0.000000       0.0000   0.000000        0.000000       6.0       2.0

To Do:
- try approximation function (Fixed Sparse Representations), (Incremental Feature Dependency Discovery)
- decay learning rate
- expand the state space - add changing pedestrian position
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from abc import ABC, abstractmethod
import os
from sklearn.linear_model import SGDRegressor
from sklearn.neural_network import MLPRegressor
import sklearn.pipeline
import sklearn.preprocessing
from sklearn.kernel_approximation import RBFSampler
plt.rcParams['figure.figsize'] = [20, 10]


class Agent(ABC):
    def __init__(self, actions_names, state_features, learning_rate=0.9, gamma=0.9, load_q_table=False):
        """
        Parent abstract class (the method learn() is to be defined)

        :param actions_names: [string] list of possible actions
        :param state_features: [string] list of features forming the state
        :param learning_rate: [int between 0 and 1] - Non-constant learning rate must be used?
        :param gamma: [int between 0 and 1] discount factor
        If gamma is closer to one, the agent will consider future rewards with greater weight,
        willing to delay the reward.
        :param load_q_table: [bool] flag to load the q-values DataFrame from file
        """
        # environment information
        self.actions_list = actions_names  # string!
        self.state_features_list = state_features  # string!
        columns_q_table = actions_names + state_features  # string!

        # hyper-parameters
        self.lr = learning_rate  # alpha
        self.gamma = gamma
        # epsilon scheduling is defined in the main

        # structure to store the q-values of the (state/action) pairs
        if load_q_table:
            if self.load_q_table():
                print("Load success")
            else:
                # ToDo: dtype=np.float32 not necessary. Try lower precision
                self.q_table = pd.DataFrame(columns=columns_q_table, dtype=np.float32)
        else:
            self.q_table = pd.DataFrame(columns=columns_q_table, dtype=np.float32)
        # print(self.q_table.columns)

        # settings for plotting
        colours_list = ['green', 'red', 'blue', 'yellow', 'orange']
        self.action_to_color = dict(zip(self.actions_list, colours_list))
        self.size_of_largest_element = 800

    def choose_action(self, observation, masked_actions_list, greedy_epsilon):
        """
        chose an action, following the policy based on the q-table
        with an e_greedy approach and action masking
        :param observation: [list of int] current discrete state
        :param masked_actions_list: [list of string] forbidden actions
        :param greedy_epsilon: [float in 0-1] probability of random choice for epsilon-greedy action selection
        :return: [string] - the name of the action
        """
        # print("state before choosing an action: %s " % observation)
        self.check_state_exist(observation)

        # apply action masking
        possible_actions = [action for action in self.actions_list if action not in masked_actions_list]

        if not possible_actions:
            print("!!!!! WARNING - No possible_action !!!!!")

        # Epsilon-greedy action selection
        if np.random.uniform() > greedy_epsilon:
            # choose best action

            # read the row corresponding to the state
            state_action = self.q_table.loc[
                (self.q_table[self.state_features_list[0]] == observation[0])
                & (self.q_table[self.state_features_list[1]] == observation[1])
                # & (self.q_table[self.state_features_list[2]] == observation[2])
            ]

            # only consider the action names - remove the state information
            state_action = state_action.filter(self.actions_list, axis=1)

            # shuffle - if different actions have equal q-values, chose randomly, not the first one
            state_action = state_action.reindex(np.random.permutation(state_action.index))

            # restrict to allowed actions
            state_action = state_action.filter(items=possible_actions)
            # print("state_action 3/3 : %s" % state_action)

            # make decision
            if state_action.empty:
                action = random.choice(possible_actions)
                print('random action sampled among allowed actions')
            else:
                action = state_action.idxmax(axis=1)
                # Return index of first occurrence of maximum over requested axis (with shuffle)

            # get first element of the pandas series
            action_to_do = action.iloc[0]
            # print("\tBEST action = %s " % action_to_do)
        
        else:
            # choose random action
            action_to_do = np.random.choice(possible_actions)
            # print("\t-- RANDOM action= %s " % action_to_do)

        return action_to_do

    @abstractmethod
    def learn(self, *args):
        """
        Update the agent's knowledge, using the most recently sampled tuple
        This method is implemented in each children agent
        """
        # raise NotImplementedError('subclasses must override learn()!')
        pass

    def check_state_exist(self, state):
        """
        read if the state has already be encountered
        if not, add it to the table and initialize its q-value
        with collections.defaultdict or np.array, this would have not be required
        :param state: [list of int] current discrete state
        :return: -
        """
        # try to find the index of the state
        state_id_list_previous_state = self.q_table.index[(self.q_table[self.state_features_list[0]] == state[0]) &
                                                          (self.q_table[self.state_features_list[1]] ==
                                                           state[1])].tolist()

        if not state_id_list_previous_state:
            # ToDo: is zero-value initialization relevant?
            # append new state to q table: Q(a,s)=0 for each action a
            new_data = np.concatenate((np.array(len(self.actions_list)*[0]), np.array(state)), axis=0)
            # print("new_data to add %s" % new_data)
            new_row = pd.Series(new_data, index=self.q_table.columns)
            self.q_table = self.q_table.append(new_row, ignore_index=True)

    def get_id_row_state(self, s):
        """

        :param s: [list of int] state
        :return: [int] id of the row corresponding to the state in self.q_table
        """
        # get id of the row of the previous state (= id of the previous state)
        id_list_state = self.q_table.index[(self.q_table[self.state_features_list[0]] == s[0]) &
                                           (self.q_table[self.state_features_list[1]] == s[1])].tolist()
        id_row_state = id_list_state[0]
        # row = self.q_table.loc[id_row_state]
        # print("row = \n{}".format(row))
        # filtered_row = row.filter(self.actions_list)
        # print("filtered_row = \n{}".format(filtered_row))
        return id_row_state

    def load_q_table(self, file_path="q_table"):
        """
        open_model
        working with h5, csv or pickle format
        :return: -
        """
        try:
            # from pickle
            grand_grand_parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            results_dir = os.path.abspath(grand_grand_parent_dir + "/results/simple_road/" + file_path + '.pkl')
            self.q_table = pd.read_pickle(results_dir)
            return True

        except Exception as e:
            print(e)
        return False

    def save_q_table(self, save_directory):
        """
        at the end, save the q-table
        several extensions are possible:
        see for comparison: https://stackoverflow.com/questions/17098654/how-to-store-a-dataframe-using-pandas
        :return: -
        """
        filename = "q_table"
        # sort series according to the position
        self.q_table = self.q_table.sort_values(by=[self.state_features_list[0]])
        try:
            # to pickle
            self.q_table.to_pickle(save_directory + filename + ".pkl")
            print("Saved as " + filename + ".pkl")

        except Exception as e:
            print(e)

    def print_q_table(self):
        """
        at the end, display the q-table
        One could also use .head()
        :return: -
        """
        # sort series according to the position
        self.q_table = self.q_table.sort_values(by=[self.state_features_list[0]])
        print(self.q_table.head())
        # print(self.q_table.to_string())

    def plot_q_table(self, folder):
        """
        plot the q(a,s) for each s

        # previously: Only plot the values attached actions - the state features serve as abscissa
        # Issue: we have 2D-space. So hard to represent all on the x-abscissa
        # data_frame_to_plot = self.q_table.filter(self.actions_list, axis=1)
        # print(data_frame_to_plot.to_string())
        # data_frame_to_plot.plot.bar()
        # plt.show()
        :return:
        """
        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        # not to overlap scatters
        shift = 0.2

        # to scale for the size of markers
        # The matrix Q can then be normalized (i.e.; converted to percentage)
        # by dividing all non-zero entries by the highest number (choice between max and abs(min))
        min_value = min(self.q_table[self.actions_list].min(axis=0))
        max_value = max(self.q_table[self.actions_list].max(axis=0))
        # mean_value = (self.q_table[self.actions_list].max(axis=0)).mean
        # self.q_table = (self.q_table - mean_value) / (max_value - min_value)

        scale_factor = self.size_of_largest_element / max(max_value, abs(min_value))
        # print(100*(self.q_table['speed_up']-min_value)/scale_factor)

        # Not very efficient, but it works:
        # not printing the non visited states (size = value = 0)
        # distinguishing positive and negative values (marker type)
        i = 0
        for action in self.actions_list:
            colour_for_action = self.action_to_color[action]
            colour_for_action_neg = colour_for_action
            markers = ['P' if i > 0 else 's' for i in self.q_table[action]]
            sizes = [scale_factor * (abs(i)) for i in self.q_table[action]]
            colours = [colour_for_action if i > 0 else colour_for_action_neg for i in self.q_table[action]]
            for x, y, m, s, c in zip(self.q_table[self.state_features_list[0]],
                                     self.q_table[self.state_features_list[1]], markers, sizes, colours):
                ax1.scatter(x, y + i*shift, alpha=0.8, c=c, marker=m, s=s)
            i += 1

        # custom labels
        labels_list = []
        for action in self.actions_list:
            label = patches.Patch(color=self.action_to_color[action], label=action)
            labels_list.append(label)
        plt.legend(handles=labels_list)

        # plot decoration
        plt.title('Normalized Q(s,a) - distinguishing positive and negative values with marker type')
        plt.xlabel(self.state_features_list[0])
        plt.ylabel(self.state_features_list[1])
        plt.xticks(np.arange(min(self.q_table[self.state_features_list[0]]),
                             max(self.q_table[self.state_features_list[0]]) + 1, 1.0))
        plt.grid(True, alpha=0.2)
        ax1.set_facecolor('silver')
        plt.savefig(folder + "plot_q_table.png")
        plt.show()

    def plot_optimal_actions_at_each_position(self, folder):
        """
        plotting the best action to take for each state
        also quantify the relative confidence
        :return: -
        """
        # scaling
        min_value = min(self.q_table[self.actions_list].min(axis=0))
        max_value = max(self.q_table[self.actions_list].max(axis=0))
        scale_factor = self.size_of_largest_element / max(max_value, abs(min_value))

        # look for the best action for each state
        fig = plt.figure()
        ax2 = fig.add_subplot(111)
        for index, row in self.q_table.iterrows():
            action_value = row.filter(self.actions_list, axis=0)
            action = action_value.idxmax()
            value = action_value.max()
            x = row[self.state_features_list[0]]
            y = row[self.state_features_list[1]]
            c = self.action_to_color[action]

            if value > 0:
                m = 'P'
            else:
                m = 's'
            s = scale_factor * abs(value)
            ax2.scatter(x, y, alpha=0.8, c=c, marker=m, s=s)

        # custom labels
        labels_list = []
        for action in self.actions_list:
            label = patches.Patch(color=self.action_to_color[action], label=action)
            labels_list.append(label)
        plt.legend(handles=labels_list)

        # plot decoration
        plt.title('Normalized max[Q(s,a)][over a] - Optimal actions - randomly selected if equal')
        plt.xlabel(self.state_features_list[0])
        plt.ylabel(self.state_features_list[1])
        plt.xticks(np.arange(min(self.q_table[self.state_features_list[0]]),
                             max(self.q_table[self.state_features_list[0]]) + 1, 1.0))
        plt.grid(True, alpha=0.2)
        ax2.set_facecolor('silver')
        plt.savefig(folder + "plot_optimal_actions_at_each_position.png")

        plt.show()


# on-policy: Unlike Q learning which is a offline updating method, Sarsa is updating while in the current trajectory
# SARSA can only learn from itself (from the experience and transition it met in the past)
class SarsaTable(Agent):
    def __init__(self, actions, state, learning_rate=0.9, reward_decay=0.9, load_q_table=False):
        super(SarsaTable, self).__init__(actions, state, learning_rate, reward_decay, load_q_table)

    def learn(self, s, a, r, s_, a_, termination_flag):
        """
        update the q-table based on the observed experience S.A.R.S.A
        :param s: previous state (list of int)
        :param a: action (str)
        :param r: reward (int)
        :param s_: new state (list of int)
        :param termination_flag: (boolean)
        :param a_: new action (str)
        :return: -
        """
        self.check_state_exist(s_)

        # get id of the row of the previous state
        id_row_previous_state = self.get_id_row_state(s)

        # get id of the row of the next state
        id_row_next_state = self.get_id_row_state(s_)

        # get q-value of the pair (previous_state, action)
        q_predict = self.q_table.loc[id_row_previous_state, a]

        # Check if new state is terminal
        if termination_flag:
            # next state is terminal
            # goal state has no value
            q_target = r
        else:
            # next state is not terminal
            # consider the value of the next state with the action a_
            # using the actual action a_ to evaluate Q(s_, a_) - SARSA is therefore said "on-policy"
            row = self.q_table.loc[id_row_next_state]
            filtered_row = row.filter(self.actions_list)
            q_max = max(filtered_row)
            q_expected = self.q_table.loc[id_row_next_state, a_]
            q_target = r + self.gamma * q_expected
            print("q_expected/q_max = {} q_expected = {} q_max = {}".format(q_expected/q_max, q_expected, q_max))

        # update q-value - Delta is the TD-error
        self.q_table.loc[id_row_previous_state, a] += self.lr * (q_target - q_predict)


# to compute the q_predict, make the average of q-values based on probabilities of each action
class ExpectedSarsa(Agent):
    def __init__(self, actions, state, learning_rate=0.9, reward_decay=0.9, load_q_table=False):
        super(ExpectedSarsa, self).__init__(actions, state, learning_rate, reward_decay, load_q_table)

    def learn(self, s, a, r, s_, termination_flag, greedy_epsilon):
        """
        update the q-table based on the observed experience S.A.R.S.A
        :param s: previous state (list of int)
        :param a: action (str)
        :param r: reward (int)
        :param s_: new state (list of int)
        :param termination_flag: (boolean)
        :param greedy_epsilon: [float]
        :return: -
        """
        self.check_state_exist(s_)

        # get id of the row of the previous state
        id_row_previous_state = self.get_id_row_state(s)

        # get id of the row of the next state
        id_row_next_state = self.get_id_row_state(s_)

        # get q-value of the tuple (previous_state, action)
        q_predict = self.q_table.loc[id_row_previous_state, a]

        # Check if new state is terminal
        if termination_flag:
            # next state is terminal
            # goal state has no value
            q_target = r
            # Trying to reduce chance of random action as we train the model.

        else:
            row = self.q_table.loc[id_row_next_state]
            filtered_row = row.filter(self.actions_list)
            q_max = max(filtered_row)
            q_expected = 0
            for a in self.actions_list:
                q_expected += filtered_row[a] * greedy_epsilon * 1 / len(self.actions_list)
            q_expected += (1 - greedy_epsilon) * q_max
            print("q_expected/q_max = {} q_expected = {} q_max = {}".format(q_expected/q_max, q_expected, q_max))

            q_target = r + self.gamma * q_expected

        # update q-value following Q-learning - Delta is the TD-error
        self.q_table.loc[id_row_previous_state, a] += self.lr * (q_target - q_predict)


# off-policy. Q-learning = sarsa_max
class QLearningTable(Agent):
    def __init__(self, actions, state, learning_rate=0.9, reward_decay=0.9, load_q_table=False):
        super(QLearningTable, self).__init__(actions, state, learning_rate, reward_decay, load_q_table)

    def learn(self, s, a, r, s_, termination_flag):
        """
        update the q-table based on the observed experience S.A.R.S.
        :param s: previous state (list of int)
        :param a: action (str)
        :param r: reward (int)
        :param s_: new state (list of int)
        :param termination_flag: (boolean)
        :return: -
        """
        self.check_state_exist(s_)

        # get id of the row of the previous state
        id_row_previous_state = self.get_id_row_state(s)

        # get id of the row of the next state
        id_row_next_state = self.get_id_row_state(s_)

        # get q-value of the tuple (previous_state, action)
        q_predict = self.q_table.loc[id_row_previous_state, a]

        # Check if new state is terminal
        if termination_flag:
            # next state is terminal
            # goal state has no value
            q_target = r
            # Trying to reduce chance of random action as we train the model.

        else:
            # next state is not terminal
            # consider the best value of the next state. Q-learning = sarsa_max
            # using max to evaluate Q(s_, a_) - Q-learning is therefore said "off-policy"
            row = self.q_table.loc[id_row_next_state]
            filtered_row = row.filter(self.actions_list)
            # print(s)
            # print("filtered_row = \n{}".format(filtered_row))
            # print("max(filtered_row) = \n{}".format(max(filtered_row)))
            q_expected = max(filtered_row)
            q_target = r + self.gamma * q_expected
            # q_target = r + self.gamma * self.q_table.loc[id_row_next_state, :].max()

        # update q-value following Q-learning - Delta is the TD-error
        self.q_table.loc[id_row_previous_state, a] += self.lr * (q_target - q_predict)


# Sarsa Lambda can learn for
# - 1 step (Sarsa) (lambda=0)
# - All the episode (Monte Carlo) (lambda=1)
# - in between (Lambda in [0,1])
# Idea is to update and give reward to all the steps that contribute to the end return
class SarsaLambdaTable(Agent):
    def __init__(self, actions, state, learning_rate=0.9, reward_decay=0.9, load_q_table=False,
                 trace_decay=0.9):
        super(SarsaLambdaTable, self).__init__(actions, state, learning_rate, reward_decay, load_q_table)

        # !!!!!!!!!
        # backward view, eligibility trace.
        self.lambda_ = trace_decay
        # same dimension as the Q-table: it counts how many times the state has been visited
        self.eligibility_trace = self.q_table.copy()

    def reset_eligibility_trace(self):
        # self.eligibility_trace *= 0
        self.eligibility_trace[self.actions_list] = 0.0
        # print(self.eligibility_trace)

    def check_state_exist(self, state):
        """
        read if the state has already be encountered
        if not, add it to the table
        update the eligibility_trace too
        :param state:
        :return: -
        """
        # try to find the index of the state
        state_id_list_previous_state = self.q_table.index[(self.q_table[self.state_features_list[0]] == state[0]) &
                                                          (self.q_table[self.state_features_list[1]] ==
                                                           state[1])].tolist()

        if not state_id_list_previous_state:
            # append new state to q table: Q(a,s)=0 for each action a
            new_data = np.concatenate((np.array(len(self.actions_list) * [0]), np.array(state)), axis=0)
            # print("new_data to add %s" % new_data)
            new_row = pd.Series(new_data, index=self.q_table.columns)
            self.q_table = self.q_table.append(new_row, ignore_index=True)

            # also update eligibility trace
            self.eligibility_trace = self.eligibility_trace.append(new_row, ignore_index=True)

    def learn(self, s, a, r, s_, a_, termination_flag):
        """
        update the q-table based on the observed experience S.A.R.S.A
        update the eligibility_trace too
        :param s: previous state (list of int)
        :param a: action (str)
        :param r: reward (int)
        :param s_: new state (list of int)
        :param termination_flag: (boolean)
        :param a_: new action (str)
        :return: -
        """
        self.check_state_exist(s_)

        # get id of the row of the previous state
        id_row_previous_state = self.get_id_row_state(s)

        # get id of the row of the next state
        id_row_next_state = self.get_id_row_state(s_)

        # get q-value of the tuple (previous_state, action)
        q_predict = self.q_table.loc[id_row_previous_state, a]

        # Check if new state is terminal
        if termination_flag:
            # next state is terminal
            # goal state has no value
            q_target = r
        else:
            # next state is not terminal
            # consider the value of the next state with the action a_
            # using the actual action a_ to evaluate Q(s_, a_) - SARSA is therefore said "on-policy"
            q_target = r + self.gamma * self.q_table.loc[id_row_next_state, a_]

        # TD-error (is it the so called?)
        error = q_target - q_predict

        # increasing the importance factor for the visited state-action pair. Two methods:
        # Method 1: accumulating trace (not quite stable)
        # self.eligibility_trace.loc[id_row_previous_state, a] += 1

        # Method 2: replacing trace (normalization) - if I visit a state more than once, it still stays at 1, not more
        self.eligibility_trace.loc[id_row_previous_state, :] *= 0
        self.eligibility_trace.loc[id_row_previous_state, a] = 1

        # Q update - most state will not be considered
        # The importance factor (=eligibility_trace) says how important is to travel by this state to get the return
        self.q_table[self.actions_list] += self.lr * error * self.eligibility_trace[self.actions_list]

        # decay eligibility trace after update (before the next step)
        self.eligibility_trace *= self.gamma*self.lambda_


# off-policy q-learning with Q-table Approximation
class QLearningApproximation(Agent):
    """
    Not finish!
    ToDo: complete it
    SGD is sensitive to feature scaling just like batch GD. Hence using StandardScaler.
    Scaling on the state before applying a featurizer
    Then two possible estimators:
     - linear regression with SGD
      - MLP
    """
    def __init__(self, actions, state, learning_rate=0.9, reward_decay=0.9, load_q_table=False,
                 regressor="linearSGD"):
        super(QLearningApproximation, self).__init__(actions, state, learning_rate, reward_decay, load_q_table)

        # hand-crafted: list all possible (pos, vel) pairs
        self.sample_states = [[position, velocity] for position in range(20) for velocity in range(6)]
        # print("sample_states = %s" % self.sample_states)

        self.with_feature = True

        if self.with_feature:
            # Feature Preprocessing: Normalize to zero mean and unit variance
            # We use a few samples from the observation space to do this
            observation_examples = np.array([self.sample_state() for _ in range(10000)])

            # Transform the data such that its distribution will have a mean value 0 and standard deviation of 1
            #  __init__ of the scaler
            scaler = sklearn.preprocessing.StandardScaler()

            # Compute the mean and std to be used for later scaling
            # Train the transformer object so it knows what means and variances to use
            scaler.fit(observation_examples)
            # print("mean = %s (should be [9.5, 2.5])" % scaler.mean_)
            # print("std = %s (should be [33.25, 2.9166])" % scaler.var_)

            # Manually set scaling parameters
            scaler.mean_ = [9.5, 2.5]
            scaler.var_ = [399/12, 35/12]  # variance of [0, 1, ..., N] = (N^2 - 1)/12
            # print("mean = %s (should be [9.5, 2.5])" % scaler.mean_)
            # print("std = %s (should be [33.25, 2.9166])" % scaler.var_)

            # Used to convert a state to a featurized representation - size = features!!
            # We use RBF kernels with different variances to cover different parts of the space
            # FeatureUnion = Concatenates results of multiple transformer objects (RBFsampler here)
            # -> useful to combine several feature extraction mechanisms into a single transformer

            # The RBFSampler constructs an approximate mapping for the radial basis function kernel,
            # also known as Random Kitchen Sinks
            # Approximates feature map of an RBF kernel by Monte Carlo approximation of its Fourier transform
            # Pipeline = to build a composite estimator, as a chain of transforms and estimators
            featurizer = sklearn.pipeline.FeatureUnion([
                # List of transformer objects to be applied to the data: rbf1, rbf2, rbf3, rbf4
                ("rbf1", RBFSampler(  # RBFSampler is a transformer
                    gamma=5.0,  # Parameter of RBF kernel: exp(-gamma * x^2)
                    n_components=100)  # Number of Monte Carlo samples per original feature.
                    # Equals the dimensionality of the computed feature space.
                 ),
                ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
                ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
                ("rbf4", RBFSampler(gamma=0.5, n_components=100))]
            )

            # Perform standardization by centering and scaling
            scaled_observations = scaler.transform(observation_examples)
            # Used sampled observations to  fit transformers
            featurizer.fit(scaled_observations)

            self.scaler = scaler
            self.featurizer = featurizer

        # list of models (one per action)
        self.models = []

        for _ in range(len(actions)):
            if regressor == "linearSGD":
                # instanciate a linear regression with SGD
                clf = SGDRegressor(
                    loss='squared_loss',
                    penalty='l2',
                    learning_rate="constant"
                )

            elif regressor == "MLP":
                # instanciate a Multi-layer Perceptron regressor
                clf = MLPRegressor(
                    alpha=0.001,  # L2 penalty (regularization term) parameter
                    hidden_layer_sizes=(10, 10),  # The ith element represents the nb of neurons in the ith hidden layer
                    max_iter=50000,
                    activation='relu',
                    # verbose='True',  # Whether to print progress messages to stdout
                    learning_rate='adaptive'
                )

            # bit of hack to initialize each model
            sample_state = [0, 1]  # this is not really random
            if self.with_feature:
                features = [self.featurize(sample_state)]
            else:
                features = [sample_state]

            target = [0]
            # clf.n_iter = np.ceil(10 ** 6 / len(features))  # rule of thumb for the number of passes over the training
            # set to get convergence

            # First operation of fitting linear model to the action "Non-change" with SGD
            # Actually do the fit AKA execute the gradient descent by calling a method of our SGDRegressor object.
            clf.partial_fit(features, target)

            self.models.append(clf)

    def sample_state(self):
        """
        sample among all possible states of the environment
        :return: on sample
        """
        sample = random.choice(self.sample_states)
        return sample

    def featurize(self, state):
        """
        scale the state and convert it to features
        :param state: state ([pos, velocity]) to be featurized
        :return: features representation
        """
        scaled = self.scaler.transform([state])
        featurized = self.featurizer.transform(scaled)
        return featurized[0]

    def predict(self, state):
        """
        For each action, compute the predicted q(a,s) values for the given state
        :param state: the current state
        :return: list of q(state) for each action
        """

        if self.with_feature:
            features = [self.featurize(state)]
        else:
            features = [state]

        # print("state for prediction: %s " % state)
        # print("predictions for the state %s = %s " % (state, [model.predict(features)[0] for model in self.models]))
        return [model.predict(features)[0] for model in self.models]

    def update(self, state, action, target):
        """
        Updates the estimator parameters for a given state and action towards the target q_target
        (One model is attached to an action and gives q(s) for each possible state s)
        Treat RL as supervised learning problem:
            with the MC- or TD-target as the label and the current state/action as the input.
        :param state: previous state (no matter of the next_state - it is included in the target)
        :param action: the action that was taken
        :param target: the q_target (it includes the reward and the next state)
        :return: -
        """
        if self.with_feature:
            features = [self.featurize(state)]
        else:
            features = [state]
        target = [target]
        # The 'models[action]' gives Q(action, s) for each state s
        # This is q_predict
        # Shape input: (n_samples, n_features)
        # Shape output: (n_samples)
        self.models[action].partial_fit(features, target)

    def learn(self, s, a, r, s_, termination_flag):
        """
        update the q-table based on the observed experience S.A.R.S.
        :param s: previous state (list of int)
        :param a: action (str)
        :param r: reward (int)
        :param s_: new state (list of int)
        :param termination_flag: (boolean)
        :return: -
        """

        # id_a = self.actions_list.index(a)
        # q_predict_approximation = self.predict(s)
        # print("q_predict_approximation = %s" % q_predict_approximation)
        # print("q_predict_approximation for action '%s' = %s" % (a, q_predict_approximation[id_a]))

        # Check if new state is terminal
        if termination_flag:
            # next state is terminal
            # goal state has no value
            q_target_approximation = r

            # Trying to reduce chance of random action as we train the model.
            # Updated epsilon
            # self.epsilon = self.epsilon / (self.epsilon + 0.05)
            # print("updated epsilon = %s" % self.epsilon)

        else:
            # next state is not terminal
            # consider the best value of the next state
            # using max to evaluate Q(s_, a_) - Q-learning is therefore said "off-policy"
            q_target_approximation = r + self.gamma * np.max(self.predict(s_))

        # update q-value following Q-learning - The TD-error is computed in the update() method
        id_a = self.actions_list.index(a)
        self.update(s, id_a, q_target_approximation)

    def choose_action(self, observation, masked_actions_list, greedy_epsilon):
        """
        chose an action based on the policy of the q-table
        with an e_greedy approach
        :param observation: current state
        :param masked_actions_list: forbidden actions
        :param greedy_epsilon: probability of random choice for epsilon-greedy action selection
        :return:
        """
        # print("state before choosing an action: %s " % observation)

        # apply action masking
        possible_actions = [action for action in self.actions_list if action not in masked_actions_list]
        if not possible_actions:
            print("!!!!! No possible_action !!!!!")

        # Epsilon-greedy action selection
        if np.random.uniform() > greedy_epsilon:
            # choose best action

            actions_value = self.predict(observation)
            ranked_id_actions_value = np.argsort(actions_value)[::-1]
            possible_id = [self.actions_list.index(a) for a in possible_actions]
            id_candidates = [index_action for index_action in ranked_id_actions_value if index_action in possible_id]
            id_action_to_do = id_candidates[0]  # should add random in case of equal results?
            action_to_do_approximation = self.actions_list[id_action_to_do]

            # print("actions_value for that state = %s" % actions_value)
            # print("np.argsort(actions_value) = %s" % np.argsort(actions_value))
            # print("ranked_id_actions_value = %s" % ranked_id_actions_value)
            # print("possible_actions = %s" % possible_actions)
            # print("possible_id = %s" % possible_id)
            # print("id_candidates = %s" % id_candidates)
            # print("id_action_to_do = %s" % id_action_to_do)
            # print("action_to_do_approximation = %s" % action_to_do_approximation)
            # print("\tBEST action = %s " % action_to_do)

        else:
            # choose random action
            action_to_do_approximation = np.random.choice(possible_actions)
            # print("\t-- RANDOM action= %s " % action_to_do)

        return action_to_do_approximation

    def create_q_table(self):
        """
        use the function approximation to fill the table
        :return: -
        """
        for position in range(20):
            for velocity in range(6):
                q_values_list = self.predict([position, velocity])
                state_list = [position, velocity]
                new_data = np.concatenate((np.array(q_values_list), np.array(state_list)), axis=0)
                # print("new_data to add %s" % new_data)
                new_row = pd.Series(new_data, index=self.q_table.columns)
                self.q_table = self.q_table.append(new_row, ignore_index=True)
        print(self.q_table.head())
