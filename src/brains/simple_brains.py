"""
This part of code defines the brain of the agent.
- decisions are made here
- the q-table is updated here

The parent Agent is a abstract class:
- learn() method is a virtual method (to be defined)
- q_table is store
- Tabular representation of the discrete (action/state) pairs and the associated q-value:

Inherited classes are:
-- One Monte-Carlo control algorithms
    - q_table is a defaultdict
-- Four TD-based model-free control algorithms
    - q_table is a pandas DataFrame
    - Only the learn() method differs:
        - Q-learning (= max-SARSA)
        - SARSA
        - SARSA-lambda
        - expected-SARSA
-- One model-based Monte-Carlo Dynamic Programming method

Note about terminology:
 - TD-based = Temporal Difference = all make Sample Back-Up (as opposed to DP = dynamic programming)

 - On-policy SARSA learns action values relative to the policy it follows
While off-policy Q-Learning does it relative to the greedy policy.
|             | SARSA | Q-learning |
|:-----------:|:-----:|:----------:|
| Choosing a_ |   π   |      π     |
| Updating Q  |   π   |      μ     |

 - In other words, Q-learning is trying to evaluate π while following another policy μ, so it's an off-policy algorithm.
    - Q-Learning tends to converge a little slower, but has the capability to continue learning while changing policies.
    - Also, Q-Learning is not guaranteed to converge when combined with linear approximation.

 - All are model-free
    - Ask yourself this question:
    - After learning, can the agent make predictions about next state and reward before it takes each action?
        -- If it can, then it’s a model-based RL algorithm.
        -- If it cannot, it’s a model-free algorithm.

Structure of the object named "q_table":
[id][-------------------------actions---------------------------] [--state features--]
    no_change   speed_up  speed_up_up  slow_down  slow_down_down  position  velocity
0      -4.500  -4.500000       3.1441  -3.434166       -3.177462       0.0       0.0
1      -1.260  -1.260000       9.0490   0.000000        0.000000       2.0       2.0
2       0.396   0.000000       0.0000   0.000000        0.000000       4.0       2.0
3       2.178   0.000000       0.0000   0.000000        0.000000       6.0       2.0

# ToDo:
- try approximation function (Fixed Sparse Representations), (Incremental Feature Dependency Discovery)
- decay learning rate
- expand the state space - add changing pedestrian position
- make the initial state random
"""

import numpy as np
import time
import pickle
from copy import copy
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import random
from abc import ABC, abstractmethod
import os
from collections import defaultdict
plt.rcParams['figure.figsize'] = [20, 10]


class Agent(ABC):
    def __init__(self, actions_names, state_features, load_q_table=False):
        """
        Parent abstract class (the method learn() is to be defined)

        :param actions_names: [string] list of possible actions
        :param state_features: [string] list of features forming the state
        :param load_q_table: [bool] flag to load the q-values DataFrame from file
        """
        # environment information
        self.actions_list = actions_names  # string!
        self.state_features_list = state_features  # string!
        self.columns_q_table = actions_names + state_features  # string!

        # structure to store the q-values of the (state/action) pairs
        self.q_table = None
        if load_q_table:
            if self.load_q_table():
                print("Load success")
            else:
                self.reset_q_table()
        else:
            self.reset_q_table()
        # print(self.q_table.columns)

        # settings for plotting
        colours_list = ['green', 'red', 'blue', 'yellow', 'orange']
        self.action_to_color = dict(zip(self.actions_list, colours_list))
        self.size_of_largest_element = 800

    def reset_q_table(self):
        self.q_table = pd.DataFrame(columns=self.columns_q_table, dtype=np.float32)

        print("reset_q_table - self.q_table has shape = {}".format(self.q_table.shape))

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

    def compare_reference_value(self):
        """
        we know the value of the last-but one state at convergence: Q(s,a)=R(s,a).
        since if termination_flag: q_target = r (# goal state has no value)
        :return: the value of a given (state, action) pair
        """
        state = [16, 3]
        action_id = 0  # "no change"
        self.check_state_exist(state)
        id_row_previous_state = self.get_id_row_state(state)
        res = self.q_table.loc[id_row_previous_state, self.actions_list[action_id]]
        # should be +40
        print("reference_value = {}".format(res))
        return res

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
            # ToDo: is zero-value initialization relevant? It seems so, yes
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

    def load_q_table(self, weight_file=None):
        """
        open_model
        working with h5, csv or pickle format
        :return: -
        """
        try:
            # from pickle
            if weight_file is None:
                grand_grand_parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
                results_dir = os.path.abspath(grand_grand_parent_dir + "/results/simple_road/" + "q_table" + '.pkl')
                self.q_table = pd.read_pickle(results_dir)
            else:
                self.q_table = pd.read_pickle(weight_file)
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
        # print(self.q_table.head())
        print(self.q_table.to_string())

    def plot_q_table(self, folder, display_flag):
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
        if display_flag:
            plt.show()

    def plot_optimal_actions_at_each_position(self, folder, display_flag):
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
        if display_flag:
            plt.show()


# on-policy: Unlike Q learning which is a offline updating method, Sarsa is updating while in the current trajectory
# SARSA can only learn from itself (from the experience and transition it met in the past)
class SarsaTable(Agent):
    def __init__(self, actions, state, load_q_table=False):
        super(SarsaTable, self).__init__(actions, state, load_q_table)

    def learn(self, s, a, r, s_, a_, termination_flag, gamma, learning_rate):
        """
        update the q-table based on the observed experience S.A.R.S.A
            using the actual action a_ to evaluate Q(s_, a_) - SARSA is therefore said "on-policy"
            q_expected = Q(s_, a_)
        :param s: previous state (list of int)
        :param a: action (str)
        :param r: reward (int)
        :param s_: new state (list of int)
        :param termination_flag: (boolean)
        :param a_: new action (str)
        :param gamma: [float between 0 and 1] discount factor
        :param learning_rate: [float between 0 and 1] - learning rate
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
            q_expected = self.q_table.loc[id_row_next_state, a_]
            q_target = r + gamma * q_expected

        # update q-value - Delta is the TD-error
        self.q_table.loc[id_row_previous_state, a] += learning_rate * (q_target - q_predict)


# to compute the q_predict, make the average of q-values based on probabilities of each action
class ExpectedSarsa(Agent):
    def __init__(self, actions, state, load_q_table=False):
        super(ExpectedSarsa, self).__init__(actions, state, load_q_table)

    def learn(self, s, a, r, s_, termination_flag, greedy_epsilon, gamma, learning_rate):
        """
        update the q-table based on the observed experience S.A.R.S.A
            Use the expected q_value of the next state for q_expected (used to build q_target)
            Expectation is w.r.t. e-greedy-policy!
            e-greedy-policy is to take action = argmax(Q) with probability = 1-e
            and a random choice with prob = e
            hence q_expected = q_mean * e + q_max * (1-e)

        :param s: previous state (list of int)
        :param a: action (str)
        :param r: reward (int)
        :param s_: new state (list of int)
        :param termination_flag: (boolean)
        :param greedy_epsilon: [float]
        :param gamma: [float between 0 and 1] discount factor
        :param learning_rate: [float between 0 and 1] - learning rate
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
            # next state is terminal - goal state has no value
            q_target = r
            # Trying to reduce chance of random action as we train the model

        else:
            # next state is not terminal
            row = self.q_table.loc[id_row_next_state]
            filtered_row = row.filter(self.actions_list)
            # print("filtered_row = \n{}".format(filtered_row))
            # print("max(filtered_row) = \n{}".format(max(filtered_row)))
            # print("sum(filtered_row) = \n{}".format(sum(filtered_row)))

            q_max = max(filtered_row)
            # print("q_max = \n{}".format(q_max))

            q_mean = 0
            if len(filtered_row):
                q_mean = sum(filtered_row)/len(filtered_row)
            # print("q_mean = \n{}".format(q_mean))

            q_expected = (1 - greedy_epsilon) * q_max + greedy_epsilon * q_mean
            # print("q_expected = \n{}".format(q_expected))

            q_target = r + gamma * q_expected

        # update q-value following Q-learning - Delta is the TD-error
        self.q_table.loc[id_row_previous_state, a] += learning_rate * (q_target - q_predict)


# off-policy. Q-learning = sarsa_max
class QLearningTable(Agent):
    def __init__(self, actions, state, load_q_table=False):
        super(QLearningTable, self).__init__(actions, state, load_q_table)

    def learn(self, s, a, r, s_, termination_flag, gamma, learning_rate):
        """
        update the q-table based on the observed experience S.A.R.S.
        :param s: previous state (list of int)
        :param a: action (str)
        :param r: reward (int)
        :param s_: new state (list of int)
        :param termination_flag: (boolean)
        :param gamma: [float between 0 and 1] discount factor
        :param learning_rate: [float between 0 and 1] - learning rate
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
            q_target = r + gamma * q_expected
            # q_target = r + gamma * self.q_table.loc[id_row_next_state, :].max()

        # update q-value following Q-learning - Delta is the TD-error
        self.q_table.loc[id_row_previous_state, a] += learning_rate * (q_target - q_predict)


# Sarsa Lambda can learn for
# - 1 step (Sarsa) (lambda=0)
# - All the episode (Monte Carlo) (lambda=1)
# - in between (Lambda in [0,1])
# Idea is to update and give reward to all the steps that contribute to the end return
class SarsaLambdaTable(Agent):
    def __init__(self, actions, state, load_q_table=False,
                 trace_decay=0.9):
        super(SarsaLambdaTable, self).__init__(actions, state, load_q_table)

        # !!!!!!!!!
        # backward view, eligibility trace.
        self.lambda_trace_decay = trace_decay
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
        # try to find the index of the state - same as for the parent Class
        state_id_list_previous_state = self.q_table.index[(self.q_table[self.state_features_list[0]] == state[0]) &
                                                          (self.q_table[self.state_features_list[1]] ==
                                                           state[1])].tolist()

        if not state_id_list_previous_state:
            # append new state to q table: Q(a,s)=0 for each action a
            new_data = np.concatenate((np.array(len(self.actions_list) * [0]), np.array(state)), axis=0)
            # print("new_data to add %s" % new_data)
            new_row = pd.Series(new_data, index=self.q_table.columns)

            # add new row in q_table
            self.q_table = self.q_table.append(new_row, ignore_index=True)

            # also add it to the eligibility trace
            self.eligibility_trace = self.eligibility_trace.append(new_row, ignore_index=True)

    def learn(self, s, a, r, s_, a_, termination_flag, gamma, learning_rate):
        """
        update the q-table based on the observed experience S.A.R.S.A
        update the eligibility_trace too
        :param s: previous state (list of int)
        :param a: action (str)
        :param r: reward (int)
        :param s_: new state (list of int)
        :param termination_flag: (boolean)
        :param a_: new action (str)
        :param gamma: [float between 0 and 1] discount factor
        :param learning_rate: [float between 0 and 1] - learning rate
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
            q_expected = self.q_table.loc[id_row_next_state, a_]
            q_target = r + gamma * q_expected

        # TD-error
        error = q_target - q_predict
        # sarsa would have just done:
        # self.q_table.loc[id_row_previous_state, a] += learning_rate * (q_target - q_predict)

        # increasing the importance factor for the visited state-action pair. Two methods:
        # Method 1: accumulating trace (not quite stable)
        # self.eligibility_trace.loc[id_row_previous_state, a] += 1

        # Method 2: replacing trace (normalization) - if I visit a state more than once, it still stays at 1, not more
        self.eligibility_trace.loc[id_row_previous_state, a] = 1

        # q_table update - most state will not be considered
        # ToDo: it is not necessary to consider all the states. Just those encountered during the episode
        # The importance factor (=eligibility_trace) says how important is to travel by this state to get the return
        self.q_table[self.actions_list] += learning_rate * error * self.eligibility_trace[self.actions_list]

        # print("self.q_table[self.actions_list] = \n{}".format(self.q_table.to_string()))
        # print("self.eligibility_trace[self.actions_list] = \n{}".format(self.eligibility_trace.to_string()))

        # decay eligibility trace after update (before the next step)
        self.eligibility_trace[self.actions_list] *= gamma * self.lambda_trace_decay


# Monte Carlo Control
class MC(Agent):
    def __init__(self, actions, state, load_q_table=False):
        super(MC, self).__init__(actions, state, load_q_table)
        self.nA = len(actions)
        # self.q_table = defaultdict(lambda: np.zeros(self.nA))

    def compare_reference_value(self):
        # ToDo: we know the value of the last-but one state at convergence: Q(s,a)=R(s,a).
        state = (16, 3)
        action_id = 0  # "no change"
        res = self.q_table[state][action_id]
        # should be +40
        print("reference_value = {}".format(res))
        return res

    def reset_q_table(self):
        # ToDo: dtype=np.float32 not necessary. Try lower precision
        self.q_table = defaultdict(lambda: np.zeros(self.nA))

    def choose_action(self, observation, masked_actions_list, greedy_epsilon):
        observation = tuple(observation)

        # apply action masking
        possible_actions = [action for action in self.actions_list if action not in masked_actions_list]

        # Epsilon-greedy action selection
        if np.random.uniform() > greedy_epsilon:
            # choose best action

            state_action = copy(self.q_table[observation])
            # print("state_action = {}".format(state_action))
            # print("possible_actions = {}".format(possible_actions))

            # restrict to allowed actions
            for action in self.actions_list:
                if action not in possible_actions:
                    action_id = self.actions_list.index(action)
                    state_action[action_id] = -np.inf  # using a copy
            # print("filtered state_action = {}".format(state_action))

            # make decision
            if np.all(np.isneginf([state_action])):
                action_id = random.choice(possible_actions)
                print('random action sampled among allowed actions')
            else:
                action_id = np.argmax(state_action)
                # Return index of first occurrence of maximum over requested axis (with shuffle)
            action_to_do = self.actions_list[action_id]
        else:
            action_to_do = np.random.choice(possible_actions)

        return action_to_do

    def learn(self, episode, gamma, learning_rate):
        """ updates the action-value function estimate using the most recent episode """
        states, actions, rewards = zip(*episode)
        # print("states = {}".format(states))
        # print("rewards = {}".format(rewards))
        # print("actions = {}".format(actions))
        # prepare for discounting
        discounts = np.array([gamma ** i for i in range(len(rewards) + 1)])
        for i, state in enumerate(states):
            action_id = self.actions_list.index(actions[i])
            # print(actions[i])
            # print(action_id)
            old_q = self.q_table[state][action_id]
            self.q_table[state][action_id] = old_q + learning_rate * (sum(rewards[i:] * discounts[:-(1 + i)]) - old_q)

    def save_q_table(self, save_directory):
        """
        """
        filename = "q_table"
        try:
            # to pickle
            output = open(save_directory + filename + ".pkl", 'wb')

            pickle.dump(dict(self.q_table), output)
            output.close()
            print("Saved as " + filename + ".pkl")

        except Exception as e:
            print(e)

    def print_q_table(self):
        # sort series according to the position
        q_table_dict = dict(self.q_table)
        q_table_pandas = pd.DataFrame(columns=self.columns_q_table, dtype=np.float32)
        for state, q_values in q_table_dict.items():
            new_data = np.concatenate((np.array(q_values), np.array(state)), axis=0)
            # print("new_data to add %s" % new_data)
            new_row = pd.Series(new_data, index=q_table_pandas.columns)
            q_table_pandas = q_table_pandas.append(new_row, ignore_index=True)

        q_table_pandas = q_table_pandas.sort_values(by=[self.state_features_list[0]])
        print(q_table_pandas.to_string())

    def load_q_table(self, weight_file=None):
        """
        open_model
        working with h5, csv or pickle format
        :return: -
        """
        try:
            # from pickle
            print(weight_file)
            loaded_dict = pd.read_pickle(weight_file)
            print(type(loaded_dict))

            self.q_table = defaultdict(lambda: np.zeros(self.nA))

            for state, value in loaded_dict.items():
                for i, q in enumerate(value):
                    # print("state = {}".format(state))
                    # print("i = {}".format(i))
                    # print("q = {}".format(q))
                    self.q_table[state][i] = q
            return True

        except Exception as e:
            print(e)
        return False


# Model-based
class DP(Agent):
    """
    DP stands for Dynamic Programming
    Model-Based: it has access to the Reward and Transition functions
    Agent used to get the optimal values (to set the success_threshold)
    """
    def __init__(self, actions, state, env, gamma, load_q_table=False):
        super(DP, self).__init__(actions, state, load_q_table)
        self.env = env
        self.nA = len(actions)
        self.n_position = 20
        self.n_velocity = 6
        self.gamma = gamma

    def learn(self):
        pass

    def get_value_from_state(self, state, action):
        """
        debug: make one step in the environment
        """
        [p, v] = state
        self.env.reset()
        self.env.move_to_state([p, v])  # teleportation
        next_observation, reward, termination_flag, _ = self.env.step(action)
        return next_observation, reward, termination_flag

    def run_policy(self, policy, initial_state, max_nb_steps=100):
        """
        run one episode with a policy
        """
        self.env.reset()

        # from Policy to Value Functions - for debug
        v_table = self.policy_evaluation(policy=policy)
        q_table = self.q_from_v(v_table)

        current_observation = initial_state
        self.env.move_to_state(initial_state)  # say the env to move to state [p][v]
        return_of_episode = 0
        trajectory = []

        step_count = 0
        while step_count < max_nb_steps:
            step_count += 1

            policy_for_this_state = policy[current_observation[0], current_observation[1]]
            print("policy_for_this_state = {}".format(policy_for_this_state))
            print("q_values_for_this_state = {}".format(q_table[current_observation[0], current_observation[1]]))

            action_id = np.argmax(policy[current_observation[0], current_observation[1]])
            action = self.actions_list[action_id]
            print("action = {}".format(action))

            trajectory.append(current_observation)
            trajectory.append(action)

            next_observation, reward, termination_flag, _ = self.env.step(action)
            print(" {}, {}, {} = results".format(next_observation, reward, termination_flag))

            return_of_episode += reward

            current_observation = next_observation
            if termination_flag:
                trajectory.append(next_observation)
                break

        print("return_of_episode = {}".format(return_of_episode))
        print("Trajectory = {}".format(trajectory))
        return return_of_episode, trajectory

    def q_from_v(self, v_table):
        """
        from the Value Function (for each state) to the Q-value Function (for each [state, action] pair)
        it makes sure masked actions have -np.inf values
        """
        q_table = np.ones((self.n_position, self.n_velocity, self.nA))

        # loop over all possible states (p, v)
        for p in range(self.n_position):
            for v in range(self.n_velocity):
                masked_actions_list = self.env.masking_function([p, v])
                possible_actions = [action for action in self.actions_list if action not in masked_actions_list]
                # print("possible_actions = {} for state = {}".format(possible_actions, [p, v]))

                for action_id in range(self.nA):
                    self.env.move_to_state([p, v])  # say the env to move on on state [p][v]
                    action = self.actions_list[action_id]
                    if action in possible_actions:
                        # print(" -- ")
                        # print(" {} taken in {}".format(action, [p, v]))
                        next_observation, reward, termination_flag, _ = self.env.step(action)

                        prob = 1  # it is a deterministic environment
                        if termination_flag:
                            # print(" with termination_flag, Q = prob {} * reward {}".format(prob, reward))
                            q_table[p][v][action_id] = prob * reward

                        else:
                            next_p = next_observation[0]
                            next_v = next_observation[1]
                            # print(" No termination_flag")
                            q_table[p][v][action_id] = prob * (reward + self.gamma * v_table[next_p][next_v])

                    else:
                        # print("Action {} cannot be taken in {}".format(action, [p, v]))
                        q_table[p][v][action_id] = -np.inf  # masked action
        return q_table

    def policy_improvement(self, v_table):
        """
        Used by Policy Iteration + Value Iteration

        Optimality Bellman operator:
        - from Value Function to a Policy
        - contains a max operator, which is non linear

        Two algorithms are highly similar (in their key steps):
        - policy improvement (this one involves a stability check) for Policy_Iteration
        - policy extraction (for Value_Iteration)
        """
        policy = np.zeros([self.n_position, self.n_velocity, self.nA]) / self.nA
        for p in range(self.n_position):
            for v in range(self.n_velocity):
                q_table = self.q_from_v(v_table)
                # OPTION 1: construct a deterministic policy
                # policy[p][v][np.argmax(q_table[p][v])] = 1  # make sure we have policy initialized with np.zeros()

                # OPTION 2: construct a stochastic policy that puts equal probability on maximizing actions
                best_a = np.argwhere(q_table[p][v] == np.max(q_table[p][v])).flatten()
                policy[p][v] = np.sum([np.eye(self.nA)[i] for i in best_a], axis=0) / len(best_a)

        return policy

    # truncated policy_evaluation
    def policy_evaluation(self, theta_value_function=10e-3, policy=None, max_counter=1e3):
        """
        From a Policy to its Value Function
        Used by Policy Iteration

        Truncated: No need to have the true absolute value function. The relative values are enough to get the Policy

        Two algorithms are highly similar except for a max operation:
        - policy evaluation (for Policy_Iteration)
        - finding optimal value function (for Value_Iteration)

        # -26.40 = v_table[19, 2] with random policy. Correct
        :param theta_value_function: threshold to consider two value functions similar
        :param policy: policy[state] = policy[p][v] = probabilities (numpy array) of taking each of the actions
        :param max_counter: truncated aspect - to stop iterations
        :return:
        """
        if policy is None:
            policy = np.ones([self.n_position, self.n_velocity, self.nA]) / self.nA  # random_policy
        # initialize arbitrarily
        v_table = np.zeros((self.n_position, self.n_velocity))
        counter = 0
        while counter < max_counter:

            counter += 1
            if counter % 1000 == 0:
                print(" --- {} policy_evaluation --- ".format(counter))
            delta_value_functions = 0

            # loop over all possible states (p, v)
            for p in range(self.n_position):
                for v in range(self.n_velocity):

                    v_state = 0
                    masked_actions_list = self.env.masking_function([p, v])
                    # print("masked_actions_list  = {}".format(masked_actions_list ))

                    possible_actions = [action for action in self.actions_list if action not in masked_actions_list]
                    # prob = 1 / len(possible_actions)

                    # policy[p][v] = [0.20 0.20 0.20 0.20 0.20]
                    for action_id, action_prob in enumerate(policy[p][v]):
                        self.env.move_to_state([p, v])  # say the env to move on on state [p][v]
                        # print(" {} == {}".format([self.env.state_ego_position, self.env.state_ego_velocity], [p, v]))

                        action = self.actions_list[action_id]
                        if action in possible_actions:
                            # print(" -- ")
                            # print(" {} taken in {}".format(action, [p, v]))
                            next_observation, reward, termination_flag, _ = self.env.step(action)
                            prob = 1  # deterministic environment
                            # print(" {}, {}, {} = results".format(next_observation, reward, termination_flag))
                            next_p = next_observation[0]
                            next_v = next_observation[1]
                            # next_p = min(next_observation[0], self.n_position - 1)
                            # next_v = min(next_observation[1], self.n_velocity - 1)
                            # print(" {} = action_prob, prob = {}".format(action_prob, prob))

                            if termination_flag:
                                # print(" with termination_flag, V = action_prob {} * prob {} * reward {}".format(
                                #     action_prob, prob, reward))
                                v_state += action_prob * prob * reward
                            else:
                                v_state += action_prob * prob * (reward + self.gamma * v_table[next_p][next_v])
                    delta_value_functions = max(delta_value_functions, np.abs(v_table[p][v] - v_state))
                    v_table[p][v] = v_state
                    # print("v_state = {}".format(v_state))

            if delta_value_functions < theta_value_function:
                break
        return v_table

    # truncated Policy_Iteration
    def policy_iteration(self, theta_value_function=1e-3, theta_final_value_function=1e-5, max_counter=1e3):
        """
        To approximate the optimal policy and value function
        Duration of Policy Iteration = 12.44 - counter = 5 - delta_policy = 0.0 with theta = 1e-3 and final theta = 1e-5

        Start with a random policy
        Policy iteration includes:
        - policy evaluation
        - policy improvement
        The two are repeated iteratively until policy converges

        In this process, each policy is guaranteed to be a strict improvement over the previous one (or we are done).
        Given a policy, its value function can be obtained using the "Bellman operator"

        Allegedly, this convergence of Policy Iteration is much faster than Value Iteration

        :param theta_value_function: for policy evaluation
        :param theta_final_value_function: for stopping the iteration. When policy have similar value_functions
        :param max_counter:
        :return:
        """
        time_start = time.time()

        policy = np.zeros([self.n_position, self.n_velocity, self.nA]) / self.nA
        counter = 0
        v_table = None
        delta_policy = None

        while counter < max_counter:
            counter += 1
            intermediate_time = time.time()
            duration = intermediate_time - time_start
            print(" - {}-th iteration in Policy_Iteration - duration = {:.2f} - delta_policy = {}".format(
                counter, duration, delta_policy))

            # 1- Evaluation: For fixed current policy, find values with policy evaluation
            v_table = self.policy_evaluation(theta_value_function=theta_value_function,
                                             policy=policy,
                                             max_counter=max_counter)
            new_policy = self.policy_improvement(v_table)

            # 2- Improvement : For fixed values, get a better policy using policy extraction (One-step look-ahead)
            # OPTION 1: stop if the policy is unchanged after an improvement step
            # if (new_policy == policy).all():
            #     break

            # OPTION 2: stop if the value function estimates for successive policies has converged
            # i.e. if policies have similar value functions
            delta_policy = np.max(abs(self.policy_evaluation(policy=policy,
                                                             theta_value_function=theta_value_function,
                                                             max_counter=max_counter)
                                      - self.policy_evaluation(policy=new_policy,
                                                               theta_value_function=theta_value_function,
                                                               max_counter=max_counter)
                                      ))
            if delta_policy < theta_final_value_function:
                break

            policy = copy(new_policy)

        if counter == max_counter:
            print("Policy_Iteration() stops because of max_counter = {}".format(max_counter))
        else:
            print("Policy_Iteration() stops because of theta_value_function = {}".format(theta_value_function))

        time_stop = time.time()
        duration = time_stop - time_start
        print("Duration of Policy Iteration = {:.2f} - counter = {} - delta_policy = {}".format(duration, counter,
                                                                                                delta_policy))

        return policy, v_table

    def value_iteration(self, theta_value_function=1e-5, max_counter=1e3):
        """
        To approximate the optimal policy and value function

        Duration of Value Iteration = 114.28 - counter = 121 - delta_value_functions = 9.687738053543171e-06

        Start with a random value function
        Value iteration includes:
        - finding optimal value function [can also be seen as a combination of
            - policy_improvement (due to max)
            - truncated policy_evaluation (reassign v_(s) after just 1 sweep of all states, regardless of convergence)]
        - one policy extraction.
        There is no repeat of the two because once the value function is optimal,
        then the policy out of it should also be optimal (i.e. converged)

        Every iteration updates both the values and (implicitly) the policy
        We don’t track the policy, but taking the max over actions implicitly recomputes it

        At the end, we derive the optimal policy from the optimal value function.
        This process is based on the "optimality Bellman operator" (contains a max operator, which is non linear)
        """
        time_start = time.time()

        # initialize V arbitrarily
        v_table = np.zeros((self.n_position, self.n_velocity))
        counter = 0
        delta_value_functions = None

        while counter < max_counter:
            counter += 1
            intermediate_time = time.time()
            duration = intermediate_time - time_start
            print(" - {}-th iteration in Value_Iteration - duration = {:.2f} - delta_value_functions = {}".format(
                counter, duration, delta_value_functions))

            delta_value_functions = 0
            # loop over all states
            for p in range(self.n_position):
                for v in range(self.n_velocity):
                    value = v_table[p][v]

                    # usually policy evaluation to update v_table[state] with Bellman. Here, one sweep only
                    q_table = self.q_from_v(v_table)
                    v_table[p][v] = np.max(q_table[p][v])

                    # check how much the value has changed
                    delta_value_functions = max(delta_value_functions, abs(v_table[p][v] - value))
            if delta_value_functions < theta_value_function:
                break
        # at this point, we have the Optimal Value_Function
        # let's obtain the corresponding policy
        policy = self.policy_improvement(v_table)

        if counter == max_counter:
            print("Value_Iteration() stops because of max_counter = {}".format(max_counter))
        else:
            print("Value_Iteration() stops because of theta_value_function = {}".format(theta_value_function))

        time_stop = time.time()
        duration = time_stop - time_start
        print("Duration of Value Iteration = {:.2f} - counter = {} - delta_value_functions = {}".format(
            duration, counter, delta_value_functions))

        return policy, v_table
