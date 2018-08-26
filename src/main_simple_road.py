"""
Reinforcement Learning example for the motion of an driving agent on a straight road.
-   The brain (called "RL") is chosen among the different implementations in RL_brain.py
-   The Environment in simple_road_env.py

Discrete State Space
-   see simple_road_env.py

Action Space:
-	“Maintain” current lane and speed,
-	“Accelerate” at rate = a1[m/s2], provided velocity does not exceed vmax[km/h],
-	“Decelerate” at rate = −a1[m/s2], provided velocity is above vmin[km/h],
-	“Hard Accelerate” at rate = a2[m/s2], provided velocity does not exceed vmax[km/h],
-	“Hard Decelerate” at rate = −a2[m/s2], provided velocity is above vmin[km/h],
(acceleration are given for a constant amount this time step)

To Do:
-	Add actions
-       -- Change lane to the left, provided there is a lane on the left,
-   	-- Change lane to the right, provided there is a lane on the right
- Trying to reduce chance of random action as we train the model
- define a test + baseline
    - simulator?
    - I could not find any reference KPI to compare with
    - See the agent running after it solved the env

Bug:
-   None
"""

import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time  # to time the learning process
import json  # to get the configuration of the environment
from environments.simple_road_env import Road
from brains.simple_brains import QLearningTable
from brains.simple_brains import SarsaTable
from brains.simple_brains import ExpectedSarsa
from brains.simple_brains import SarsaLambdaTable
from brains.simple_brains import QLearningApproximation
from brains.simple_DQN import DeepQNetwork
from collections import deque
import math

plt.rcParams['figure.figsize'] = [20, 10]


def train_agent(using_tkinter, agent, method, window_success, threshold_success, returns_list, steps_counter_list,
                eps_start=0.9, eps_end=0.01, eps_decay=0.935,
                max_nb_episodes=2000, max_nb_steps=25, sleep_time=0.001):
    # def train_agent(using_tkinter, agent, method, eps_start=0.9, eps_end=0.01, eps_decay=0.935,
    #                 max_nb_episodes=2000, max_nb_steps=25, sleep_time=0.001):
    """

    :param using_tkinter: [bool] to display the environment, or not
    :param agent: [brain object]
    :param method: [string] value-based learning method - either sarsa or q-learning
    :param window_success: [int]
    :param threshold_success: [float] to solve the env, = average score over the last x scores, where x = window_success
    :param returns_list: [list of float]
    :param steps_counter_list: [list of int]
    :param eps_start: [float]
    :param eps_end: [float]
    :param eps_decay: [float]
    :param max_nb_episodes: [int] limit of training episodes
    :param max_nb_steps: [int] maximum number of timesteps per episode
    :param sleep_time: [int] sleep_time between two steps [ms]
    :return: [list] returns_list - to be displayed
    """
    returns_window = deque(maxlen=window_success)  # last x scores, where x = window_success

    # probability of random choice for epsilon-greedy action selection
    greedy_epsilon = eps_start

    # record for each episode:
    # steps_counter_list = []  # number of steps in each episode - look if some get to max_nb_steps
    # returns_list = []  # return in each episode
    best_trajectories_list = []

    # track maximum return
    max_return = -math.inf  # to be set low enough (setting max_nb_steps * max_cost_per_step should do it)

    # initialize updated variable
    episode_id = 0
    current_action = None
    next_observation = None

    # measure the running time
    time_start = time.time()

    #
    while episode_id < max_nb_episodes:  # limit the number of episodes during training
        episode_id = episode_id + 1

        # reset metrics
        step_counter = max_nb_steps  # length of episode
        return_of_episode = 0  # = score
        trajectory = []  # sort of replay-memory, just for debugging
        rewards = []
        actions = []
        changes_in_state = 0
        next_action = None

        # reset the environment for a new episode
        current_observation, masked_actions_list = env.reset()  # initial observation = initial state

        # for sarsa - agent selects next action based on observation
        if (method == "sarsa") or (method == "sarsa_lambda"):
            current_action = agent.choose_action(current_observation, masked_actions_list, greedy_epsilon)
            if method == "sarsa_lambda":
                # for sarsa_lambda - initial all zero eligibility trace
                agent.reset_eligibility_trace()

        # run episodes
        for step_id in range(max_nb_steps):
            # ToDo: how to penalize the agent that does not terminate the episode?

            # fresh env
            if using_tkinter:
                env.render(sleep_time)

            if (method == "sarsa") or (method == "sarsa_lambda"):
                next_observation, reward, termination_flag, masked_actions_list = env.step(current_action)
                return_of_episode += reward
                if not termination_flag:  # if done
                    # Online-Policy: Choose an action At+1 following the same e-greedy policy based on current Q
                    next_action = agent.choose_action(next_observation, masked_actions_list=[],
                                                      greedy_epsilon=greedy_epsilon)

                    # agent learn from this transition
                    agent.learn(current_observation, current_action, reward, next_observation, next_action,
                                termination_flag)
                    current_observation = next_observation
                    current_action = next_action

                if termination_flag:  # if done
                    agent.learn(current_observation, current_action, reward, next_observation, next_action,
                                termination_flag)
                    # ToDo: check it ignore next_observation and next_action
                    step_counter = step_id
                    steps_counter_list.append(step_id)
                    returns_list.append(return_of_episode)
                    break

            else:
                current_action = agent.choose_action(current_observation, masked_actions_list, greedy_epsilon)
                next_observation, reward, termination_flag, masked_actions_list = env.step(current_action)
                return_of_episode += reward

                if method == "q":
                    # agent learn from this transition
                    agent.learn(current_observation, current_action, reward, next_observation, termination_flag)

                elif method == "expected_sarsa":
                    agent.learn(current_observation, current_action, reward, next_observation, termination_flag,
                                greedy_epsilon)

                elif method == "q_approximation":
                    # Update the function approximator using our target
                    # estimator.update(state, current_action, td_target)
                    agent.learn(current_observation, current_action, reward, next_observation, termination_flag)

                else:  # DQN
                    # New: store transition in memory - subsequently to be sampled from
                    agent.store_transition(current_observation, current_action, reward, next_observation)

                    # if the number of steps is larger than a threshold, start learn ()
                    if (step_id > 5) and (step_id % 5 == 0):  # for 1 to T
                        # print('learning')
                        # pick up some transitions from the memory and learn from these samples
                        agent.learn()
                current_observation = next_observation
                if termination_flag:  # if done
                    step_counter = step_id
                    steps_counter_list.append(step_id)
                    returns_list.append(return_of_episode)
                    break

            # log
            trajectory.append(current_observation)
            trajectory.append(current_action)

            # monitor actions, states and rewards are not constant
            rewards.append(reward)
            actions.append(current_action)
            if not (next_observation[0] == current_observation[0] and next_observation[1] == current_observation[1]):
                changes_in_state = changes_in_state + 1

        # At this point, the episode is terminated
        # decay epsilon
        greedy_epsilon = max(eps_end, eps_decay * greedy_epsilon)

        # log
        trajectory.append(next_observation)  # final state
        returns_window.append(return_of_episode)  # save most recent score
        if episode_id % 100 == 0:
            time_intermediate = time.time()
            print('\n --- Episode={} ---\n eps={}\n Average Score in returns_window = {:.2f} \n duration={:.2f}'.format(
                episode_id, greedy_epsilon, np.mean(returns_window), time_intermediate - time_start))

        if episode_id % 20 == 0:
            print('Episode {} / {}. Eps = {}. Total_steps = {}. Return = {}. Max return = {}, Top 10 = {}'.format(
                episode_id+1, max_nb_episodes, greedy_epsilon, step_counter, return_of_episode, max_return,
                sorted(returns_list, reverse=True)[:10]))

        if return_of_episode == max_return:
            if trajectory not in best_trajectories_list:
                best_trajectories_list.append(trajectory)
        elif return_of_episode > max_return:
            del best_trajectories_list[:]
            best_trajectories_list.append(trajectory)
            max_return = return_of_episode

        # test success
        if np.mean(returns_window) >= threshold_success:
            time_stop = time.time()
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}, duration={:.2f} [s]'.format(
                episode_id - window_success, np.mean(returns_window), time_stop - time_start))
            break

    # ToDo: save weights
    # where to save the weights
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder = os.path.join(parent_dir, "results/simple_road/")
    agent.save_q_table(folder)

    print('End of training')
    print('Best return : %s --- with %s different trajectory(ies)' % (max_return, len(best_trajectories_list)))
    for trajectory in best_trajectories_list:
        print(trajectory)

    if using_tkinter:
        env.destroy()

    # return returns_list, steps_counter_list


def display_results(agent, method_used_to_plot, returns_to_plot, smoothing_window, threshold_success,
                    steps_counter_list_to_plot):

    # where to save the plots
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder = os.path.join(parent_dir, "results/simple_road/")

    # plot step_counter for each episode
    plt.grid(True)
    plt.xlabel('Episode')
    plt.title("Episode Step_counts over Time (Smoothed over window size {})".format(smoothing_window))
    plt.ylabel("Episode step_count (Smoothed)")
    steps_smoothed = pd.Series(steps_counter_list_to_plot).rolling(
        smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(steps_counter_list_to_plot, linewidth=0.5)
    plt.plot(steps_smoothed, linewidth=2.0)
    plt.savefig(folder + "step_counter.png")
    plt.show()

    plt.grid(True)
    returns_smoothed = pd.Series(returns_to_plot).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(returns_to_plot, linewidth=0.5)
    plt.plot(returns_smoothed, linewidth=2.0)
    plt.axhline(y=threshold_success, color='r', linestyle='-')
    plt.xlabel("Episode")
    plt.ylabel("Episode Return(Smoothed)")
    plt.title("Episode Return over Time (Smoothed over window size {})".format(smoothing_window))
    plt.savefig(folder + "return.png")
    plt.show()

    # bins = range(min(returns_to_plot), max(returns_to_plot) + 1, 1)
    plt.hist(returns_to_plot, normed=True, bins=100)
    plt.ylabel('reward distribution')
    plt.show()

    # Plot policy
    if method_used_to_plot == "q_approximation":
        agent.create_q_table()
    if method_used_to_plot != "DQN":
        agent.print_q_table()
        agent.plot_q_table(folder)
        agent.plot_optimal_actions_at_each_position(folder)


def test_agent(using_tkinter_test, agent, nb_episodes=1, max_nb_steps=20, sleep_time=0.001):
    """
    load weights and show one run
    :param using_tkinter_test: [bool]
    :param agent: [brain object]
    :param nb_episodes: [int]
    :param max_nb_steps: [int]
    :param sleep_time: [float]
    :return: -
    """
    returns_list = []
    grand_parent_dir_test = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weight_file = os.path.abspath(grand_parent_dir_test + "/results/simple_road/q_table")
    if agent.load_q_table(weight_file):

        for episode_id in range(nb_episodes):
            # reset the environment for a new episode
            current_observation, masked_actions_list = env.reset()  # initial observation = initial state
            print("{} = initial_observation".format(current_observation))
            score = 0  # initialize the score
            step_id = 0
            while step_id < max_nb_steps:
                step_id += 1
                # fresh env
                if using_tkinter_test:
                    env.render(sleep_time)

                # agent choose current_action based on observation
                greedy_epsilon = 0
                current_action = agent.choose_action(current_observation, masked_actions_list, greedy_epsilon)

                next_observation, reward, termination_flag, masked_actions_list = env.step(current_action)

                score += reward  # update the score

                # update state
                current_observation = next_observation
                print("\r {}, {}, {}.".format(current_action, reward, termination_flag), end="")
                sys.stdout.flush()
                if termination_flag:  # exit loop if episode finished
                    break

            returns_list.append(score)
            print("\n{}/{} - Return: {}".format(episode_id, nb_episodes, score))

        print("---")
        print("{} = average return".format(np.mean(returns_list)))
    else:
        print("cannot load weight_file at {}".format(weight_file))


if __name__ == "__main__":
    actions_list = ["no_change", "speed_up", "speed_up_up", "slow_down", "slow_down_down"]
    state_features_list = ["position", "velocity"]  # , "obstacle_position"]

    # the environment
    flag_tkinter = False
    initial_state = [0, 3, 12]
    goal_velocity = 3
    env = Road(flag_tkinter, actions_list, state_features_list, initial_state, goal_velocity)

    # getting the configuration of the test
    env_configuration = vars(env)
    dict_configuration = dict(env_configuration)

    # trick to avoid crashing:
    not_to_consider = ["tk", "children", "canvas", "_tclCommands", "master", "_tkloaded", "colour_action_code",
                       "colour_velocity_code", "origin_coord", "display_canvas", "origin", "_last_child_ids", "rect",
                       "logger"]
    for elem in not_to_consider:
        if elem in dict_configuration:
            del dict_configuration[elem]
    # saving the configuration in a json
    with open('environments/simple_road_env_configuration.json', 'w') as outfile:
        json.dump(dict_configuration, outfile)

    # Three possible algorithm to learn the state-action table:
    method_used = "q"
    # method_used = "q_approximation"
    # method_used = "sarsa"
    # method_used = "expected_sarsa"
    # method_used = "sarsa_lambda"
    # method_used = "DQN"

    # Instanciate an Agent - the brain
    brain_agent = None
    if method_used == "q":
        brain_agent = QLearningTable(actions=actions_list, state=state_features_list, load_q_table=False)
    elif method_used == "q_approximation":
        # regressor = "linearSGD"
        regressor = "MLP"
        brain_agent = QLearningApproximation(actions=actions_list, state=state_features_list, load_q_table=False,
                                             regressor=regressor)
    elif method_used == "sarsa":
        brain_agent = SarsaTable(actions=actions_list, state=state_features_list, load_q_table=False)
    elif method_used == "expected_sarsa":
        brain_agent = ExpectedSarsa(actions=actions_list, state=state_features_list, load_q_table=False)
    elif method_used == "sarsa_lambda":
        brain_agent = SarsaLambdaTable(actions=actions_list, state=state_features_list, load_q_table=False)
    elif method_used == "DQN":
        grand_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_dir = os.path.abspath(grand_parent_dir + "/results/simple_road/")
        print("results_dir = {}".format(results_dir))
        brain_agent = DeepQNetwork(actions=actions_list,
                                   state=state_features_list,
                                   learning_rate=0.01,
                                   reward_decay=0.9,
                                   # e_greedy=0.9,
                                   replace_target_iter=300,  # replace net parameters every X learning
                                   memory_size=50,
                                   summaries_dir=results_dir,
                                   # saver_dir='/tmp/tensorflow_logs/RL/saver/'
                                   saver_dir=None
                                   )
    flag_training = True
    flag_testing = False

    if flag_training:
        display_learning_results = True
        # training parameters
        eps_start_learning = 1.0
        eps_end_training = 0.01
        # 0.99907 for 5000 at 0.01/1.0
        eps_decay_training = 0.99907  # reach eps_end at episode_id = log10(eps_end/eps_start) / log10(eps_decay)
        # to reach eps_end at episode episode_id, eps_decay = (eps_end / eps_start) ** (1/episode_id)
        max_nb_episodes_training = 5000
        max_nb_steps_training = 25
        sleep_time_between_steps = 0.0005

        # success conditions
        window_success_res = 100
        threshold_success_training = 13
        # 22.97 for self.reward = 1 + self.reward / max(self.rewards_dict.values())
        # q_max = 9.23562904132267 for expected_sarsa

        # after = Register an alarm callback that is called after a given time.
        # give results as reference
        returns_list_res, steps_counter_list_res = [], []
        if flag_tkinter:
            # after(self, time [ms] before execution of func(*args), func=None, *args):
            # !! callback function. No return value can be read
            env.after(100, train_agent, flag_tkinter, brain_agent, method_used,
                      window_success_res, threshold_success_training, returns_list_res,
                      steps_counter_list_res, eps_start_learning, eps_end_training, eps_decay_training,
                      max_nb_episodes_training, max_nb_steps_training, sleep_time_between_steps)
            env.mainloop()
            print("returns_list_res = {}, window_success_res = {}, steps_counter_list_res = {}".format(
                returns_list_res, window_success_res, steps_counter_list_res))
        else:
            train_agent(flag_tkinter, brain_agent, method_used, window_success_res, threshold_success_training,
                        returns_list_res, steps_counter_list_res, eps_start_learning, eps_end_training,
                        eps_decay_training, max_nb_episodes_training, max_nb_steps_training, sleep_time_between_steps)
        if display_learning_results:
            # ToDo: plot horizontal threshold_success_training
            display_results(brain_agent, method_used, returns_list_res, window_success_res, threshold_success_training,
                            steps_counter_list_res)

    if flag_testing:
        max_nb_steps_testing = 50
        nb_tests = 10
        sleep_time_between_steps = 0.5  # slow to see the steps
        if flag_tkinter:
            env.after(100, test_agent, flag_tkinter, brain_agent, nb_tests, max_nb_steps_testing,
                      sleep_time_between_steps)
            env.mainloop()
        else:
            test_agent(flag_tkinter, brain_agent, nb_tests, max_nb_steps_testing, sleep_time_between_steps)
