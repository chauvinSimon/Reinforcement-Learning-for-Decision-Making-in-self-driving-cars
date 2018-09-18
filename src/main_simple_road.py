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
from brains.simple_brains import MC
from brains.simple_brains import QLearningTable
from brains.simple_brains import SarsaTable
from brains.simple_brains import ExpectedSarsa
from brains.simple_brains import SarsaLambdaTable
from brains.simple_brains import DP
from brains.simple_DQN_tensorflow import DeepQNetwork
from brains.simple_DQN_pytorch import Agent
from collections import deque
import math
from utils.logger import Logger

# seed = np.random.seed(0)
plt.rcParams['figure.figsize'] = [20, 10]
np.set_printoptions(formatter={'float': lambda x: "{0:0.2f}".format(x)})


def train_agent(using_tkinter, agent, method, gamma, learning_rate, eps_start, eps_end, eps_decay,
                window_success, threshold_success, returns_list, steps_counter_list, info_training,
                max_nb_episodes, max_nb_steps, sleep_time, folder_name=""):
    """

    :param using_tkinter: [bool] to display the environment, or not
    :param agent: [brain object]
    :param method: [string] value-based learning method - either sarsa or q-learning
    :param gamma: [float between 0 and 1] discount factor
    If gamma is closer to one, the agent will consider future rewards with greater weight,
    willing to delay the reward.
    :param learning_rate: [float between 0 and 1] - Non-constant learning rate must be used?
    :param eps_start: [float]
    :param eps_end: [float]
    :param eps_decay: [float]
    :param window_success: [int]
    :param threshold_success: [float] to solve the env, = average score over the last x scores, where x = window_success
    :param returns_list: [list of float]
    :param steps_counter_list: [list of int]
    :param info_training: [dict]
    :param max_nb_episodes: [int] limit of training episodes
    :param max_nb_steps: [int] maximum number of timesteps per episode
    :param sleep_time: [int] sleep_time between two steps [ms]
    :param folder_name: [string] to distinguish between runs during hyper-parameter tuning
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
    max_window = -np.inf

    # initialize updated variable
    current_action = None
    next_observation = None

    # measure the running time
    time_start = time.time()
    nb_episodes_seen = max_nb_episodes
    #
    for episode_id in range(max_nb_episodes):  # limit the number of episodes during training
        # while episode_id < max_nb_episodes
        # episode_id = episode_id + 1

        # reset metrics
        step_counter = max_nb_steps  # length of episode
        return_of_episode = 0  # = score
        trajectory = []  # sort of replay-memory, just for debugging
        rewards = []
        actions = []
        changes_in_state = 0
        reward = 0
        next_action = None

        # reset the environment for a new episode
        current_observation, masked_actions_list = env.reset()  # initial observation = initial state

        # for sarsa - agent selects next action based on observation
        if (method == "sarsa") or (method == "sarsa_lambda"):
            current_action = agent.choose_action(current_observation, masked_actions_list, greedy_epsilon)
            if method == "sarsa_lambda":
                # for sarsa_lambda - initial all zero eligibility trace
                agent.reset_eligibility_trace()

        if method_used == "mc_control":
            # generate an episode by following epsilon-greedy policy
            episode = []
            current_observation, _ = env.reset()
            for step_id in range(max_nb_steps):  # while True
                current_action = agent.choose_action(tuple(current_observation), masked_actions_list, greedy_epsilon)
                next_observation, reward, termination_flag, masked_actions_list = env.step(current_action)
                return_of_episode += reward

                # a tuple is hashable and can be used in defaultdict
                episode.append((tuple(current_observation), current_action, reward))
                current_observation = next_observation

                if termination_flag:
                    step_counter = step_id
                    steps_counter_list.append(step_id)
                    returns_list.append(return_of_episode)
                    break

            # update the action-value function estimate using the episode
            # print("episode = {}".format(episode))
            # agent.compare_reference_value()
            agent.learn(episode, gamma, learning_rate)

        else:  # TD
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
                        # ToDo: here, we should read the masked_actions_list associated to the next_observation
                        masked_actions_list = env.masking_function(next_observation)
                        next_action = agent.choose_action(next_observation, masked_actions_list=masked_actions_list,
                                                          greedy_epsilon=greedy_epsilon)

                        # agent learn from this transition
                        agent.learn(current_observation, current_action, reward, next_observation, next_action,
                                    termination_flag, gamma, learning_rate)
                        current_observation = next_observation
                        current_action = next_action

                    if termination_flag:  # if done
                        agent.learn(current_observation, current_action, reward, next_observation, next_action,
                                    termination_flag, gamma, learning_rate)
                        # ToDo: check it ignore next_observation and next_action
                        step_counter = step_id
                        steps_counter_list.append(step_id)
                        returns_list.append(return_of_episode)
                        break

                elif (method == "q") or (method == "expected_sarsa") or (method == "simple_dqn_pytorch"):
                    current_action = agent.choose_action(current_observation, masked_actions_list, greedy_epsilon)
                    next_observation, reward, termination_flag, masked_actions_list = env.step(current_action)
                    return_of_episode += reward

                    if method == "q":
                        # agent learn from this transition
                        agent.learn(current_observation, current_action, reward, next_observation, termination_flag,
                                    gamma, learning_rate)

                    elif method == "simple_dqn_pytorch":
                        agent.step(current_observation, current_action, reward, next_observation, termination_flag)

                    elif method == "expected_sarsa":
                        agent.learn(current_observation, current_action, reward, next_observation, termination_flag,
                                    greedy_epsilon, gamma, learning_rate)

                    else:  # DQN with tensorflow
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
                        # agent.compare_reference_value()

                        break

                # log
                trajectory.append(current_observation)
                trajectory.append(current_action)

                # monitor actions, states and rewards are not constant
                rewards.append(reward)
                actions.append(current_action)
                if not (next_observation[0] == current_observation[0]
                        and next_observation[1] == current_observation[1]):
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
            # agent.print_q_table()

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

        if np.mean(returns_window) > max_window:
            max_window = np.mean(returns_window)

        # test success
        if np.mean(returns_window) >= threshold_success:
            time_stop = time.time()
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}, duration={:.2f} [s]'.format(
                episode_id - window_success, np.mean(returns_window), time_stop - time_start))
            info_training["nb_episodes_to_solve"] = episode_id - window_success
            nb_episodes_seen = episode_id
            break

    time_stop = time.time()
    info_training["duration"] = int(time_stop - time_start)
    info_training["nb_episodes_seen"] = nb_episodes_seen
    info_training["final_epsilon"] = greedy_epsilon
    info_training["max_window"] = max_window
    info_training["reference_values"] = agent.compare_reference_value()

    # where to save the weights
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder = os.path.join(parent_dir, "results/simple_road/" + folder_name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    agent.save_q_table(folder)

    print('End of training')
    print('Best return : %s --- with %s different trajectory(ies)' % (max_return, len(best_trajectories_list)))
    for trajectory in best_trajectories_list:
        print(trajectory)

    if using_tkinter:
        env.destroy()

    # return returns_list, steps_counter_list


def display_results(agent, method_used_to_plot, returns_to_plot, smoothing_window, threshold_success,
                    steps_counter_list_to_plot, display_flag=True, folder_name=""):
    """
    Use to SAVE + plot (optional)
    :param agent:
    :param method_used_to_plot:
    :param returns_to_plot:
    :param smoothing_window:
    :param threshold_success:
    :param steps_counter_list_to_plot:
    :param display_flag:
    :param folder_name:
    :return:
    """
    # where to save the plots
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder = os.path.join(parent_dir, "results/simple_road/" + folder_name)
    if not os.path.exists(folder):
        os.makedirs(folder)

    # plot step_counter for each episode
    plt.figure()
    plt.grid(True)
    plt.xlabel('Episode')
    plt.title("Episode Step_counts over Time (Smoothed over window size {})".format(smoothing_window))
    plt.ylabel("Episode step_count (Smoothed)")
    steps_smoothed = pd.Series(steps_counter_list_to_plot).rolling(
        smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(steps_counter_list_to_plot, linewidth=0.5)
    plt.plot(steps_smoothed, linewidth=2.0)
    plt.savefig(folder + "step_counter.png", dpi=800)
    if display_flag:
        plt.show()

    plt.figure()
    plt.grid(True)
    returns_smoothed = pd.Series(returns_to_plot).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(returns_to_plot, linewidth=0.5)
    plt.plot(returns_smoothed, linewidth=2.0)
    plt.axhline(y=threshold_success, color='r', linestyle='-')
    plt.xlabel("Episode")
    plt.ylabel("Episode Return(Smoothed)")
    plt.title("Episode Return over Time (Smoothed over window size {})".format(smoothing_window))
    plt.savefig(folder + "return.png", dpi=800)
    if display_flag:
        plt.show()

    # bins = range(min(returns_to_plot), max(returns_to_plot) + 1, 1)
    plt.figure()
    plt.hist(returns_to_plot, norm_hist=True, bins=100)
    plt.ylabel('reward distribution')
    if display_flag:
        plt.show()

    agent.print_q_table()
    if method_used_to_plot not in ["simple_dqn_tensorflow", "simple_dqn_pytorch", "mc_control"]:
        agent.plot_q_table(folder, display_flag)
        agent.plot_optimal_actions_at_each_position(folder, display_flag)


def test_agent(using_tkinter_test, agent, returns_list, nb_episodes=1, max_nb_steps=20, sleep_time=0.001,
               weight_file_name="q_table.pkl"):
    """
    load weights and show one run
    :param using_tkinter_test: [bool]
    :param agent: [brain object]
    :param returns_list: [float list] - argument passed by reference
    :param nb_episodes: [int]
    :param max_nb_steps: [int]
    :param sleep_time: [float]
    :param weight_file_name: [string]
    :return: -
    """
    grand_parent_dir_test = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    weight_file = os.path.abspath(grand_parent_dir_test + "/results/simple_road/" + weight_file_name)
    if agent.load_q_table(weight_file):

        for episode_id in range(nb_episodes):
            trajectory = []
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

                trajectory.append(current_observation)
                trajectory.append(current_action)
                trajectory.append(reward)
                trajectory.append(termination_flag)

                # update state
                current_observation = next_observation
                print("\r {}, {}, {}.".format(current_action, reward, termination_flag), end="")
                sys.stdout.flush()
                if termination_flag:  # exit loop if episode finished
                    trajectory.append(next_observation)
                    break

            returns_list.append(score)
            print("\n{}/{} - Return: {}".format(episode_id, nb_episodes, score))
            print("\nTrajectory = {}".format(trajectory))
            # Best trajectory= [[0, 3], 'no_change', [3, 3], 'no_change', [6, 3], 'no_change', [9, 3], 'slow_down',
            # [11, 2], 'no_change', [13, 2], 'speed_up', [16, 3], 'no_change', [19, 3]]

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

    # avoid special types:
    not_to_consider = ["tk", "children", "canvas", "_tclCommands", "master", "_tkloaded", "colour_action_code",
                       "colour_velocity_code", "origin_coord", "display_canvas", "origin", "_last_child_ids", "rect",
                       "logger"]
    for elem in not_to_consider:
        if elem in dict_configuration:
            del dict_configuration[elem]
    # saving the configuration in a json
    with open('environments/simple_road_env_configuration.json', 'w') as outfile:
        json.dump(dict_configuration, outfile)

    # Different possible algorithms to update the state-action table:
    # -1- Monte-Carlo  # working
    # method_used = "mc_control"  # definitely the faster [in term of duration not nb_episodes]. below 1 min

    # -2- Temporal-Difference  # all are working - "q" performs the best
    method_used = "q"
    # method_used = "sarsa"
    # method_used = "expected_sarsa"
    # method_used = "sarsa_lambda"  # worked with trace_decay=0.3

    # -3- deep TD
    # method_used = "simple_dqn_tensorflow"
    # method_used = "simple_dqn_pytorch"  # model is correct - ToDo: hyper-parameter tuning

    # -4- Model-Based Dynamic Programming
    # Dynamic programming assumes that the agent has full knowledge of the MDP
    # method_used = "DP"

    # Instanciate an Agent
    brain_agent = None
    if method_used == "mc_control":
        brain_agent = MC(actions=actions_list, state=state_features_list, load_q_table=False)
    elif method_used == "q":
        brain_agent = QLearningTable(actions=actions_list, state=state_features_list, load_q_table=False)
    elif method_used == "sarsa":
        brain_agent = SarsaTable(actions=actions_list, state=state_features_list, load_q_table=False)
    elif method_used == "expected_sarsa":
        brain_agent = ExpectedSarsa(actions=actions_list, state=state_features_list, load_q_table=False)
    elif method_used == "sarsa_lambda":
        brain_agent = SarsaLambdaTable(actions=actions_list, state=state_features_list, load_q_table=False,
                                       trace_decay=0.3)
    elif method_used == "simple_dqn_tensorflow":
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
    elif method_used == "simple_dqn_pytorch":
        brain_agent = Agent(actions=actions_list, state=state_features_list)

    elif method_used == "DP":
        # make sure it does not do any training or testing (it has all its methods are implemented internally)
        brain_agent = DP(actions=actions_list, state=state_features_list, env=env, gamma=0.9)

        # ToDo: Problem at state = [3, 3]
        # q_values_for_this_state = [8.40 9.23 -inf 4.76 -2.11] makes the agent go for speed_up => Best = 17 (not 18)

        # check the interface with the environment through specific values
        final_state = [19, 3]
        final_action = "no_change"
        next_observation_dp, reward_dp, termination_flag_dp = brain_agent.get_value_from_state(final_state,
                                                                                               final_action)
        action = "no_change"
        obstacle_state = [12, 2]
        next_observation_dp, reward_dp, termination_flag_dp = brain_agent.get_value_from_state(obstacle_state, action)
        print(" {}, {}, {} = results".format(next_observation_dp, reward_dp, termination_flag_dp))

        # compare value_iteration and policy_iteration
        opt_policy_pi, opt_v_table_pi = brain_agent.policy_iteration()
        np.save('opt_policy_pi.npy', opt_policy_pi)
        np.save('opt_v_table_pi.npy', opt_v_table_pi)
        opt_q_table_pi = brain_agent.q_from_v(opt_v_table_pi)
        np.save('opt_q_table_pi.npy', opt_q_table_pi)
        print("final_state_values p_i = {}".format(opt_q_table_pi[final_state[0]][final_state[1]]))
        print(opt_v_table_pi)
        print(opt_q_table_pi)

        # opt_policy_pi = np.load('opt_policy_pi.npy')
        return_of_episode_pi, trajectory_pi = brain_agent.run_policy(opt_policy_pi, [0, 3])
        print("p_i has return = {} for trajectory = {}".format(return_of_episode_pi, trajectory_pi))

        print("\n --- \n")

        opt_policy_vi, opt_v_table_vi = brain_agent.value_iteration()
        np.save('opt_policy_vi.npy', opt_policy_vi)
        np.save('opt_v_table_vi.npy', opt_v_table_vi)
        opt_q_table_vi = brain_agent.q_from_v(opt_v_table_vi)
        np.save('opt_q_table_vi.npy', opt_q_table_vi)
        print("final_state_values v_i = {}".format(opt_q_table_vi[final_state[0]][final_state[1]]))
        print(opt_v_table_vi)
        print(opt_q_table_vi)

        return_of_episode_vi, trajectory_vi = brain_agent.run_policy(opt_policy_vi, [0, 3])
        print("v_i has return = {} for trajectory = {}".format(return_of_episode_vi, trajectory_vi))

    # Training and/or Testing
    flag_training_once = True
    flag_testing = False
    flag_training_hyper_parameter_tuning = False  # Tkinter is not used when tuning hyper-parameters
    display_learning_results = False  # only used for training_once

    # for testing
    max_nb_steps_testing = 50
    nb_tests = 10
    sleep_time_between_steps_testing = 0.5  # slow to see the steps

    # for learning
    # hyper-parameters
    gamma_learning = 0.99
    learning_rate_learning = 0.02
    eps_start_learning = 1.0
    eps_end_training = 0.01
    # reach eps_end at episode_id = log10(eps_end/eps_start) / log10(eps_decay)
    # 0.99907 for 5000 at 0.01/1.0
    eps_decay_training = 0.998466
    # eps_decay_training = 0.99907  # - when 70000 episode
    # 0.99907  # for getting to 0.01 in ~5000 episodes

    # to reach eps_end at episode episode_id, eps_decay = (eps_end / eps_start) ** (1/episode_id)
    max_nb_episodes_training = 7000
    max_nb_steps_training = 25
    sleep_time_between_steps_learning = 0.0005

    # success conditions
    window_success_res = 100
    threshold_success_training = 17
    dict_info_training = {}
    # 22.97 for self.reward = 1 + self.reward / max(self.rewards_dict.values())
    # q_max = 9.23562904132267 for expected_sarsa

    if flag_training_hyper_parameter_tuning:

        # No tkinter used
        learning_rate_list = [0.003, 0.01, 0.03, 0.1, 0.3, 1]

        gamma_learning_list = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95, 0.99, 1]
        nb_episodes_to_plateau_list = [300, 500, 800, 1000, 3000, 5000]
        # [0.954992586021, 0.9847666521101, 0.995405417351, 0.998466120868, 0.9995395890030, 0.9999846495505]
        eps_decay_list = [(eps_end_training / eps_start_learning) ** (1/nb) for nb in nb_episodes_to_plateau_list]

        for i, param in enumerate(eps_decay_list):
            brain_agent.reset_q_table()  # re-initialize the model!!

            folder_name_training = str(i) + '/'
            logger_name = str(i) + '.log'
            logger = Logger(folder_name_training, logger_name, 0)

            hyper_parameters = (
                method_used,
                gamma_learning,
                learning_rate_learning,
                eps_start_learning,
                eps_end_training,
                param  # decay
            )
            logger.log(str(hyper_parameters), 1)
            # after = Register an alarm callback that is called after a given time.
            # give results as reference
            returns_list_res, steps_counter_list_res = [], []
            dict_info_training = {}

            train_agent(flag_tkinter, brain_agent, *hyper_parameters,
                        window_success_res, threshold_success_training, returns_list_res,
                        steps_counter_list_res, dict_info_training,
                        max_nb_episodes_training, max_nb_steps_training, sleep_time_between_steps_learning,
                        folder_name_training)
            logger.log(dict_info_training, 1)

            try:
                display_results(brain_agent, method_used, returns_list_res, window_success_res,
                                threshold_success_training, steps_counter_list_res,
                                display_flag=False, folder_name=folder_name_training)
            except Exception as e:
                print('Exception = {}'.format(e))

            # testing
            returns_list_testing = []  # passed as a reference
            test_agent(flag_tkinter, brain_agent, returns_list_testing, nb_tests, max_nb_steps_testing,
                       sleep_time_between_steps_learning, folder_name_training + "q_table.pkl")
            logger.log(returns_list_testing, 1)

    if flag_training_once:
        hyper_parameters = (
            method_used,
            gamma_learning,
            learning_rate_learning,
            eps_start_learning,
            eps_end_training,
            eps_decay_training
        )
        print("hyper_parameters = {}".format(hyper_parameters))
        returns_list_res, steps_counter_list_res = [], []
        if flag_tkinter:
            # after(self, time [ms] before execution of func(*args), func=None, *args):
            # !! callback function. No return value can be read
            env.after(100, train_agent, flag_tkinter, brain_agent,
                      *hyper_parameters,
                      window_success_res, threshold_success_training, returns_list_res,
                      steps_counter_list_res, dict_info_training,
                      max_nb_episodes_training, max_nb_steps_training, sleep_time_between_steps_learning)
            env.mainloop()
            print("returns_list_res = {}, window_success_res = {}, steps_counter_list_res = {}".format(
                returns_list_res, window_success_res, steps_counter_list_res))
        else:
            train_agent(flag_tkinter, brain_agent, *hyper_parameters,
                        window_success_res, threshold_success_training, returns_list_res,
                        steps_counter_list_res, dict_info_training,
                        max_nb_episodes_training, max_nb_steps_training, sleep_time_between_steps_learning)
        try:
            display_results(brain_agent, method_used, returns_list_res, window_success_res,
                            threshold_success_training, steps_counter_list_res,
                            display_flag=display_learning_results)
        except Exception as e:
            print('Exception = {}'.format(e))
        print("hyper_parameters = {}".format(hyper_parameters))

        # print(brain_agent.reference_list)

    if flag_testing:
        returns_list_testing = []
        if flag_tkinter:
            env.after(100, test_agent, flag_tkinter, brain_agent, returns_list_testing, nb_tests, max_nb_steps_testing,
                      sleep_time_between_steps_testing)
            env.mainloop()
        else:
            test_agent(flag_tkinter, brain_agent, returns_list_testing, nb_tests, max_nb_steps_testing,
                       sleep_time_between_steps_testing)
