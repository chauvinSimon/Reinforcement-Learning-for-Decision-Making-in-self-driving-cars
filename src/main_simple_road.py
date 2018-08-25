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
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time  # to time the learning process
import json  # to get the configuration of the environment
from environments.simple_road_env import Road
from brains.simple_brains import QLearningTable
from brains.simple_brains import SarsaTable
from brains.simple_brains import SarsaLambdaTable
from brains.simple_brains import QLearningApproximation
from brains.simple_DQN import DeepQNetwork
from collections import deque
import math

plt.rcParams['figure.figsize'] = [20, 10]


def train_agent(using_tkinter, agent, method, eps_start=0.9, eps_end=0.01, eps_decay=0.935,
                max_nb_episodes=2000, max_nb_steps=25, sleep_time=0.001):
    """

    :param using_tkinter: [bool] to display the environment, or not
    :param agent: [brain object]
    :param method: [string] value-based learning method - either sarsa or q-learning
    :param eps_start: [float]
    :param eps_end: [float]
    :param eps_decay: [float]
    :param max_nb_episodes: [int] limit of training episodes
    :param max_nb_steps: [int] maximum number of timesteps per episode
    :param sleep_time: [int] sleep_time between two steps [ms]
    :return: [list] returns_list - to be displayed
    """
    # condition for success
    threshold_success = 10  # to solve the env, = average score over the last x scores, where x = window_success
    window_success = 100
    returns_window = deque(maxlen=window_success)  # last x scores, where x = window_success

    # probability of random choice for epsilon-greedy action selection
    greedy_epsilon = eps_start

    # record for each episode:
    steps_counter_list = []  # number of steps in each episode - look if some get to max_nb_steps
    returns_list = []  # return in each episode
    best_trajectories_list = []

    # track maximum return
    max_return = -math.inf  # to be set low enough (setting max_nb_steps * max_cost_per_step should do it)

    # initialize updated variable
    episode_id = 0
    current_action = ""
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

        # reset the environment for a new episode
        current_observation, masked_actions_list = env.reset()  # initial observation = initial state

        # for sarsa - agent selects action based on observation
        if method == "sarsa":
            current_action = agent.choose_action(current_observation, masked_actions_list, greedy_epsilon)

        if method == "sarsa_lambda":
            # for sarsa_lambda - initial all zero eligibility trace
            agent.reset_eligibility_trace()
            current_action = agent.choose_action(current_observation, masked_actions_list, greedy_epsilon)

        # run episodes
        for step_id in range(max_nb_steps):

            # fresh env
            if using_tkinter:
                env.render(sleep_time)

            if (method == "sarsa") or (method == "sarsa_lambda"):
                next_observation, reward, termination_flag, masked_actions_list = env.step(current_action)

                # agent choose current_action based on observation
                next_action = agent.choose_action(current_observation, masked_actions_list, greedy_epsilon)
                trajectory.append(current_observation)
                trajectory.append(current_action)

                # agent learn from this transition
                agent.learn(current_observation, current_action, reward, next_observation, next_action,
                            termination_flag)
                current_action = next_action

            else:  # Q-learning (classic and with approximation) or DQN
                # agent choose action based on observation
                current_action = agent.choose_action(current_observation, masked_actions_list, greedy_epsilon)
                trajectory.append(current_observation)
                trajectory.append(current_action)
                # print("current_action selected = %s" % current_action)

                # agent take action and get next observation (~state) and reward.
                # Also the masked actions for the next step
                next_observation, reward, termination_flag, masked_actions_list = env.step(current_action)

                if method == "q":
                    # agent learn from this transition
                    agent.learn(current_observation, current_action, reward, next_observation, termination_flag)

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

            # update state
            current_observation = next_observation

            # monitor actions, states and rewards are not constant
            rewards.append(reward)
            actions.append(current_action)
            if not (next_observation[0] == current_observation[0] and next_observation[1] == current_observation[1]):
                changes_in_state = changes_in_state + 1

            return_of_episode += reward

            # break while-loop when end of this episode
            if termination_flag:  # if done
                step_counter = step_id
                steps_counter_list.append(step_id)
                # return_list.append(return_of_episode)
                break

        # decay epsilon
        greedy_epsilon = max(eps_end, eps_decay * greedy_epsilon)

        # final state
        trajectory.append(next_observation)

        # log
        returns_window.append(return_of_episode)  # save most recent score
        if episode_id % 100 == 0:
            time_intermediate = time.time()
            print('\n --- Episode={} ---\n eps={}\n Average Score= {:.2f} \n duration={:.2f}'.format(
                episode_id, greedy_epsilon, np.mean(returns_window), time_intermediate - time_start))

        if episode_id % 20 == 0:
            print('Episode {} / {}. Eps = {}. Total_steps = {}. Return = {}. Max return = {}, Top 10 = {}'.format(
                episode_id+1, max_nb_episodes, greedy_epsilon, step_counter, return_of_episode, max_return,
                sorted(returns_list, reverse=True)[:10]))

        returns_list.append(return_of_episode)
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
    # Done subsequently in agent.save_q_table(folder)?
    # --

    print('End of training')
    print('Best return : %s --- with %s different trajectory(ies)' % (max_return, len(best_trajectories_list)))
    for trajectory in best_trajectories_list:
        print(trajectory)

    if using_tkinter:
        env.destroy()

    return returns_list, window_success, steps_counter_list


def display_results(agent, method_used_to_plot, returns_to_plot, smoothing_window, steps_counter_list_to_plot):

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
    plt.xlabel("Episode")
    plt.ylabel("Episode Return(Smoothed)")
    plt.title("Episode Return over Time (Smoothed over window size {})".format(smoothing_window))
    plt.savefig(folder + "return.png")
    plt.show()

    plt.hist(returns_to_plot, normed=True, bins=range(min(returns_to_plot), max(returns_to_plot) + 1, 1))
    plt.ylabel('reward distribution')
    plt.show()

    # Plot policy
    if method_used_to_plot == "q_approximation":
        agent.create_q_table()
    if method_used_to_plot != "DQN":
        agent.print_q_table()
        agent.plot_q_table(folder)
        agent.plot_optimal_actions_at_each_position(folder)
        agent.save_q_table(folder)


def test_agent(nb_training_episodes=20, sleep_time=0.001):
    # ToDo: load weights and show one run
    # Done subsequently in agent.save_q_table(folder)?
    # --
    print(nb_training_episodes)
    print(sleep_time)
    print("to be implemented")
    pass


if __name__ == "__main__":
    actions_list = ["no_change", "speed_up", "speed_up_up", "slow_down", "slow_down_down"]
    state_features_list = ["position", "velocity"]  #, "obstacle_position"]

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
                       "colour_velocity_code", "origin_coord", "display_canvas", "origin", "_last_child_ids", "rect"]
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
    flag_testing = True

    if flag_training:
        # training parameters
        eps_start_learning = 1.0
        eps_end_training = 0.01
        # 0.99907 for 5000 at 0.01/1.0
        eps_decay_training = 0.99907  # reach eps_end at episode_id = log10(eps_end/eps_start) / log10(eps_decay)
        # to reach eps_end at episode episode_id, eps_decay = (eps_end / eps_start) ** (1/episode_id)
        max_nb_episodes_training = 10000
        max_nb_steps_training = 25
        sleep_time_between_steps = 0.0005

        # after = Register an alarm callback that is called after a given time.
        # giving the reference to the function as well as the parameter
        if flag_tkinter:
            # (time_delay, method_to_execute):
            returns_list_res, window_success_res, steps_counter_list_res = env.after(
                100, train_agent, flag_tkinter, brain_agent, method_used,
                eps_start_learning, eps_end_training, eps_decay_training, max_nb_episodes_training,
                max_nb_steps_training, sleep_time_between_steps)
            env.mainloop()
        else:
            returns_list_res, window_success_res, steps_counter_list_res = train_agent(
                flag_tkinter, brain_agent, method_used, eps_start_learning, eps_end_training, eps_decay_training,
                max_nb_episodes_training, max_nb_steps_training, sleep_time_between_steps)
        display_results(brain_agent, method_used, returns_list_res, window_success_res, steps_counter_list_res)

    if flag_testing:
        max_nb_episodes_testing = 5
        sleep_time_between_steps = 0.5  # slow to see the steps
        if flag_tkinter:
            env.after(100, test_agent, flag_tkinter, brain_agent, method_used, max_nb_episodes_testing,
                      sleep_time_between_steps)
            env.mainloop()
        else:
            test_agent()
