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

plt.rcParams['figure.figsize'] = [20, 10]


def train_agent(using_tkinter, agent, method, max_nb_episodes=2000, sleep_time=0.001):
    """

    :param agent: brain object
    :param method: value-based learning method - either sarsa or q-learning
    :param max_nb_episodes: [int] limit of training episodes
    :param sleep_time: [ms] sleep_time between two steps
    :param using_tkinter: [bool] to display the environment
    :return:
    """
    window_success = 100
    eps = 1
    total_steps = 0
    steps_counter_list = []
    return_list = []
    returns_list = []
    max_return = -10000  # to be set low enough (basically nb_step * max_cost should do it)
    best_trajectories_list = []
    action = ""
    threshold_success = 10  # to solve the env, = average score over the last x scores, where x = window_success
    scores_window = deque(maxlen=window_success)  # last x scores, where x = window_success
    time_start = time.time()

    for episode in range(max_nb_episodes):
        # initial observation = initial state
        observation, masked_actions_list = env.reset()
        step_counter = 0
        the_return = 0
        trajectory = []

        if method == "sarsa":
            # agent choose action based on observation
            action = agent.choose_action(observation, masked_actions_list)

        if method == "sarsa_lambda":
            # agent choose action based on observation
            action = agent.choose_action(observation, masked_actions_list)

            # initial all zero eligibility trace
            agent.reset_eligibility_trace()

        while True:  # similar to "for t in itertools.count():"
            step_counter += 1
            total_steps += 1

            # fresh env
            if using_tkinter:
                env.render(sleep_time)

            if (method == "sarsa") or (method == "sarsa_lambda"):
                observation_, reward, termination_flag, masked_actions_list = env.step(action)

                # agent choose action based on observation
                action_ = agent.choose_action(observation, masked_actions_list)
                trajectory.append(observation)
                trajectory.append(action)

                # agent learn from this transition
                agent.learn(observation, action, reward, observation_, action_, termination_flag)
                action = action_

            else:  # Q-learning (classic and with approximation) or DQN
                # agent choose action based on observation
                action = agent.choose_action(observation, masked_actions_list)
                trajectory.append(observation)
                trajectory.append(action)
                # print("action selected = %s" % action)

                # agent take action and get next observation (~state) and reward.
                # Also the masked actions for the next step
                observation_, reward, termination_flag, masked_actions_list = env.step(action)

                if method == "q":
                    # agent learn from this transition
                    agent.learn(observation, action, reward, observation_, termination_flag)

                elif method == "q_approximation":
                    # Update the function approximator using our target
                    # estimator.update(state, action, td_target)
                    agent.learn(observation, action, reward, observation_, termination_flag)

                else:  # DQN
                    # New: store transition in memory - subsequently to be sampled from
                    agent.store_transition(observation, action, reward, observation_)

                    # if the number of steps is larger than a threshold, start learn ()
                    # if (total_steps > 200) and (total_steps % 5 == 0):  # for 1 to T
                    if (step_counter > 5) and (step_counter % 5 == 0):  # for 1 to T
                        # print('learning')
                        # pick up some transitions from the memory and learn from these samples
                        agent.learn()

            # update state
            observation = observation_

            the_return += reward

            # break while-loop when end of this episode
            if termination_flag:
                steps_counter_list.append(step_counter)
                return_list.append(the_return)
                break

        # final state
        trajectory.append(observation_)

        # log
        scores_window.append(the_return)  # save most recent score
        if episode % 100 == 0:
            time_intermediate = time.time()
            print('\n --- Episode={} ---\n eps={}\n Average Score= {:.2f} \n duration={:.2f}'.format(
                episode, eps, np.mean(scores_window), time_intermediate - time_start))

        if episode % 20 == 0:
            print('Episode %s / %s. Total_steps = %s. Return = %s. Max return so far = %s, List of top 10 = %s' % (
                episode+1, max_nb_episodes, step_counter, the_return, max_return, sorted(returns_list,
                                                                                         reverse=True)[:10]))
        returns_list.append(the_return)
        if the_return == max_return:
            if trajectory not in best_trajectories_list:
                best_trajectories_list.append(trajectory)
        elif the_return > max_return:
            del best_trajectories_list[:]
            best_trajectories_list.append(trajectory)
            max_return = the_return

        if np.mean(scores_window) >= threshold_success:
            time_stop = time.time()
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}, duration={:.2f} [s]'.format(
                episode - window_success, np.mean(scores_window), time_stop - time_start))
            break

    print('End of training')
    print('Best return : %s --- with %s different trajectory(ies)' % (max_return, len(best_trajectories_list)))
    for trajectory in best_trajectories_list:
        print(trajectory)

    # Plot parameters
    smoothing_window = window_success

    # where to save the plots
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    folder = os.path.join(parent_dir, "results/simple_road/")

    # plot step_counter for each episode
    plt.grid(True)
    plt.xlabel('Episode')
    plt.title("Episode Step_counts over Time (Smoothed over window size {})".format(smoothing_window))
    plt.ylabel("Episode step_count (Smoothed)")
    steps_smoothed = pd.Series(steps_counter_list).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(steps_counter_list, linewidth=0.5)
    plt.plot(steps_smoothed, linewidth=2.0)
    plt.savefig(folder + "step_counter.png")
    plt.show()

    plt.grid(True)
    returns_smoothed = pd.Series(return_list).rolling(smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(return_list, linewidth=0.5)
    plt.plot(returns_smoothed, linewidth=2.0)
    plt.xlabel("Episode")
    plt.ylabel("Episode Return(Smoothed)")
    plt.title("Episode Return over Time (Smoothed over window size {})".format(smoothing_window))
    plt.savefig(folder + "return.png")
    plt.show()

    plt.hist(returns_list, normed=True, bins=range(min(returns_list), max(returns_list) + 1, 1))
    plt.ylabel('reward distribution')
    plt.show()

    # Plot policy
    if method_used == "q_approximation":
        agent.create_q_table()
    if method_used != "DQN":
        agent.print_q_table()
        agent.plot_q_table(folder)
        agent.plot_optimal_actions_at_each_position(folder)
        agent.save_q_table(folder)

    if using_tkinter:
        env.destroy()


def test_agent(nb_training_episodes=20, sleep_time=0.001):
    print(nb_training_episodes)
    print(sleep_time)
    print("to be implemented")
    pass


if __name__ == "__main__":
    actions_list = ["no_change", "speed_up", "speed_up_up", "slow_down", "slow_down_down"]
    state_features_list = ["position", "velocity"]

    # the environment
    flag_tkinter = False
    initial_state = [0, 3]
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
                                   e_greedy=0.9,
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
        max_nb_episodes_training = 2000
        sleep_time_between_steps = 0.0005

        # after = Register an alarm callback that is called after a given time.
        # giving the reference to the function as well as the parameter
        if flag_tkinter:
            # (time_delay, method_to_execute):
            env.after(100, train_agent, flag_tkinter, brain_agent, method_used, max_nb_episodes_training,
                      sleep_time_between_steps)
            env.mainloop()
        else:
            train_agent(flag_tkinter, brain_agent, method_used, max_nb_episodes_training, sleep_time_between_steps)

    if flag_testing:
        max_nb_episodes_testing = 5
        sleep_time_between_steps = 0.5  # slow to see the steps
        if flag_tkinter:
            env.after(100, test_agent, flag_tkinter, brain_agent, method_used, max_nb_episodes_testing,
                      sleep_time_between_steps)
            env.mainloop()
        else:
            test_agent()
