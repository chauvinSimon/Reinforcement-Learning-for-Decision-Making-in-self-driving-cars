"""
This script is the environment part of this example
In particular:

-1- transition model
-	x(t+1) = x(t) + vx(t)∆t
-	vx(t+1) = vx(t) + a(t)∆t

-2- reward function with
(The weighting may change depending on the aggressiveness of the driver)
 -- - time efficiency = Progress
 -- v -- reaching goal = global goal indicators
 -- v -- reaching velocity goal
 -- v -- per-step cost - This term discourages the driver from making unnecessary maneuvers
 -- v -- under-speed

 -- - traffic law
 -- v -- over-speed
 -- - -- stop at stop / red light
 -- - -- yield to right

 -- - safety = sparse constraint violation alerts
 -- - -- distance to other vehicles
 -- - -- h is headway distance - The range to the car directly in front ([-1; 0; 1 depending on close; nominal; far])
 -- - -- Time-to-Collision (TTC)?
 -- - -- speed difference with other vehicles
 -- - -- speed at intersection
 -- v -- speed near obstacles
 -- - -- crash
 -- - -- braking distance

 -- - comfort and effort
 -- - --  include a cost of changing the desire in the reward function
 if changing the
 -- v --  change in actions (esp. avoid stops and sudden braking)
 -- - --  change in acc = jerk

Not covered yet:
 -- - -- traffic disturbance
 -- - -- include also constraints violation (time spent on opposite lane)
 -- - -- fuel consumption
 -- - -- Apply rewards that consider all the over-flight states
 (otherwise, you can jump to escape the pedestrian constrain)
 -- ~v -- Include previous state for that

-3- hard constraints
- imposed before action selection. Should a security check be implemented after the selection?
YES, when selecting the final trajectory
- no acceleration allowed if that leads to v>v_max or deceleration that would cause v<0
- Impose hard constraints to avoid the occurrence of such combinations:
-	1) If a car in the left lane is in a parallel position, the controlled car cannot change lane to the left
-	2) If a car in the right lane is in a parallel position, the controlled car cannot change lane to the right
-	3) If a car in the left lane is “close” and “approaching,” the controlled car cannot change lane to the left
-	4) If a car in the right lane is “close” and “approaching,” the controlled car cannot change lane to the right.
- The use of these hard constrains eliminates the clearly undesirable behaviors
better than through penalizing them in the reward function
- It increases the learning speed during training
- Also mask some actions:
 -- - -- put a mask on the Q-value associated with the left action such that it is never selected in such a state (if
already at max left).
 -- v --  if the ego car is driving at the maximum speed then the accelerate action is masked

To do:
- selecting the final trajectory

The RL is in RL_brain.py.
"""

import time
import numpy as np  # but trying to avoid using it (np.array cannot be converted to JSON)
# import sys
# if sys.version_info.major == 2:
#     import Tkinter as tk
# else:
import tkinter as tk
import random
from utils.logger import Logger

UNIT = 20   # pixels per grid cell
MAZE_H = 4  # grid height
MAZE_W = 20  # grid width  !! Adapt the threshold_success in the main accordingly
HALF_SIZE = UNIT * 0.35  # half size factor of square in a cell
Y_COORD = 0  # blocking motion of ego agent in one row - no vertical motion allowed


def one_hot_encoding(feature_to_encode):
    """

    :param feature_to_encode: int. For instance 5
    :return: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0] if MAZE_W=10
    """
    repr_size = MAZE_W
    one_hot_list = np.zeros(repr_size)
    # one_hot_list = [0] * UNIT
    if feature_to_encode < repr_size:
        one_hot_list[feature_to_encode] = 1
    else:
        print('feature is out of scope for one_hot_encoding: %i / %i' % (feature_to_encode, repr_size))
    # print(one_hot_list)
    return one_hot_list


def process_state(input_state):
    """
    extract features from the state to build an "observation"
    The agent may not know exactly its absolute position.
    But it may have an estimate of the relative distance to the obstacle
    Hence compute the difference in position
    :param input_state:
    :return: representation of the state understood by the RL agent
    """
    ego_position = input_state[0]
    velocity = input_state[1]
    # obstacle_position = input_state[2]
    #
    # # one-hot encoding of the state
    # repr_size = MAZE_W
    # encoded_position = one_hot_encoding(ego_position)
    # encoded_velocity = one_hot_encoding(velocity)
    #
    # one_hot_state = np.row_stack((encoded_position, encoded_velocity))

    # Filling the state representation with other rows
    # one_hot_state = np.row_stack((one_hot_state, np.zeros((repr_size, 82))))
    # nb_iter = repr_size - np.shape(one_hot_state)[0]
    # for _ in range(nb_iter):
    #     one_hot_state = np.row_stack((one_hot_state, np.zeros(repr_size)))

    # print("one_hot_state has shape =")
    # print(np.shape(one_hot_state))  # Here one_hot_state is (84, 84)

    # ToDo: increase state for the brain
    return [ego_position, velocity]  # , obstacle_position]

    # make one_hot_state have mean 0 and a variance of 1
    # print(one_hot_state)
    # print(np.mean(one_hot_state))
    # print(np.var(one_hot_state))
    # one_hot_state = (one_hot_state - np.mean(one_hot_state)) / ((np.var(one_hot_state))**0.5)
    # print(np.var(one_hot_state))
    # print(np.mean(one_hot_state))
    # print(one_hot_state)
    # return one_hot_state


# !! Depending is tk is supported or not, manually change the inheritance
# !! uncomment the next line and comment the two next
# class Road:  # if tk is NOT supported. Then make sure using_tkinter=False
class Road(tk.Tk, object):  # if tk is supported
    def __init__(self, using_tkinter, actions_names, state_features, initial_state, goal_velocity=4):
        """

        :param using_tkinter: [bool] flag for the graphical interface
        :param actions_names: [string] list of possible actions
        :param state_features: [string] list of features forming the state. sting !!
        :param initial_state: [list of int]
        :param goal_velocity: [int]
        """
        # graphical interface
        if using_tkinter:
            super(Road, self).__init__()

        # action space
        self.actions_list = actions_names

        # state is composed of
        # - absolute ego_position
        # - velocity
        # - absolute position of obstacle
        self.state_features = state_features
        # print("state_features = {}".format(state_features))

        # Reward - the reward is update
        # - during the transition (hard-constraints)
        # - in the reward_function (only considering the new state)
        self.reward = 0
        self.rewards_dict = {
            # efficiency = Progress
            "goal_with_good_velocity": 40,
            "goal_with_bad_velocity": -40,
            "per_step_cost": -3,
            "under_speed": -15,

            # traffic law
            "over_speed": -10,
            "over_speed_2": -10,

            # safety
            "over_speed_near_pedestrian": -40,

            # Comfort
            "negative_speed": -15,
            "action_change": -2
        }

        self.max_velocity_1 = 4
        self.max_velocity_2 = 2
        self.max_velocity_pedestrian = 2
        self.min_velocity = 0

        # state - for the moment distinct variables
        self.initial_state = initial_state
        self.state_ego_position = self.initial_state[0]
        self.state_ego_velocity = self.initial_state[1]
        self.state_obstacle_position = self.initial_state[2]
        self.previous_state_position = self.state_ego_position
        self.previous_state_velocity = self.state_ego_velocity
        self.previous_action = None

        # environment:
        self.goal_coord = [MAZE_W - 1, 1]
        self.goal_velocity = goal_velocity
        self.obstacle1_coord = [self.state_obstacle_position, 2]
        self.obstacle2_coord = [1, 3]
        self.initial_position = [self.initial_state[0], Y_COORD]
        # self.goal_coord = np.array([MAZE_W - 1, 1])
        # self.obstacle1_coord = np.array([12, 2])
        # self.obstacle2_coord = np.array([1, 3])

        # adjusting the colour of the agent depending on its speed
        colours_list = ["white", "yellow", "orange", "red2", "red3", "red4", "black", "black", "black", "black",
                        "black", "black"]
        velocity_list = range(len(colours_list)+1)
        self.colour_velocity_code = dict(zip(velocity_list, colours_list))

        # graphical interface
        self.using_tkinter = using_tkinter
        # create the origin point in  the Tk frame
        self.origin_coord = [(x + y) for x, y in zip(self.initial_position, [0.5, 0.5])]
        # self.origin = UNIT * self.origin_coord
        self.origin = [x * UNIT for x in self.origin_coord]
        self.canvas = None
        self.rect = None
        self.obstacle = None

        # self.rect = self.canvas.create_rectangle(
        #     self.origin[0] - HALF_SIZE, self.origin[1] - HALF_SIZE,
        #     self.origin[0] + HALF_SIZE, self.origin[1] + HALF_SIZE,
        #     fill='red')

        if self.using_tkinter:
            # Tk window
            self.title('road')
            self.geometry('{0}x{1}'.format(MAZE_W * UNIT, MAZE_H * UNIT))
            self.build_road()

        # logging configuration
        self.logger = Logger("road", "road_env.log", 0)

    @staticmethod
    def sample_position_obstacle():
        fix_position_obstacle = 12
        return fix_position_obstacle
        # random_position_obstacle = random.randint(1, MAZE_W)
        # random_position_obstacle = random.randint(MAZE_W//2 - 1, MAZE_W//2 + 2)
        # print("{} = random_position_obstacle".format(random_position_obstacle))
        # return random_position_obstacle

    def build_road(self):
        """
        To build the Tk window
        Only called once at the start
        :return:  None
        """
        if self.using_tkinter:

            # create canvas
            self.canvas = tk.Canvas(self, bg='white', height=MAZE_H * UNIT, width=MAZE_W * UNIT)

            # create grids
            for c in range(0, MAZE_W * UNIT, UNIT):
                x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
                self.canvas.create_line(x0, y0, x1, y1)
            for r in range(0, MAZE_H * UNIT, UNIT):
                x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
                self.canvas.create_line(x0, y0, x1, y1)

            # create ego agent
            self.rect = self.canvas.create_rectangle(
                self.origin[0] - HALF_SIZE, self.origin[1] - HALF_SIZE,
                self.origin[0] + HALF_SIZE, self.origin[1] + HALF_SIZE,
                fill='white')

            # obstacle1
            obstacle1_center = np.asarray(self.origin) + UNIT * np.asarray(self.obstacle1_coord)
            self.obstacle = self.canvas.create_rectangle(
                obstacle1_center[0] - HALF_SIZE, obstacle1_center[1] - HALF_SIZE,
                obstacle1_center[0] + HALF_SIZE, obstacle1_center[1] + HALF_SIZE,
                fill='black')

            # obstacle2
            obstacle2_center = np.asarray(self.origin) + UNIT * np.asarray(self.obstacle2_coord)
            self.canvas.create_rectangle(
                obstacle2_center[0] - HALF_SIZE, obstacle2_center[1] - HALF_SIZE,
                obstacle2_center[0] + HALF_SIZE, obstacle2_center[1] + HALF_SIZE,
                fill='black')

            # create oval for the goal
            goal_center = np.asarray(self.origin) + UNIT * np.asarray(self.goal_coord)
            self.canvas.create_oval(
                goal_center[0] - HALF_SIZE, goal_center[1] - HALF_SIZE,
                goal_center[0] + HALF_SIZE, goal_center[1] + HALF_SIZE,
                fill='yellow')

            # pack all
            self.canvas.pack()

    def reset(self):
        """
        Clean the canvas (remove agent)
        Clean the state (reinitialize it)
        Sample a random position for the obstacle
        :return: the initial state amd the list of the masked actions
        """
        # self.update() - Not necessary?
        time.sleep(0.005)
        random_position_obstacle = self.sample_position_obstacle()
        self.obstacle1_coord = [random_position_obstacle, 2]

        if self.using_tkinter:
            self.canvas.delete(self.rect)
            self.rect = self.canvas.create_rectangle(
                self.origin[0] - HALF_SIZE, self.origin[1] - HALF_SIZE,
                self.origin[0] + HALF_SIZE, self.origin[1] + HALF_SIZE,
                fill='white')

            self.canvas.delete(self.obstacle)
            obstacle1_center = np.asarray(self.origin) + UNIT * np.asarray(self.obstacle1_coord)
            self.obstacle = self.canvas.create_rectangle(
                obstacle1_center[0] - HALF_SIZE, obstacle1_center[1] - HALF_SIZE,
                obstacle1_center[0] + HALF_SIZE, obstacle1_center[1] + HALF_SIZE,
                fill='black')

        self.state_ego_position = self.initial_state[0]
        self.state_ego_velocity = self.initial_state[1]
        self.state_obstacle_position = random_position_obstacle
        self.initial_state[2] = random_position_obstacle
        self.previous_state_position = self.initial_state[0]
        self.previous_state_velocity = self.initial_state[1]

        return process_state(self.initial_state), self.masking_function()

    def transition(self, action):
        """
        update velocity in state according to the desired command
        :param action: desired action
        :return:
        """
        delta_velocity = 0

        # print("current velocity: %s" % self.state_ego_velocity)
        if action == self.actions_list[0]:  # maintain velocity
            delta_velocity = 0
        elif action == self.actions_list[1]:  # accelerate
            delta_velocity = 1
        elif action == self.actions_list[2]:  # accelerate a lot
            delta_velocity = 2
        elif action == self.actions_list[3]:  # slow down
            delta_velocity = -1
        elif action == self.actions_list[4]:  # slow down a lot
            delta_velocity = -2
        # print("new velocity: %s" % self.state_ego_velocity)

        return self.state_ego_velocity + delta_velocity

    def step(self, action):
        """
        Transforms the action into the new state
        -calls the transition model
        -calls checks hard conditions
        -calls masking - to be implemented

        :param action: [string] the desired action
        :return: tuple with:
        -new state (list)
        -reward (int)
        -termination_flag (bool)
        -masked_actions_list (list)
        """
        # print("desired action: %s" % action)

        # reminding the previous state
        self.previous_state_velocity = self.state_ego_velocity
        self.previous_state_position = self.state_ego_position
        self.previous_action = action

        # Transition = get the new state:
        self.state_ego_velocity = self.transition(action)
        if self.state_ego_velocity < 0:
            self.state_ego_velocity = 0
            message = "self.state_ego_velocity cannot be < 0 - a = {} - p = {} - v = {} " \
                      "in step()".format(action, self.state_ego_position, self.state_ego_velocity)
            self.logger.log(message, 3)

        # Assume simple relation: velocity expressed in [step/sec] and time step = 1s
        desired_position_change = self.state_ego_velocity

        # convert information from velocity to the change in position = number of steps
        tk_update_steps = [0, 0]

        # update the state - position
        # print("old position: %s" % self.state_ego_position)
        self.state_ego_position = self.state_ego_position + desired_position_change
        tk_update_steps[0] += desired_position_change * UNIT

        # print("new position: %s" % self.state_ego_position)

        if self.using_tkinter:
            # move agent in canvas
            self.canvas.move(self.rect, tk_update_steps[0], tk_update_steps[1])
            # update colour depending on speed
            # print("self.state_ego_velocity = {}".format(self.state_ego_velocity))
            new_colour = self.colour_velocity_code[self.state_ego_velocity]
            self.canvas.itemconfig(self.rect, fill=new_colour)

        # observe reward
        [reward, termination_flag] = self.reward_function(action)

        # for the next decision, these actions are not possible (it uses the output state):
        if termination_flag:
            masked_actions_list = []
        else:
            masked_actions_list = self.masking_function()

        state_to_return = process_state(
            [self.state_ego_position, self.state_ego_velocity, self.state_obstacle_position]
        )
        return state_to_return, reward, termination_flag, masked_actions_list

    def reward_function(self, action):
        """
        ToDo: normalize it
        To be refined
        - it needs to consider all the intermediate points between previous state and new state
        :return: the reward (int) and termination_flag (bool)
        """
        # reward put to for the new step
        self.reward = 0

        # penalizing changes in action
        # it penalizes big changes (e.g. from speed_up_up to slow_down_down)
        if self.state_ego_velocity != self.previous_state_velocity:
            change_in_velocity = self.state_ego_velocity - self.previous_state_velocity
            self.reward += self.rewards_dict["action_change"] * abs(change_in_velocity)

        # test about the position
        # - for the goal
        if self.state_ego_position >= self.goal_coord[0]:
            # not over-exceeding the goal
            self.state_ego_position = self.goal_coord[0]
            if self.state_ego_velocity == self.goal_velocity:
                self.reward += self.rewards_dict["goal_with_good_velocity"]
            else:
                self.reward += self.rewards_dict["goal_with_bad_velocity"]
            termination_flag = True

        # - for all other states
        else:
            self.reward += self.rewards_dict["per_step_cost"]
            termination_flag = False

        # check max speed limitation
        if self.state_ego_velocity > self.max_velocity_1:
            excess_in_velocity = self.state_ego_velocity - self.max_velocity_1
            self.reward += self.rewards_dict["over_speed"] * excess_in_velocity
            message = "Too fast! in reward_function() -- hard constraints should have masked it. " \
                      "a = {} - p = {} - v = {}".format(action, self.state_ego_position, self.state_ego_velocity)
            self.logger.log(message, 3)

        # check minimal speed
        if self.state_ego_velocity < self.min_velocity:
            excess_in_velocity = self.min_velocity - self.state_ego_velocity
            # well, basically, it will stay as rest - but still, we need to prevent negative speeds
            self.reward += self.rewards_dict["under_speed"] * excess_in_velocity

            if self.state_ego_velocity < 0:
                excess_in_velocity = abs(self.min_velocity)
                message = "Under speed! in reward_function() -- hard constraints should have masked it. " \
                          "a = {} - p = {} - v = {}".format(action, self.state_ego_position, self.state_ego_velocity)
                self.logger.log(message, 3)

                self.state_ego_velocity = 0
                self.reward += self.rewards_dict["negative_speed"] * excess_in_velocity

        # limit speed when driving close to a pedestrian
        if self.previous_state_position <= self.obstacle1_coord[0] <= self.state_ego_position:
            # print(self.previous_state_position)
            # print('passing the pedestrian')
            # print(self.state_ego_position)
            if self.state_ego_velocity > self.max_velocity_pedestrian:
                excess_in_velocity = self.state_ego_velocity - self.max_velocity_pedestrian
                self.reward += self.rewards_dict["over_speed_near_pedestrian"] * excess_in_velocity
                message = "Too fast close to obstacle! in reward_function() - a = {} - p = {} - po= {} - v = {}".format(
                    action, self.state_ego_position, self.state_obstacle_position, self.state_ego_velocity)
                self.logger.log(message, 1)

        # test about the velocity
        # if self.state_ego_position == self.pedestrian[0]:
        #     if self.state_ego_velocity == self.goal_coord[0]:
        #         self.reward += self.rewards_dict["goal"]

        # normalization
        # self.reward = 1 + self.reward / max(self.rewards_dict.values())

        return self.reward, termination_flag

    def masking_function(self):
        """
        hard constraints
        using the state (position, velocity)
        :return: masked_actions_list (a sub_list from self.action_list)
        """
        masked_actions_list = []

        # check if maximum / minimum speed has been reached
        for action_candidate in self.actions_list:
            # simulation for each action
            velocity_candidate = self.transition(action_candidate)
            # print(velocity_candidate)
            if velocity_candidate > self.max_velocity_1:
                # print("hard _ constraint : to fast")
                masked_actions_list.append(action_candidate)
            elif velocity_candidate < 0:
                # print("hard _ constraint : negative speed")
                masked_actions_list.append(action_candidate)

        # checking there are still possibilities left:
        if masked_actions_list == self.actions_list:
            print("velocity %s and position %s" % (self.state_ego_velocity, self.state_ego_position))
            message = "No possible_action found! in masking_function() - a = {} - p = {} - po= {} - v = {}".format(
                self.previous_action, self.state_ego_position, self.state_obstacle_position, self.state_ego_velocity)
            self.logger.log(message, 4)

        return masked_actions_list

    def render(self, sleep_time):
        """
        :param sleep_time: [float]
        necessary for demo()
        :return:
        """
        if self.using_tkinter:
            time.sleep(sleep_time)
            self.update()


def demo(actions, nb_episodes_demo):
    """
    Just used for the demo when running this single script
    No brain

    :param actions:
    :param nb_episodes_demo:
    :return:
    """
    for t in range(nb_episodes_demo):
        _, masked_actions_list = env.reset()
        print("New Episode")
        while True:
            sleep_time = 0.5
            if env.using_tkinter:
                env.render(sleep_time)

            # Pick randomly an action among non-masked actions

            # possible_actions = [action for action in actions]
            possible_actions = [action for action in actions if action not in masked_actions_list]
            if not possible_actions:
                print("!!!!! WARNING - No possible_action !!!!!")
            action = np.random.choice(possible_actions)

            # Give the action to the environment and observe new state and reward
            state, reward, termination_flag, masked_actions_list = env.step(action)
            print("Action=", action, " -- State=", state, " -- Reward=", reward, " -- Termination_flag=",
                  termination_flag, sep='')

            # Check end of episode
            if termination_flag:
                break


if __name__ == '__main__':
    flag_tkinter = True
    nb_episodes = 5
    actions_list = ["no_change", "speed_up", "speed_up_up", "slow_down", "slow_down_down"]
    state_features_list = ["position", "velocity"]
    the_initial_state = [0, 3, 12]
    env = Road(flag_tkinter, actions_list, state_features_list, the_initial_state)
    # Wait 100 ms and run several episodes
    if flag_tkinter:
        env.after(100, demo, actions_list, nb_episodes)
        env.mainloop()  # need to be close manually
    else:
        demo(actions_list, nb_episodes)
