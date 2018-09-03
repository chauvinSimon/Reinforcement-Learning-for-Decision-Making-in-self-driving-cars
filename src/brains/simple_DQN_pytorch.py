# the definition of the Agent

import numpy as np
import random
from collections import namedtuple, deque

from model_pytorch import QNetwork

import torch
import torch.nn.functional as nn_functional
import torch.optim as torch_optimization

BUFFER_SIZE = int(1e1)  # replay buffer size  # int(1e5)
BATCH_SIZE = 4         # mini-batch size  # 64
GAMMA = 0.9             # discount factor  # 0.99
TAU = 1e-3              # for soft update of target parameters  # 1e-3
LR = 5e-6               # learning rate # 5e-4
UPDATE_EVERY = 4        # how often to update the network  # 4

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Agent:
    """Interacts with and learns from the environment."""

    def __init__(self, actions, state, seed=0):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.actions_list = actions
        self.state_features_list = state
        state_size = len(state)
        action_size = len(actions)

        self.action_size = action_size
        self.state_size = state_size
        self.seed = random.seed(seed)

        # define Q-Network
        self.q_network_local = QNetwork(state_size, action_size, seed).to(device)
        self.q_network_target = QNetwork(state_size, action_size, seed).to(device)
        self.optimizer = torch_optimization.Adam(self.q_network_local.parameters(), lr=LR)

        # define Replay Memory (uniform)
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
    
    def step(self, state, action, reward, next_state, done):
        """ Save a transition """
        action_id = self.actions_list.index(action)

        # normalize
        # state[0] = state[0] / 20
        # state[1] = state[1] / 5
        # next_state[0] = next_state[0] / 20
        # next_state[1] = next_state[1] / 5

        # Save experience in replay memory
        self.memory.add(state, action_id, reward, next_state, done)
        
        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > BATCH_SIZE:
                # print("get random subset and learn")
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA)

    def choose_action(self, state, masked_actions_list, greedy_epsilon):  # act()
        """
        Returns actions for given state as per current policy.
        Read

        :param state: (array_like): current state
        :param eps: (float): epsilon, for epsilon-greedy action selection
        :return: the int index of the action - in range(action_state_size)
        """
        # Creates a Tensor from a numpy.ndarray
        state = np.array(state)
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.q_network_local.eval()  # needed?
        with torch.no_grad():
            action_values = self.q_network_local(state)
        self.q_network_local.train()  # needed?

        possible_actions = [action for action in self.actions_list if action not in masked_actions_list]

        # Epsilon-greedy action selection
        if random.random() > greedy_epsilon:

            # Retrieve a tensor held by the Variable action_values.cpu(), using the .data attribute
            actions_values = action_values.cpu().data.numpy()
            actions_values = actions_values[0]
            # print("action_values.cpu().data.numpy() = {}".format(actions_values))
            # print("possible_actions = {}".format(possible_actions))

            for action in self.actions_list:
                if action not in possible_actions:
                    action_id = self.actions_list.index(action)
                    actions_values[action_id] = -np.inf

            # make decision
            if np.all(np.isneginf([actions_values])):
                action_id = random.choice(possible_actions)
                print('random action sampled among allowed actions')
            else:
                action_id = np.argmax(actions_values)
            selected_action = self.actions_list[action_id]
        else:
            # action_id = random.choice(np.arange(self.action_size))
            selected_action = random.choice(possible_actions)

        # print("selected_action = {}".format(selected_action))

        return selected_action

    def compare_reference_value(self):
        # ToDo: we know the value of the last-but one state at convergence: Q(s,a)=R(s,a).
        state = [16, 3]
        action_id = 0
        state = np.array(state)
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.q_network_local.eval()  # needed?
        with torch.no_grad():
            action_values = self.q_network_local(state)

        actions_values = action_values.cpu().data.numpy()
        actions_values = actions_values[0]

        print(actions_values)
        return action_values[action_id]

    def learn(self, experiences, gamma):
        """
        Update value parameters using given batch of experience tuples.

        :param experiences: (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
        :param gamma: (float): discount factor
        :return: -
        """
        # print("experiences = {}".format(experiences))

        states, actions, rewards, next_states, dones = experiences
        # Get max predicted Q values (for next states) from target model
        q_targets_next = self.q_network_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        q_targets = rewards + (gamma * q_targets_next * (1 - dones))

        # Get expected Q values from local model
        q_expected = self.q_network_local(states).gather(1, actions)

        # Compute loss
        loss = nn_functional.mse_loss(q_expected, q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.q_network_local, self.q_network_target, TAU)

    @staticmethod
    def soft_update(local_model, target_model, tau):
        """
        By reference, soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        :param local_model: (PyTorch model): weights will be copied from
        :param target_model: (PyTorch model): weights will be copied to
        :param tau: (float): interpolation parameter
        :return:
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """
        Initialize a ReplayBuffer object.

        Params
        ======
        :param action_size: (int): dimension of each action
        :param buffer_size: (int): maximum size of buffer
        :param batch_size: (int): size of each training batch
        :param seed: (int): random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)
