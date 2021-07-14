import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class Network(nn.Module):

    def __init__(self, lr, input_dims, n_actions):
        super(Network, self).__init__()
        self.lr = lr
        self.input_dims = input_dims
        self.n_actions = n_actions

        # Create network layers
        self.fc1 = nn.Linear(self.input_dims, 8)
        self.fc2 = nn.Linear(8, 16)
        self.fc3 = nn.Linear(16, self.n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=self.lr)
        self.loss = nn.MSELoss()

        self.device = T.device("cude:0" if T.cuda.is_available() else "cpu")
        print(f"Created network with device {self.device}")
        self.to(self.device)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x


class DeepQAgent():

    def __init__(self, gamma, epsilon, lr, input_dims, batch_size,
                 n_actions, memory_size=1000, epsilon_min=0.01, eps_dec=5e-4,
                 update_target_frequency=1000):
        self.gamma = gamma
        self.epsilon = epsilon  # explore chance
        self.lr = lr
        self.input_dims = input_dims
        self.epsilon_min = epsilon_min
        self.eps_dec = eps_dec
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.n_actions = n_actions
        self.update_target_frequency = update_target_frequency

        self.update_target_counter = 0
        self.action_space = [i for i in range(n_actions)]
        self.memory_counter = 0
        self.state_memory = np.zeros(
            (self.memory_size, self.input_dims), dtype=np.float32)
        self.new_state_memory = np.zeros(
            (self.memory_size, self.input_dims), dtype=np.float32)
        self.action_memory = np.zeros(self.memory_size, dtype=np.int32)
        self.reward_memory = np.zeros(self.memory_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.memory_size, dtype=np.bool)

        self.model = Network(self.lr, self.input_dims, self.n_actions)
        self.target_model = Network(self.lr, self.input_dims, self.n_actions)
        self.target_model.load_state_dict(self.model.state_dict())

    def load_from_file(self, file_name):
        self.model.load_state_dict(T.load(file_name))
        self.target_model.load_state_dict(self.model.state_dict())

    def store_transition(self, state, action, reward, new_state, done):
        index = self.memory_counter % self.memory_size
        self.state_memory[index] = state
        self.new_state_memory[index] = new_state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.terminal_memory[index] = done

        self.memory_counter += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.tensor([observation]).to(self.target_model.device)
            actions = self.target_model.forward(state)
            action = T.argmax(actions).item()
        else:
            action = np.random.choice(self.action_space)

        return action

    def learn(self):
        if self.memory_counter < self.batch_size:
            return

        self.model.optimizer.zero_grad()
        self.target_model.optimizer.zero_grad()

        current_memory_size = min(self.memory_size, self.memory_counter)
        batch = np.random.choice(
            current_memory_size, self.batch_size, replace=False)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        state_batch = T.tensor(self.state_memory[batch]).to(self.model.device)
        new_state_batch = T.tensor(
            self.new_state_memory[batch]).to(self.model.device)
        reward_batch = T.tensor(
            self.reward_memory[batch]).to(self.model.device)
        terminal_batch = T.tensor(
            self.terminal_memory[batch]).to(self.model.device)
        action_batch = self.action_memory[batch]

        # Select for update only those actions, that were taken
        current_q = self.model.forward(state_batch)[batch_index, action_batch]
        next_q = self.target_model.forward(new_state_batch)
        next_q[terminal_batch] = 0.0

        target_q = reward_batch + self.gamma * T.max(next_q, dim=1)[0]
        loss = self.target_model.loss(
            target_q, current_q).to(self.model.device)
        loss.backward()
        self.model.optimizer.step()

        self.epsilon = self.epsilon - \
            self.eps_dec if self.epsilon > self.epsilon_min else self.epsilon_min

        self.update_target_counter += 1
        if self.update_target_counter > self.update_target_frequency:
            self.update_target_counter = 0
            self.target_model.load_state_dict(self.model.state_dict())
