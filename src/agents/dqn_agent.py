import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim

from agents.base_agent import BaseAgent
from neural.networks import DQN


class DQNAgent(BaseAgent):
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"--- DQN AGENT na: {self.device} ---")

        self.policy_net = DQN(state_size, action_size).to(self.device)
        self.target_net = DQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.0001)
        self.memory = deque(maxlen=100000)
        self.batch_size = 128
        self.gamma = 0.99

    def get_action(self, state, epsilon=0.0):
        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)
        with torch.no_grad():
            state = state.to(self.device)
            q_values = self.policy_net(state)
            return q_values.max(1)[1].item()

    def update(self, state, action, reward, next_state, done):
        if isinstance(state, torch.Tensor):
            state = state.cpu()
        if isinstance(next_state, torch.Tensor):
            next_state = next_state.cpu()
        self.memory.append((state, action, reward, next_state, done))

        if len(self.memory) < 1000:
            return

        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.cat(states).to(self.device)
        next_states = torch.cat(next_states).to(self.device)
        actions = torch.tensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.tensor(rewards).float().to(self.device)
        dones = torch.tensor(dones).float().to(self.device)

        current_q = self.policy_net(states).gather(1, actions).squeeze(1)
        next_q = self.target_net(next_states).max(1)[0]
        expected_q = rewards + (1 - dones) * self.gamma * next_q

        loss = nn.SmoothL1Loss()(current_q, expected_q.detach())
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
