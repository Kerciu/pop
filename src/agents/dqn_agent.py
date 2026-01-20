import random
from collections import deque

import torch
import torch.nn as nn
import torch.optim as optim


class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()

        self.feature_layer = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU()
        )

        # Value Stream (Ocenia jak dobry jest stan)
        self.value_stream = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1)
        )

        # Advantage Stream (Ocenia przewagę każdej akcji nad innymi)
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, output_dim)
        )

    def forward(self, x):
        features = self.feature_layer(x)
        values = self.value_stream(features)
        advantages = self.advantage_stream(features)
        # Combine: Q = V + (A - mean(A))
        return values + (advantages - advantages.mean(dim=1, keepdim=True))


class DQNAgent:
    def __init__(self, state_size, action_size, learning_rate=0.001):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=20000)
        self.gamma = 0.99
        self.batch_size = 64
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DuelingDQN(state_size, action_size).to(self.device)
        self.target_net = DuelingDQN(state_size, action_size).to(self.device)
        self.update_target_network()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()

    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def get_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_size - 1)

        with torch.no_grad():
            state = state.to(self.device)
            if state.dim() == 1:
                state = state.unsqueeze(0)
            q_values = self.policy_net(state)
            return torch.argmax(q_values).item()

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def update(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)

        states = torch.stack([x[0] for x in minibatch]).to(self.device)
        actions = torch.tensor([x[1] for x in minibatch]).to(self.device).unsqueeze(1)
        rewards = torch.tensor([x[2] for x in minibatch]).to(self.device)
        next_states = torch.stack([x[3] for x in minibatch]).to(self.device)
        dones = torch.tensor([x[4] for x in minibatch], dtype=torch.float32).to(
            self.device
        )

        # DDQN Logic: Wybierz akcję siecią Policy, Oceń siecią Target
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1).unsqueeze(1)
            next_q_values = (
                self.target_net(next_states).gather(1, next_actions).squeeze(1)
            )
            target_q = rewards + (self.gamma * next_q_values * (1 - dones))

        current_q = self.policy_net(states).gather(1, actions).squeeze(1)

        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping dla stabilności
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

    def save(self, path):
        torch.save(self.policy_net.state_dict(), path)

    def load(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=self.device))
        self.update_target_network()
