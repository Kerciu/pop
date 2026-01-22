import copy

import torch
import torch.nn as nn


class GeneticAgent(nn.Module):
    def __init__(self, state_size, action_size):
        super(GeneticAgent, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, action_size)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)

        x = x.to(self.device)

        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

    def get_action(self, state):
        with torch.no_grad():
            q_values = self.forward(state)
            return torch.argmax(q_values).item()

    def mutate(self, mutation_power=0.02):
        """Tworzy zmutowanego klona tego agenta."""
        child = copy.deepcopy(self)
        for param in child.parameters():
            if len(param.shape) > 1:
                noise = torch.randn_like(param) * mutation_power
                param.data += noise
            else:
                noise = torch.randn_like(param) * mutation_power
                param.data += noise
        return child
