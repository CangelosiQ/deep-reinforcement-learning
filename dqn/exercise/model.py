import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, hidden_layers_sizes:list = None, seed=42):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        if hidden_layers_sizes is not None:
            self.first_layer = nn.Linear(state_size, hidden_layers_sizes[0])
            n_hidden_layers = len(hidden_layers_sizes)
            if n_hidden_layers > 1:
                self.hidden_layers = [nn.Linear(hidden_layers_sizes[i], hidden_layers_sizes[i+1]) for i in range(n_hidden_layers-1)]
            print(self.hidden_layers)
            self.last_layer = nn.Linear(hidden_layers_sizes[-1], action_size)
        else:
            self.first_layer = nn.Linear(state_size, action_size)
            self.hidden_layers = []
            self.last_layer = None

    def forward(self, state):
        """Build a network that maps state -> action values."""
        x = F.relu(self.first_layer(state))
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        if self.last_layer:
            x = self.last_layer(x)
        return x


# class QNetworkSequential(nn.Module):
#     """Actor (Policy) Model."""
#
#     def __init__(self, state_size, action_size, hidden_sizes:list = None, seed=42):
#         """Initialize parameters and build model.
#         Params
#         ======
#             state_size (int): Dimension of each state
#             action_size (int): Dimension of each action
#             seed (int): Random seed
#         """
#         super(QNetworkSequential, self).__init__()
#         self.seed = torch.manual_seed(seed)
#         from collections import OrderedDict
#         self.model = nn.Sequential(OrderedDict([
#             ('fc1', nn.Linear(state_size, hidden_sizes[0])),
#             ('relu1', nn.ReLU()),] + [
#             ('fc2', nn.Linear(hidden_sizes[0], hidden_sizes[1])),
#             ('relu2', nn.ReLU()),
#             ('output', nn.Linear(hidden_sizes[1], action_size)),
#             ('softmax', nn.Softmax(dim=1))]))
#
#
#
#     def forward(self, state):
#         """Build a network that maps state -> action values."""
#         x = F.relu(self.first_layer(state))
#         for layer in self.hidden_layers:
#             x = F.relu(layer(x))
#         if self.last_layer:
#             x = self.last_layer(x)
#         return x