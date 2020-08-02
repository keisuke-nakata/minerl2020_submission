import torch
import torch.nn as nn
import torch.nn.functional as F

from pfrl import action_value
from pfrl.q_function import StateQFunction
from pfrl.q_functions.dueling_dqn import constant_bias_initializer
from pfrl.initializers import init_chainer_default


def parse_arch(arch, n_actions, n_input_channels):
    if arch == 'dueling':
        # Conv2Ds of (channel, kernel, stride): [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
        # return DuelingDQN(n_actions, n_input_channels=n_input_channels, hiddens=[256])
        raise NotImplementedError('dueling')
    elif arch == 'distributed_dueling':
        n_atoms = 51
        v_min = -10
        v_max = 10
        return DistributionalDuelingDQN(n_actions, n_atoms, v_min, v_max, n_input_channels=n_input_channels)
    else:
        raise RuntimeError('Unsupported architecture name: {}'.format(arch))


class DistributionalDuelingDQN(nn.Module, StateQFunction):
    """Distributional dueling fully-connected Q-function with discrete actions."""

    def __init__(
        self,
        n_actions,
        n_atoms,
        v_min,
        v_max,
        n_input_channels=4,
        activation=torch.relu,
        bias=0.1,
    ):
        assert n_atoms >= 2
        assert v_min < v_max

        self.n_actions = n_actions
        self.n_input_channels = n_input_channels
        self.activation = activation
        self.n_atoms = n_atoms

        super().__init__()
        self.z_values = torch.linspace(v_min, v_max, n_atoms, dtype=torch.float32)

        self.conv_layers = nn.ModuleList(
            [
                nn.Conv2d(n_input_channels, 32, 8, stride=4),
                nn.Conv2d(32, 64, 4, stride=2),
                nn.Conv2d(64, 64, 3, stride=1),
            ]
        )

        # ここだけ変える必要があった
        # self.main_stream = nn.Linear(3136, 1024)
        self.main_stream = nn.Linear(1024, 1024)
        self.a_stream = nn.Linear(512, n_actions * n_atoms)
        self.v_stream = nn.Linear(512, n_atoms)

        self.apply(init_chainer_default)
        self.conv_layers.apply(constant_bias_initializer(bias=bias))

    def forward(self, x):
        h = x
        for l in self.conv_layers:
            h = self.activation(l(h))

        # Advantage
        batch_size = x.shape[0]

        h = self.activation(self.main_stream(h.view(batch_size, -1)))
        h_a, h_v = torch.chunk(h, 2, dim=1)
        ya = self.a_stream(h_a).reshape((batch_size, self.n_actions, self.n_atoms))

        mean = ya.sum(dim=1, keepdim=True) / self.n_actions

        ya, mean = torch.broadcast_tensors(ya, mean)
        ya -= mean

        # State value
        ys = self.v_stream(h_v).reshape((batch_size, 1, self.n_atoms))
        ya, ys = torch.broadcast_tensors(ya, ys)
        q = F.softmax(ya + ys, dim=2)

        self.z_values = self.z_values.to(x.device)
        return action_value.DistributionalDiscreteActionValue(q, self.z_values)
