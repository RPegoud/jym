from typing import Sequence

from flax import linen as nn


class MLP(nn.Module):
    neurons_per_layer: Sequence[int]

    @nn.compact
    def __call__(self, x):
        activation = x
        for i, num_nurons in enumerate(self.neurons_per_layer):
            activation = nn.Dense(num_nurons)(activation)
            if i != len(self.neurons_per_layer) - 1:
                activation = nn.relu(activation)
        return activation
