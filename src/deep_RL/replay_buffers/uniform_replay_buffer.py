import jax.numpy as jnp
from jax import random

from .base_buffer import BaseReplayBuffer, Experience


class UniformReplayBuffer(BaseReplayBuffer):
    def __init__(self, size: int) -> None:
        super(UniformReplayBuffer, self).__init__(size)

    def sample(self, key: random.PRNGKey) -> Experience:
        """
        Samples a random experience from the replay buffer using
        the uniform distribution.
        """
        key, subkey = random.split(key)
        choices = jnp.array(list(self.buffer.keys()))
        random_idx = random.choice(subkey, choices)

        return self.buffer[random_idx.item()]
