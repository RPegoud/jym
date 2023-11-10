from functools import partial

import jax.numpy as jnp
from jax import random, tree_map, vmap

from .base_buffer import BaseReplayBuffer


class UniformReplayBuffer(BaseReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        batch_size: int,
    ) -> None:
        super(UniformReplayBuffer, self).__init__(buffer_size, batch_size)

    # @partial(jit, static_argnums=(0))
    def sample(
        self,
        key: random.PRNGKey,
        buffer: dict,
        n_experiences: int,
    ):
        """
        Samples a random experience from the replay buffer using
        the uniform distribution.

        Args:
            key (random.PRNGKey): the random key used to sample the buffer
            buffer (dict): the buffer to sample experiences from,
                keys: "states", "actions", "rewards", "next_states", "dones"
            n_experiences (int): the number of experiences currently stocked in the buffer
        """

        # used to avoid sampling from empty experiences, i.e. zeros
        choices = jnp.arange(
            jnp.min(
                jnp.array([n_experiences, self.buffer_size]),
            ),
        )

        @partial(vmap, in_axes=(0, None))
        def sample_batch(indexes, buffer):
            return tree_map(lambda x: x[indexes], buffer)

        key, subkey = random.split(key)
        indexes = random.choice(
            subkey,
            choices,
            shape=(self.batch_size,),
        )
        samples = sample_batch(indexes, buffer)
        return [exp for exp in zip(*samples.values())]
