from functools import partial

from jax import jit, random, tree_map, vmap

from .base_buffer import BaseReplayBuffer


class UniformReplayBuffer(BaseReplayBuffer):
    def __init__(
        self,
        buffer_size: int,
        batch_size: int,
    ) -> None:
        super(UniformReplayBuffer, self).__init__(buffer_size, batch_size)

    @partial(jit, static_argnums=(0))
    def sample(
        self,
        key: random.PRNGKey,
        buffer: dict,
        current_buffer_size: int,
    ):
        """
        Samples a random experience from the replay buffer using
        the uniform distribution.

        Args:
            key (random.PRNGKey): the random key used to sample the buffer
            buffer (dict): the buffer to sample experiences from,
                keys: "states", "actions", "rewards", "next_states", "dones"
            current_buffer_size (int): the number of experiences currently stocked in the buffer

        returns:
            dict[str: jnp.ndarray]: A dictionary with keys "states", "actions", "next_states",
            "dones", "rewards"
        """

        @partial(vmap, in_axes=(0, None))
        def sample_batch(indexes, buffer):
            return tree_map(lambda x: x[indexes], buffer)

        key, subkey = random.split(key)
        indexes = random.randint(
            subkey,
            shape=(self.batch_size,),
            minval=0,
            maxval=current_buffer_size,
        )
        experiences = sample_batch(indexes, buffer)

        return experiences, subkey
