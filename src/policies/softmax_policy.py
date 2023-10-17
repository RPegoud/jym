from functools import partial

import jax.numpy as jnp
from jax import jit

from .base_policy import BasePolicy


class Softmax_policy(BasePolicy):
    def __init__(self, temperature) -> None:
        self.temperature = temperature

    @partial(jit, static_argnums=(0))
    def call(self, state, q_values):
        def _softmax_fn(q_values):
            return jnp.divide(
                jnp.exp(q_values * self.temperature),
                jnp.sum(
                    jnp.exp(q_values * self.temperature),
                ),
            )

        q = q_values.at[tuple(state)].get()
        return _softmax_fn(q)
