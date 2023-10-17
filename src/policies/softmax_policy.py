from functools import partial

import jax.numpy as jnp
from jax import jit, random

from .base_policy import BasePolicy


class Softmax_policy(BasePolicy):
    def __init__(self, temperature=1) -> None:
        self.temperature = temperature

    @partial(jit, static_argnums=(0, 2))
    def prob_distr(self, q_values):
        return jnp.divide(
            jnp.exp(q_values * self.temperature),
            jnp.sum(
                jnp.exp(q_values * self.temperature),
            ),
        )

    @partial(jit, static_argnums=(0, 2))
    def call(self, key, n_actions, state, q_values):
        """
        Returns the argmax w.r.t the softmax distribution over Q-values
        """
        q = q_values.at[tuple(state)].get()
        key, subkey = random.split(key)
        return (
            random.choice(subkey, jnp.arange(n_actions), p=self.prob_distr(q)),
            subkey,
        )
