from functools import partial

import jax.numpy as jnp
from jax import jit, lax, random

from .base_policy import BasePolicy


class EpsilonGreedy(BasePolicy):
    """
    Epsilon-Greedy policy with random tie-breaks
    """

    def __init__(self, epsilon):
        self.epsilon = epsilon

    @partial(jit, static_argnums=(0, 2))
    def call(self, key, n_actions, state, q_values):
        def _random_action_fn(subkey):
            return random.choice(subkey, jnp.arange(n_actions))

        def _greedy_action_fn(subkey):
            """
            Selects the greedy action with random tie-break
            If multiple Q-values are equal, sample uniformly from their indexes
            """
            q = q_values.at[tuple(state)].get()
            q_max = jnp.max(q, axis=-1)
            q_max_mask = jnp.equal(q, q_max)
            p = jnp.divide(q_max_mask, q_max_mask.sum())
            choice = random.choice(subkey, jnp.arange(n_actions), p=p)
            return jnp.int32(choice)

        explore = random.uniform(key) < self.epsilon
        key, subkey = random.split(key)
        action = lax.cond(
            explore,
            _random_action_fn,
            _greedy_action_fn,
            operand=subkey,
        )

        return action, subkey
