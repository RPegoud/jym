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
    def call(self, key, n_actions, q_values):
        def _random_action_fn(subkey):
            return random.choice(subkey, jnp.arange(n_actions))

        def _greedy_action_fn(subkey):
            """
            Selects the greedy action with random tie-break
            """
            q_max = jnp.max(q_values)
            q_max_mask = jnp.equal(q_values, q_max)
            p = q_max_mask / q_max_mask.sum()
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
