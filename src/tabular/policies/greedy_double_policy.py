from functools import partial

import jax.numpy as jnp
from jax import jit, lax, random

from .base_policy import BasePolicy


class DoubleEpsilonGreedy(BasePolicy):
    """
    Epsilon-Greedy policy with random tie-breaks
    Combines the Q-values of double models, either by adding or averaging
    """

    def __init__(self, epsilon, sum_qs=1):
        self.epsilon = epsilon
        self.sum_qs = sum_qs

    @partial(jit, static_argnums=(0, 2))
    def call(self, key, n_actions, state, q_values_1, q_values_2):
        def _random_action_fn(subkey):
            return random.choice(subkey, jnp.arange(n_actions))

        def _greedy_action_fn(subkey):
            """
            Selects the greedy action with random tie-break
            If multiple Q-values are equal, sample uniformly from their indexes
            """

            def add_qs():
                return jnp.add(q1, q2)

            def avg_qs():
                return jnp.mean(jnp.array([q1, q2]), axis=0)

            q1 = q_values_1.at[tuple(state)].get()
            q2 = q_values_2.at[tuple(state)].get()

            # if sum_qs is true, the action are selected using Q1 + Q2
            # otherwise mean(Q1, Q2)
            q_merged = lax.cond(
                self.sum_qs,
                add_qs,
                avg_qs,
            )

            q_max = jnp.max(q_merged, axis=-1)
            q_max_mask = jnp.equal(q_merged, q_max)
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
