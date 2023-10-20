from functools import partial

import jax.numpy as jnp
from jax import jit, lax, random, vmap

from .base_policy import BasePolicy


class BanditEpsilonGreedy(BasePolicy):
    """
    Epsilon-Greedy policy with random tie-breaks
    """

    @staticmethod
    @partial(jit, static_argnums=(1, 3))
    def call(key, n_actions, q_values, epsilon):
        def _random_action_fn(subkey):
            return random.choice(subkey, jnp.arange(n_actions))

        def _greedy_action_fn(subkey):
            """
            Selects the greedy action with random tie-break
            If multiple Q-values are equal, sample uniformly from their indexes
            """
            q_max = jnp.max(q_values, axis=-1)
            q_max_mask = jnp.equal(q_values, q_max)
            p = jnp.divide(q_max_mask, q_max_mask.sum())
            choice = random.choice(subkey, jnp.arange(n_actions), p=p)
            return jnp.int32(choice)

        explore = random.uniform(key) < epsilon
        key, subkey = random.split(key)
        action = lax.cond(
            explore,
            _random_action_fn,
            _greedy_action_fn,
            operand=subkey,
        )

        return action, subkey

    @staticmethod
    @partial(jit, static_argnums=(1))
    def batched_call(key, n_actions, q_values, epsilon):
        return vmap(
            BanditEpsilonGreedy.call,
            in_axes=(0, None, -1, 0),
        )(key, n_actions, q_values, epsilon)

    @staticmethod
    @partial(jit, static_argnums=(1))
    def multi_run_batched_call(keys, n_actions, q_values, epsilons):
        # Vmap over both the number of runs and the batch dimension
        return vmap(
            BanditEpsilonGreedy.batched_call,
            in_axes=(1, None, -1, None),
            out_axes=(-1, 1),
        )(keys, n_actions, q_values, epsilons)
