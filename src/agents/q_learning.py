from functools import partial

import jax.numpy as jnp
from jax import jit

from .base_agent import BaseAgent


class Q_learning(BaseAgent):
    def __init__(
        self, key, n_states, n_actions, discount, learning_rate, policy
    ) -> None:
        super(Q_learning, self).__init__(
            key,
            n_states,
            n_actions,
            discount,
        )

        self.learning_rate = learning_rate
        self.policy = policy

    # @partial(jit, static_argnums=(0,))
    # def update(self, state, action, reward, done, next_state, q_values):
    #     def update_fn():
    #         update = q_values[state[0], state[1], action]
    #         update += self.learning_rate * (
    #             reward + self.discount * jnp.max(q_values[tuple(next_state)]) - update
    #         )
    #         return q_values.at[state[0], state[1], action].set(update)

    #     def terminal_update_fn():
    #         return q_values.at[tuple(next_state)].set(0)

    #     return lax.cond(
    #         done,
    #         terminal_update_fn,
    #         update_fn,
    #     )

    # TODO: add terminal state update
    @partial(jit, static_argnums=(0,))
    def update(self, state, action, reward, done, next_state, q_values):
        update = q_values[state[0], state[1], action]
        update += self.learning_rate * (
            reward + self.discount * jnp.max(q_values[tuple(next_state)]) - update
        )
        return q_values.at[state[0], state[1], action].set(update)
