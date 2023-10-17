from functools import partial

import jax.numpy as jnp
from jax import jit

from .base_agent import BaseAgent


class Q_learning(BaseAgent):
    def __init__(self, key, n_states, n_actions, discount, learning_rate) -> None:
        super(Q_learning, self).__init__(
            key,
            n_states,
            n_actions,
            discount,
        )

        self.learning_rate = learning_rate

    @partial(jit, static_argnums=(0,))
    def update(self, state, action, reward, done, next_state, q_values):
        update = q_values[state[0], state[1], action]
        update += self.learning_rate * (
            reward + self.discount * jnp.max(q_values[tuple(next_state)]) - update
        )
        return q_values.at[state[0], state[1], action].set(update)

    def act(self):
        pass
