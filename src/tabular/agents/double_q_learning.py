from functools import partial

import jax.numpy as jnp
from jax import jit, lax

from .base_agent import BaseAgent


class Double_Q_learning(BaseAgent):
    def __init__(self, discount, learning_rate) -> None:
        super(Double_Q_learning, self).__init__(
            discount,
            learning_rate,
        )

    @partial(jit, static_argnums=(0))
    def update(self, state, action, reward, done, next_state, q1, q2, to_update: bool):
        @partial(jit, static_argnums=(0,))
        def update_q1():
            best_action = jnp.argmax(q1[tuple(next_state)])
            target = q1[tuple(jnp.append(state, action))]
            target += self.learning_rate * (
                reward
                + self.discount * q2[tuple(jnp.append(next_state, best_action))]
                - target
            )
            return q1.at[tuple(jnp.append(state, action))].set(target), q2

        def update_q2():
            best_action = jnp.argmax(q2[tuple(next_state)])
            target = q2[tuple(jnp.append(state, action))]
            target += self.learning_rate * (
                reward
                + self.discount * q1[tuple(jnp.append(next_state, best_action))]
                - target
            )
            return q1, q2.at[tuple(jnp.append(state, action))].set(target)

        return lax.cond(
            to_update,
            update_q1,
            update_q2,
        )
