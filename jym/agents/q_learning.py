from functools import partial

import jax.numpy as jnp
from jax import jit, vmap

from .base_agents import BaseTabularAgent


class Q_learning(BaseTabularAgent):
    def __init__(self, discount, learning_rate) -> None:
        super(Q_learning, self).__init__(
            discount,
            learning_rate,
        )

    @partial(jit, static_argnums=(0,))
    def update(self, state, action, reward, done, next_state, q_values):
        target = q_values[tuple(jnp.append(state, action))]
        target += self.learning_rate * (
            reward + self.discount * jnp.max(q_values[tuple(next_state)]) - target
        )
        return q_values.at[tuple(jnp.append(state, action))].set(target)

    @partial(jit, static_argnums=(0,))
    def batch_update(self, state, action, reward, done, next_state, q_values):
        return vmap(
            Q_learning.update,
            #  self, state, action, reward, done, next_state, q_values
            in_axes=(None, 0, 0, 0, 0, 0, -1),
            # return the batch dimension as last dimension of the output
            out_axes=-1,
            axis_name="batch_axis",
        )(self, state, action, reward, done, next_state, q_values)
