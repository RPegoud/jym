from functools import partial

import jax.numpy as jnp
from jax import jit, vmap

from .base_agent import BaseAgent


class Expected_Sarsa(BaseAgent):
    def __init__(self, discount, learning_rate) -> None:
        super(Expected_Sarsa, self).__init__(
            discount,
            learning_rate,
        )

    @partial(jit, static_argnums=(0,))
    def softmax_prob_distr(
        self,
        q_values,
    ):
        return jnp.divide(
            jnp.exp(q_values),
            jnp.sum(
                jnp.exp(q_values),
            ),
        )

    @partial(jit, static_argnums=(0))
    def update(self, state, action, reward, done, next_state, q_values):
        next_q_values = q_values[tuple(next_state)]
        target = q_values[tuple(jnp.append(state, action))]
        target += self.learning_rate * (
            reward
            + self.discount
            * jnp.sum(next_q_values * self.softmax_prob_distr(next_q_values))
            - target
        )
        return q_values.at[tuple(jnp.append(state, action))].set(target)

    @partial(jit, static_argnums=(0))
    def batch_update(self, state, action, reward, done, next_state, q_values):
        return vmap(
            Expected_Sarsa.update,
            #  self, state, action, reward, done, next_state, q_values
            in_axes=(None, 0, 0, 0, 0, 0, -1),
            # return the batch dimension as last dimension of the output
            out_axes=-1,
            axis_name="batch_axis",
        )(self, state, action, reward, done, next_state, q_values)
