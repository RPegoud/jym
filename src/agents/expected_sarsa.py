from functools import partial

import jax.numpy as jnp
from jax import jit

from .base_agent import BaseAgent


class Expected_Sarsa(BaseAgent):
    def __init__(self, key, n_states, n_actions, discount, learning_rate) -> None:
        super(Expected_Sarsa, self).__init__(
            key,
            n_states,
            n_actions,
            discount,
        )

        self.learning_rate = learning_rate
        # self.policy = Softmax_policy()

    @partial(jit, static_argnums=(0,))
    def softmax_prob_distr(self, q_values, temperature):
        return jnp.divide(
            jnp.exp(q_values * temperature),
            jnp.sum(
                jnp.exp(q_values * temperature),
            ),
        )

    @partial(jit, static_argnums=(0,))
    def update(self, state, action, reward, done, next_state, q_values, temperature=1):
        target = q_values[tuple(jnp.append(state, action))]
        target += self.learning_rate * (
            reward
            + self.discount
            * jnp.mean(
                q_values[tuple(next_state)]
                * self.softmax_prob_distr(
                    q_values[tuple(next_state)],
                    temperature,
                ),
            )
            - target
        )
        return q_values.at[tuple(jnp.append(state, action))].set(target)

    def act(self):
        pass
