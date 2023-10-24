from functools import partial

import jax.numpy as jnp
from jax import jit, lax

from .grid_world import GridWorld


class CliffWalking(GridWorld):
    def __init__(
        self,
        initial_state: jnp.array,
        goal_state: jnp.array,
        grid_size: jnp.array,
        terminal_states: jnp.array,
        stochastic_reset: bool = False,
    ) -> None:
        super().__init__(
            initial_state,
            goal_state,
            grid_size,
            stochastic_reset,
        )

        self.movements = jnp.array([[0, 1], [1, 0], [0, -1], [-1, 0]])

    @partial(jit, static_argnums=0)
    def _get_reward_done(self, new_state):
        def _check_terminal_state_fn():
            is_last_row = new_state[0] == self.grid_size[0] - 1
            is_in_cliff_cols = jnp.isin(
                new_state[1], jnp.arange(0, self.grid_size[1] - 1)
            )

            return is_last_row & is_in_cliff_cols

        terminal_state = _check_terminal_state_fn()
        reward = lax.cond(
            terminal_state,
            lambda _: -100,
            lambda _: -1,
            operand=None,
        )

        done = jnp.all(new_state == self.goal_state)
        reward = jnp.int32(done)

        return reward, done
