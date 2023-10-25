from functools import partial

import jax.numpy as jnp
from jax import jit

from .grid_world import GridWorld


class CliffWalking(GridWorld):
    def __init__(
        self,
        initial_state: jnp.array,
        goal_state: jnp.array,
        grid_size: jnp.array,
        stochastic_reset: bool = False,
    ) -> None:
        super().__init__(
            initial_state,
            goal_state,
            grid_size,
            stochastic_reset,
        )

        self.movements = jnp.array([[0, 1], [1, 0], [0, -1], [-1, 0]])
        # The cliff walking environment has rewards of -1 per step and
        # -100 for falling off the cliff (cells in the middle of the bottom row)
        self.reward_map = (
            (jnp.ones(self.grid_size) * -1)
            .at[-1, 1:-1]
            .set(-100)
            .at[tuple(self.goal_state)]
            .set(0)
        )

    @partial(jit, static_argnums=0)
    def _get_reward_done(self, new_state):
        reward = self.reward_map[tuple(new_state)]
        # done = jnp.equal(reward, -100) & jnp.all(new_state == self.goal_state)
        done = jnp.bool_(new_state[0] == 3 & new_state[1] >= 1)

        return reward, done
