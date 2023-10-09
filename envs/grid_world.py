from functools import partial

import jax.numpy as jnp
from jax import jit, lax

from .base_env import BaseEnv


class GridWorld(BaseEnv):
    def __init__(self, initial_state, goal_state, grid_size) -> None:
        super(GridWorld, self).__init__()

        self.initial_state = initial_state
        self.goal_state = goal_state
        self.grid_size = grid_size

    def __repr__(self) -> str:
        return str(self.__dict__)

    def _get_obs(self, state):
        return state

    def _reset(self, key):
        return self.initial_state, key

    def _reset_if_done(self, env_state, done):
        key = env_state[1]
        return lax.cond(
            done,
            self._reset,
            lambda key: env_state,
            key,
        )

    def _get_reward_done(self, new_state):
        done = jnp.all(new_state == self.goal_state)
        reward = jnp.int32(done)

        return reward, done

    @partial(jit, static_argnums=(0,))
    def step(self, env_state, action):
        state, key = env_state
        new_state = jnp.clip(state + action, jnp.array([0, 0]), self.grid_size)
        reward, done = self._get_reward_done(new_state)

        env_state = new_state, key
        env_state = self._reset_if_done(env_state, done)
        new_state = env_state[0]

        return env_state, self._get_obs(new_state), reward, done

    @partial(jit, static_argnums=(0,))
    def reset(self, key):
        env_state = self._reset(key)
        new_state = env_state[0]
        return env_state, self._get_obs(new_state)
