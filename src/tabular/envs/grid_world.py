from functools import partial

import jax.numpy as jnp
import jax.random as random
from jax import jit, lax, vmap

from .tabular_base_env import TabularBaseEnv


class GridWorld(TabularBaseEnv):
    def __init__(
        self,
        initial_state: jnp.array,
        goal_state: jnp.array,
        grid_size: jnp.array,
        stochastic_reset: bool = False,
    ) -> None:
        super(GridWorld, self).__init__()

        self.initial_state = initial_state
        self.goal_state = goal_state
        self.grid_size = grid_size
        self.stochastic_reset = stochastic_reset
        self.movements = jnp.array([[0, 1], [1, 0], [0, -1], [-1, 0]])

    def __repr__(self) -> str:
        return str(self.__dict__)

    def _get_obs(self, state):
        """
        The state is fully observable, it doesn't require processing
        """
        return state

    @partial(jit, static_argnums=0)
    def _reset(self, key):
        """
        Resets an agent at the end of an episode
        If stochastic_reset is True, uniformly sample an initial state
        from a set of coordinates
        """
        key, sub_key = random.split(key)

        def _deterministic_reset():
            return self.initial_state, sub_key

        def _stochastic_reset():
            stochastic_state = random.randint(
                key,
                (2,),
                minval=jnp.array([3, 3]),
                maxval=jnp.array([self.grid_size[0], self.grid_size[1]]),
            )

            return stochastic_state, sub_key

        return lax.cond(
            self.stochastic_reset,
            _stochastic_reset,
            _deterministic_reset,
        )

    @partial(jit, static_argnums=0)
    def _reset_if_done(self, env_state, done):
        key = env_state[1]
        return lax.cond(
            done,
            self._reset,
            lambda key: env_state,
            key,
        )

    @partial(jit, static_argnums=0)
    def _get_reward_done(self, new_state):
        done = jnp.all(new_state == self.goal_state)
        reward = jnp.int32(done)

        return reward, done

    @partial(jit, static_argnames=("self"))
    def step(self, env_state, action):
        state, key = env_state
        action = self.movements[action]
        new_state = jnp.clip(
            jnp.add(state, action), jnp.array([0, 0]), self.grid_size - 1
        )
        reward, done = self._get_reward_done(new_state)

        env_state = new_state, key
        env_state = self._reset_if_done(env_state, done)
        new_state = env_state[0]

        return env_state, self._get_obs(new_state), reward, done

    def reset(self, key):
        env_state = self._reset(key)
        new_state = env_state[0]
        return env_state, self._get_obs(new_state)

    @partial(jit, static_argnums=(0))
    def batch_step(self, env_sate, action):
        return vmap(
            GridWorld.step,
            in_axes=(None, (0, 0), 0),  # ((env_state), action)
            out_axes=((0, 0), 0, 0, 0),  # ((env_state), obs, reward, done)
            axis_name="batch_axis",
        )(self, env_sate, action)

    @partial(jit, static_argnums=(0))
    def batch_reset(self, key):
        return vmap(
            GridWorld.reset,
            in_axes=(None, 0),
            out_axes=((0, 0), 0),  # ((env_state), obs)
            axis_name="batch_axis",
        )(self, key)
