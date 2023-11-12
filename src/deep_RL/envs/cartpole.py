import math
from functools import partial

import jax.numpy as jnp
from jax import jit, lax, random

from .base_env import BaseControlEnv


class CartPole(BaseControlEnv):
    def __init__(self) -> None:
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5  # half the pole's length
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02  # seconds between state updates
        self.reset_bounds = 0.05

        # Limits defining episode termination
        self.x_limit = 2.4
        self.theta_limit_rads = 12 * 2 * math.pi / 360

    def __repr__(self) -> str:
        return str(self.__dict__)

    def _get_obs(self, state):
        # the state is fully observable
        return state

    @partial(jit, static_argnums=0)
    def _reset(self, key):
        new_state = random.uniform(
            key,
            shape=(4,),
            minval=-self.reset_bounds,
            maxval=self.reset_bounds,
        )
        key, sub_key = random.split(key)

        return new_state, sub_key

    @partial(jit, static_argnums=0)
    def _reset_if_done(self, env_state, done):
        key = env_state[1]

        def reset_fn(key):
            return self._reset(key)

        def no_reset_fn(key):
            return env_state

        return lax.cond(
            done,
            reset_fn,
            no_reset_fn,
            operand=key,
        )

    @partial(jit, static_argnums=(0))
    def step(self, env_state, action):
        state, key = env_state
        x, x_dot, theta, theta_dot = state

        force = lax.cond(
            jnp.all(action) == 1,
            lambda _: self.force_mag,
            lambda _: -self.force_mag,
            operand=None,
        )
        cos_theta, sin_theta = jnp.cos(theta), jnp.sin(theta)

        temp = (
            force + self.polemass_length * jnp.square(theta_dot) * sin_theta
        ) / self.total_mass
        theta_accel = (self.gravity * sin_theta - cos_theta * temp) / (
            self.length
            * (4.0 / 3.0 - self.masspole * jnp.square(cos_theta) / self.total_mass)
        )
        x_accel = (
            temp - self.polemass_length * theta_accel * cos_theta / self.total_mass
        )

        # euler
        x += self.tau * x_dot
        x_dot += self.tau * x_accel
        theta += theta + self.tau * theta_dot
        theta_dot += self.tau * theta_accel

        new_state = jnp.array([x, x_dot, theta, theta_dot])

        done = (
            (x < -self.x_limit)
            | (x > self.x_limit)
            | (theta > self.theta_limit_rads)
            | (theta < -self.theta_limit_rads)
        )
        reward = jnp.int32(jnp.invert(done))

        env_state = new_state, key
        env_state = self._reset_if_done(env_state, done)
        new_state = env_state[0]

        return env_state, self._get_obs(new_state), reward, done

    def reset(self, key):
        env_state = self._reset(key)
        new_state = env_state[0]
        return env_state, self._get_obs(new_state)
