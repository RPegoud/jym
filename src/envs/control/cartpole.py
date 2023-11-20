import math
from functools import partial

import jax.numpy as jnp
from jax import jit, lax, random

from ..base_envs import BaseEnv


class CartPole(BaseEnv):
    """
    Copied from https://github.com/Farama-Foundation/Gymnasium/blob/main/gymnasium/envs/classic_control/cartpole.py
    ## Description

    This environment corresponds to the version of the cart-pole problem described by Barto, Sutton, and Anderson in
    ["Neuronlike Adaptive Elements That Can Solve Difficult Learning Control Problem"](https://ieeexplore.ieee.org/document/6313077).
    A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track.
    The pendulum is placed upright on the cart and the goal is to balance the pole by applying forces
     in the left and right direction on the cart.

    ## Action Space

    The action is a `ndarray` with shape `(1,)` which can take values `{0, 1}` indicating the direction
     of the fixed force the cart is pushed with.

    - 0: Push cart to the left
    - 1: Push cart to the right

    **Note**: The velocity that is reduced or increased by the applied force is not fixed and it depends on the angle
     the pole is pointing. The center of gravity of the pole varies the amount of energy needed to move the cart underneath it

    ## Observation Space

    The observation is a `ndarray` with shape `(4,)` with the values corresponding to the following positions and velocities:

    | Num | Observation           | Min                 | Max               |
    |-----|-----------------------|---------------------|-------------------|
    | 0   | Cart Position         | -4.8                | 4.8               |
    | 1   | Cart Velocity         | -Inf                | Inf               |
    | 2   | Pole Angle            | ~ -0.418 rad (-24°) | ~ 0.418 rad (24°) |
    | 3   | Pole Angular Velocity | -Inf                | Inf               |

    **Note:** While the ranges above denote the possible values for observation space of each element,
        it is not reflective of the allowed values of the state space in an unterminated episode. Particularly:
    -  The cart x-position (index 0) can take values between `(-4.8, 4.8)`, but the episode terminates
       if the cart leaves the `(-2.4, 2.4)` range.
    -  The pole angle can be observed between  `(-.418, .418)` radians (or **±24°**), but the episode terminates
       if the pole angle is not in the range `(-.2095, .2095)` (or **±12°**)

    ## Rewards

    Since the goal is to keep the pole upright for as long as possible, a reward of `+1` for every step taken,
    including the termination step, is allotted. The threshold for rewards is 500 for v1 and 200 for v0.

    ## Starting State

    All observations are assigned a uniformly random value in `(-0.05, 0.05)`

    ## Episode End

    The episode ends if any one of the following occurs:

    1. Termination: Pole Angle is greater than ±12°
    2. Termination: Cart Position is greater than ±2.4 (center of the cart reaches the edge of the display)
    3. Truncation: Episode length is greater than 500 (200 for v0)
    """

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

        def _reset_fn(key):
            return self._reset(key)

        def _no_reset_fn(key):
            return env_state

        return lax.cond(
            done,
            _reset_fn,
            _no_reset_fn,
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
        theta += self.tau * theta_dot
        theta_dot += self.tau * theta_accel

        new_state = jnp.array([x, x_dot, theta, theta_dot])

        done = (
            (x < -self.x_limit)
            | (x > self.x_limit)
            | (theta > self.theta_limit_rads)
            | (theta < -self.theta_limit_rads)
        )
        reward = jnp.float32(jnp.invert(done))

        env_state = new_state, key
        env_state = self._reset_if_done(env_state, done)
        new_state = env_state[0]

        return env_state, self._get_obs(new_state), reward, done

    def reset(self, key):
        env_state = self._reset(key)
        new_state = env_state[0]
        return env_state, self._get_obs(new_state)
