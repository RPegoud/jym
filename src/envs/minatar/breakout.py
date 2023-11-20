from typing import Tuple

import jax.numpy as jnp
from chex import dataclass
from jax import lax, random

from ..base_envs import BaseEnv


@dataclass
class EnvState:
    ball_x: int
    last_x: int
    ball_y: int
    last_y: int
    ball_dir: int
    pos: int
    brick_map: jnp.ndarray
    strike: bool
    time: int
    done: bool


class Breakout(BaseEnv):
    """
    Atari Breakout environment.
    Loosely based on the Gymnax version:
        https://github.com/RobertTLange/gymnax/blob/main/gymnax/environments/minatar/breakout.py

    EnvState attributes:
    * ball_x (int): x-axis position of the ball, range: 0-9
    * ball_x (int): x-axis position of the ball, range: 0-9
    * last_y (int): y-axis position of the ball at the last timestep, range: 0-9
    * last_y (int): y-axis position of the ball at the last timestep, range: 0-9
    * ball_dir (int): direction of the ball
            * 0: Up and to the left
            * 1: Up and to the right
            * 2: Down and to the right
            * 3: Down and to the left
    * pos (int): x-axis position of the player's paddle, range: 0-9
    * brick_map (jnp.ndarray): array representation of the brick layers,
    three layers are initially present, from rows 2 to 4
    * strike (bool): whether the ball has hit a brick at the current timestep
    * time (int): current timestep
    * done (bool): done flag
    """

    def __init__(self) -> None:
        self.channels = {"paddle": 0, "ball": 1, "trail": 2, "brick": 3}
        self.obs_shape = (10, 10, 4)  # x, y, channels [paddle, ball, trail, brick]
        self.actions = jnp.arange(3)  # no action, left, right

    def __repr__(self) -> str:
        return str(self.__dict__)

    def _get_obs(self, state: EnvState) -> jnp.array:
        """
        Converts EnvStates to observations.
        """
        obs = jnp.zeros(self.obs_shape, dtype=jnp.bool_)
        obs = obs.at[9, state.pos, self.channels["paddle"]].set(1)
        obs = obs.at[state.ball_y, state.ball_x, self.channels["ball"]].set(1)
        obs = obs.at[state.last_y, state.last_x, self.channels["trail"]].set(1)
        obs = obs.at[:, :, self.channels["brick"]].set(state.brick_map)
        return obs.astype(jnp.float32)

    def _reset(self, key: random.PRNGKey) -> Tuple[jnp.array, EnvState]:
        ball_init = random.uniform(key, ()) > 0.5
        ball_x, ball_dir = lax.select(
            ball_init,
            jnp.array([0, 2]),  # ball in the leftmost column, going down/right
            jnp.array([9, 3]),  # ball in the rightmost column, going down/left
        )
        env_state = EnvState(
            ball_x=ball_x,
            last_x=ball_x,
            ball_y=3,
            last_y=3,
            ball_dir=ball_dir,
            pos=4,
            brick_map=jnp.zeros((10, 10)).at[1:4, :].set(1),
            strike=False,
            time=0,
            done=False,
        )
        return self._get_obs(env_state), env_state

    def _reset_if_done(
        self, state: EnvState, key: random.PRNGKey
    ) -> Tuple[jnp.array, EnvState]:
        def _reset_fn(key):
            return self._reset(key)

        def _no_reset_fn(key):
            return self._get_obs(state), state

        return lax.cond(state.done, _reset_fn, _no_reset_fn, operand=key)

    def step(self, state: EnvState, action: int):
        raise NotImplementedError

    def reset(self, key: random.PRNGKey) -> Tuple[jnp.array, EnvState]:
        return self._reset(key)


def agent_step(state: EnvState, action: int) -> Tuple[EnvState, int, int]:
    """
    Handles agent movement and boundary checks.
    """

    def _dont_move(pos):
        return pos

    def _move_left(pos):
        return jnp.maximum(0, pos - 1)

    def _move_right(pos):
        return jnp.maximum(9, pos + 1)

    # update agent position
    pos = lax.switch(
        action,
        [_dont_move, _move_left, _move_right],
        operand=state.pos,
    )
    last_x = state.ball_x
    last_y = state.ball_y

    # update ball position
    new_x = lax.cond(
        jnp.isin(action, jnp.array([1, 2])),
        lambda x: x + 1,
        lambda x: x - 1,
        operand=state.ball_x,
    )
    new_y = lax.cond(
        jnp.isin(action, jnp.array([2, 3])),
        lambda y: y + 1,
        lambda y: y - 1,
        operand=state.ball_y,
    )

    border_cond_x = jnp.logical_or(new_x < 0, new_x > 9)
    # ensure the ball doesn't leave the grid
    new_x = jnp.clip(new_x, 0, 9)

    ball_dir = lax.select(
        border_cond_x,
        jnp.array([1, 0, 3, 2]),
        state.ball_dir,
    )

    return (
        state.replace(
            pos=pos,
            last_x=last_x,
            last_y=last_y,
            ball_dir=ball_dir,
        ),
        new_x,
        new_y,
    )
