from functools import partial
from typing import Tuple

import jax.numpy as jnp
from chex import dataclass
from jax import jit, lax, random

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
    Original MinAtar env:
        https://github.com/kenjyoung/MinAtar/blob/master/minatar/environments/breakout.py

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

    def __init__(self, max_steps_in_episode: int = 1000) -> None:
        self.channels = {"paddle": 0, "ball": 1, "trail": 2, "brick": 3}
        self.obs_shape = (10, 10, 4)  # x, y, channels ([paddle, ball, trail, brick])
        self.actions = jnp.arange(3)  # no action, left, right
        self.max_steps_in_episode = max_steps_in_episode

    def __repr__(self) -> str:
        return str(self.__dict__)

    @property
    def n_actions(self) -> int:
        return len(self.actions)

    def _get_obs(self, state: EnvState) -> jnp.ndarray:
        """
        Converts EnvStates to observations (jnp.ndarray).
        """
        obs = jnp.zeros(self.obs_shape, dtype=jnp.bool_)
        obs = obs.at[9, state.pos, self.channels["paddle"]].set(1)
        obs = obs.at[state.ball_y, state.ball_x, self.channels["ball"]].set(1)
        obs = obs.at[state.last_y, state.last_x, self.channels["trail"]].set(1)
        obs = obs.at[:, :, self.channels["brick"]].set(state.brick_map)
        return obs.astype(jnp.float32)

    def _reset(self, key: random.PRNGKey) -> Tuple[jnp.array, EnvState]:
        key, subkey = random.split(key)

        ball_init = random.uniform(subkey, ()) > 0.5
        ball_x, ball_dir = lax.select(
            ball_init,
            jnp.array([0, 2]),  # ball in the leftmost column, going down/right
            jnp.array([9, 3]),  # ball in the rightmost column, going down/left
        )
        state = EnvState(
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

        return state, self._get_obs(state), subkey

    def _reset_if_done(
        self, state: EnvState, env_key: random.PRNGKey
    ) -> Tuple[EnvState, jnp.ndarray, random.PRNGKey]:
        def _reset_fn(env_key):
            return self._reset(env_key)

        def _no_reset_fn(env_key):
            return state, self._get_obs(state), env_key

        return lax.cond(state.done, _reset_fn, _no_reset_fn, operand=env_key)

    @partial(jit, static_argnums=(0))
    def step(self, state: EnvState, env_key: random.PRNGKey, action: int):
        state, new_x, new_y = agent_step(state, action)
        state, reward = step_ball_brick(state, new_x, new_y)

        state = state.replace(time=state.time + 1)
        done = jnp.logical_or(state.done, state.time >= self.max_steps_in_episode)
        state = state.replace(done=done)

        state, obs, env_key = self._reset_if_done(state, env_key)

        return state, obs, reward, done, env_key

    def reset(self, key: random.PRNGKey) -> Tuple[jnp.array, EnvState]:
        return self._reset(key)


@jit
def agent_step(state: EnvState, action: int) -> Tuple[EnvState, int, int]:
    """
    Handles agent movement and boundary checks.

    The movement functions (_dont_move, _move_left, _move_right) are used to update the paddle position
    based on the action. The paddle moves within the grid boundaries.
    Ball's position is updated considering its direction and grid boundaries.
    """

    def _dont_move(pos):
        return pos

    def _move_left(pos):
        return jnp.maximum(0, pos - 1)

    def _move_right(pos):
        return jnp.maximum(pos + 1, 9)

    # Update agent's position based on the action
    pos = lax.switch(
        action,
        [_dont_move, _move_left, _move_right],
        operand=state.pos,
    )
    last_x = state.ball_x
    last_y = state.ball_y

    # Update ball position based on its direction
    new_x = lax.cond(
        # if the ball is moving right
        jnp.isin(state.ball_dir, jnp.array([1, 2])),
        lambda x: x + 1,
        lambda x: x - 1,
        operand=state.ball_x,
    )
    new_y = lax.cond(
        # if the ball is moving up
        jnp.isin(state.ball_dir, jnp.array([2, 3])),
        lambda y: y + 1,
        lambda y: y - 1,
        operand=state.ball_y,
    )

    # Check if the ball's new position is within the grid boundaries
    border_cond_x = jnp.logical_or(new_x < 0, new_x > 9)
    new_x = jnp.clip(new_x, 0, 9)  # Ensure ball stays within the grid

    # Update ball's direction if it hits horizontal boundaries
    ball_dir = lax.select(
        border_cond_x,
        jnp.array([1, 0, 3, 2])[state.ball_dir],
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


@jit
def step_ball_brick(state: EnvState, new_x: int, new_y: int) -> Tuple[EnvState, float]:
    """
    Handles the ball's interaction with bricks and the paddle, and determines the game state.

    - Updates the game state based on the ball's position and actions taken.
    - Manages collisions with bricks, the paddle, and the grid boundaries.
    - Computes the reward based on brick collisions and determines if the game has reached a terminal state.

    Variables:
        * `border_cond_y`: whether the ball has hit the top of the grid and needs to be redirected.
        * `strike_toggle`: whether the ball has hit a brick (True when border_cond_y is False
        and the ball collides with a brick).
        * `strike_bool`: whether the ball has struck a brick and `strike` wasn't already set.
        If True, update the `strike` status and increase the reward.
        * `brick_cond`: whether the ball is at the bottom of the grid but hasn't hit a brick.
        * `spawn_bricks`: whether to reset the brick wall. Set to true when `brick_cond` is
        True and `brick_map` is empty.
        * `redirect_ball_old`: whether the ball has collided with the paddle at
        the paddle's old position (state.ball_x == state.pos).
        * `not_redirected`: preliminary check for a collision with the new paddle
        position, ensuring the ball didn't already hit the paddle at its old position.
        * `redirect_ball_new`: whether the ball has collided with the paddle at
        the paddle's new position (new_x == state.pos).
        * `not_redirected`: True when the ball has not collided with the paddle
        at either its old or new position.
        * `terminal`: whether the game has reached a terminal state, i.e.,
        when the ball misses the paddle.
    """

    reward = 0

    # Reflect the ball's direction if it hits the top border
    border_cond_y = new_y < 0
    new_y = jnp.clip(new_y, 0, 9)  # Ensure new_y remains within the grid

    # Check for collision with a brick
    strike_toggle = jnp.logical_and(
        jnp.invert(border_cond_y), state.brick_map[new_y, new_x] == 1
    )
    strike_bool = jnp.logical_and(jnp.invert(state.strike), strike_toggle)
    reward += jnp.float32(strike_bool)  # Increment reward if a brick is struck

    # Remove the brick on collision
    brick_map = lax.select(
        strike_bool, state.brick_map.at[new_y, new_x].set(0), state.brick_map
    )

    # Update ball position and direction post-collision with brick
    new_y = lax.select(strike_bool, state.last_y, new_y)
    ball_dir = lax.select(
        strike_bool, jnp.array([3, 2, 1, 0])[state.ball_dir], state.ball_dir
    )

    # Check for ball at the bottom row but not colliding with a brick
    brick_cond = jnp.logical_and(jnp.invert(strike_toggle), new_y == 9)

    # Spawn new bricks if all are cleared
    spawn_bricks = jnp.logical_and(brick_cond, jnp.count_nonzero(brick_map) == 0)
    brick_map = lax.select(spawn_bricks, brick_map.at[1:4, :].set(1), brick_map)

    # Handle ball collision with paddle's old position
    redirect_ball_old = jnp.logical_and(brick_cond, state.ball_x == state.pos)
    ball_dir = lax.select(
        redirect_ball_old, jnp.array([3, 2, 1, 0])[ball_dir], ball_dir
    )
    new_y = lax.select(redirect_ball_old, state.last_y, new_y)

    # Handle ball collision with paddle's new position
    redirect_ball_new = jnp.logical_and(brick_cond, jnp.invert(redirect_ball_old))
    collision_new_pos = jnp.logical_and(redirect_ball_new, new_x == state.pos)
    ball_dir = lax.select(
        collision_new_pos, jnp.array([2, 3, 0, 1])[ball_dir], ball_dir
    )
    new_y = lax.select(redirect_ball_new, state.last_y, new_y)

    # Check if ball missed the paddle
    not_redirected = jnp.logical_and(
        jnp.invert(redirect_ball_old), jnp.invert(redirect_ball_new)
    )

    # The game ends if the ball is on the bottom row and is not being redirected
    done = jnp.logical_and(brick_cond, not_redirected)

    # Update the strike state
    strike = jnp.bool_(strike_toggle)

    return (
        state.replace(
            ball_dir=ball_dir,
            brick_map=brick_map,
            strike=strike,
            ball_x=new_x,
            ball_y=new_y,
            done=done,
        ),
        reward,
    )

    # ORIGINAL GYMNAX VERSION
    # reward = 0

    # # Reflect the ball's direction if it hits the top border
    # border_cond_y = new_y < 0
    # new_y = jnp.clip(new_y, 0, 9)  # Ensure new_y remains within the grid

    # ball_dir = lax.select(
    #     border_cond_y,
    #     jnp.array([3, 2, 1, 0])[state.ball_dir],
    #     state.ball_dir,
    # )

    # # --- brick collision ---
    # strike_toggle = jnp.logical_and(
    #     jnp.invert(border_cond_y), state.brick_map[new_y, new_x] == 1
    # )
    # strike_bool = jnp.logical_and(jnp.invert(state.strike), strike_toggle)
    # reward += jnp.float32(strike_bool)
    # strike = lax.select(strike_toggle, strike_bool, False)

    # # delete the brick after strike
    # brick_map = lax.select(
    #     strike_bool, state.brick_map.at[new_y, new_x].set(0), state.brick_map
    # )
    # # conditionally update the ball's position after strike
    # new_y = lax.select(strike_bool, state.last_y, new_y)
    # ball_dir = lax.select(
    #     strike_bool,
    #     jnp.array([3, 2, 1, 0])[ball_dir],
    #     ball_dir,
    # )

    # # --- wall collision ---
    # brick_cond = jnp.logical_and(jnp.invert(strike_toggle), new_y == 9)
    # # spawn new bricks if brick map is empty
    # spawn_bricks = jnp.logical_and(brick_cond, jnp.count_nonzero(brick_map) == 0)
    # brick_map = lax.select(spawn_bricks, brick_map.at[1:4, :].set(1), brick_map)

    # # redirect ball if it collides with the paddle's old position
    # redirect_ball = jnp.logical_and(brick_cond, state.ball == state.pos)
    # ball_dir = lax.select(redirect_ball, jnp.array([3, 2, 1, 0])[ball_dir], ball_dir)
    # new_y = lax.select(redirect_ball, state.last_y, new_y)

    # # redirect ball if it collides with the paddle's new position:
    # # 1. the ball has not already been redirected
    # redirect_ball_new = jnp.logical_and(brick_cond, jnp.invert(redirect_ball))
    # # 2. check whether the ball and paddle collided
    # redirect_ball_new2 = jnp.logical_and(redirect_ball_new, new_x == state.pos)
    # ball_dir = lax.select(
    #     redirect_ball_new2, jnp.array([2, 3, 0, 1])[ball_dir], ball_dir
    # )
    # new_y = lax.select(redirect_ball_new, state.last_y, new_y)
    # # check whether both redirect flags are false
    # not_redirected = jnp.logical_and(
    #     jnp.invert(redirect_ball), jnp.invert(redirect_ball_new)
    # )
    # # the episode ends
    # terminal = jnp.logical_and(brick_cond, not_redirected)
    # strike = jnp.bool_(strike_toggle)

    # return (
    #     state.replace(
    #         ball_dir=ball_dir,
    #         brick_map=brick_map,
    #         strike=strike,
    #         ball_x=new_x,
    #         ball_y=new_y,
    #         terminal=terminal,
    #     ),
    #     reward,
    # )
