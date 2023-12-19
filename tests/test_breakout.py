import chex
import jax.numpy as jnp
from absl.testing import parameterized
from jax import random

from jym import Breakout


@chex.dataclass
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


class BreakoutTests(chex.TestCase, parameterized.TestCase):
    @parameterized.named_parameters(
        # random.uniform(key) => 0.41 => left init
        ("seed_0", 0, {"x": 9, "y": 3, "ball_dir": 3}),
        # random.uniform(key) => 0.11 => left init
        ("seed_1", 1, {"x": 9, "y": 3, "ball_dir": 3}),
        # random.uniform(key) => 0.86 => right init
        ("seed_2", 3, {"x": 0, "y": 3, "ball_dir": 2}),
    )
    def test_reset(self, seed: int, expected):
        """
        Test random setting of ball x, y and direction.
        """
        key = random.PRNGKey(seed)
        env = Breakout()
        state, obs, env_key = env.reset(key)

        expected_state = EnvState(
            ball_x=expected["x"],
            last_x=expected["x"],
            ball_y=expected["y"],
            last_y=expected["y"],
            ball_dir=expected["ball_dir"],
            pos=4,
            brick_map=jnp.zeros((10, 10)).at[1:4, :].set(1),
            strike=False,
            time=0,
            done=False,
        )

        for field in state:
            assert jnp.all(state[field] == expected_state[field])

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        # initial position: 4
        ("case_no_movement", 0, 4),  # no movement => expected pos = 4
        ("case_moving_left", 1, 3),  # moving left => expected pos = 3
        ("case_moving_right", 2, 5),  # moving right=> expected pos = 5
    )
    def test_player_movement(
        self,
        action,
        expected,
    ):
        """
        Tests player movement within the _agent_step function
        """
        key = random.PRNGKey(0)
        env = Breakout()
        state, obs, env_key = self.variant(env.reset)(key)
        state, new_x, new_y = self.variant(env._agent_step)(state, action)
        assert state.pos == expected

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("case_up_left", {"start_pos": (4, 4), "ball_dir": 0, "next_pos": (3, 3)}),
        ("case_up_right", {"start_pos": (4, 4), "ball_dir": 1, "next_pos": (5, 3)}),
        ("case_down_right", {"start_pos": (0, 3), "ball_dir": 2, "next_pos": (1, 4)}),
        ("case_down_left", {"start_pos": (9, 3), "ball_dir": 3, "next_pos": (8, 4)}),
    )
    def test_ball_movement(
        self,
        expected,
    ):
        """
        Tests ball movement within the _agent_step function.
        """
        env = Breakout()
        state = EnvState(
            ball_x=expected["start_pos"][0],
            last_x=expected["start_pos"][0],
            ball_y=expected["start_pos"][1],
            last_y=expected["start_pos"][1],
            ball_dir=expected["ball_dir"],
            pos=4,
            brick_map=jnp.zeros((10, 10)).at[1:4, :].set(1),
            strike=False,
            time=0,
            done=False,
        )
        state, new_x, new_y = self.variant(env._agent_step)(state, action=0)
        chex.assert_trees_all_equal((new_x, new_y), expected["next_pos"])
        assert state.ball_dir == expected["ball_dir"]  # there was no collision

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        (
            "case_down_left",
            {
                "start_coord": (0, 6),
                "next_coord": (1, 7),
                "next_next_coord": (2, 8),
                "ball_dir": 3,
                "next_ball_dir": 2,
            },
        ),
        # (
        #     "case_up_left",
        #     {
        #         "start_coord": (0, 5),
        #         "ball_dir": 0,
        #         "next_ball_dir": 1,
        #         "next_coord": (1, 4),
        #     },
        # ),
        # (
        #     "case_down_right",
        #     {
        #         "start_coord": (9, 5),
        #         "ball_dir": 1,
        #         "next_ball_dir": 0,
        #         "next_coord": (8, 6),
        #     },
        # ),
        # (
        #     "case_up_right",
        #     {
        #         "start_coord": (9, 6),
        #         "ball_dir": 2,
        #         "next_ball_dir": 3,
        #         "next_coord": (8, 5),
        #     },
        # ),
    )
    def test_ball_rebound_agent_step(self, expected):
        """
        Tests the ball position and direction update after a lateral rebound
        within _agent_step.
        """
        env = Breakout()
        state = EnvState(
            ball_x=expected["start_coord"][0],
            last_x=expected["start_coord"][0],
            ball_y=expected["start_coord"][1],
            last_y=expected["start_coord"][1],
            ball_dir=expected["ball_dir"],
            pos=4,
            brick_map=jnp.zeros((10, 10)).at[1:4, :].set(1),
            strike=False,
            time=0,
            done=False,
        )
        state, new_x, new_y = self.variant(env._agent_step)(state, action=0)
        state, reward = self.variant(env._step_ball_brick)(state, new_x, new_y)
        assert (state.ball_dir) == expected["next_ball_dir"]
        chex.assert_trees_all_equal((new_x, new_y), expected["next_coord"])
        assert reward == 0.0
        state, new_x, new_y = self.variant(env._agent_step)(state, action=0)
        state, reward = self.variant(env._step_ball_brick)(state, new_x, new_y)
        chex.assert_trees_all_equal((new_x, new_y), expected["next_next_coord"])

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        (
            "case_termination_down_right",
            {"start_coord": (1, 8), "ball_dir": 2, "pos": 9, "next_coord": (2, 9)},
        ),
        (
            "case_termination_down_left",
            {"start_coord": (1, 8), "ball_dir": 3, "pos": 9, "next_coord": (0, 9)},
        ),
        (
            "case_termination_left_corner",
            {"start_coord": (8, 8), "ball_dir": 2, "pos": 0, "next_coord": (9, 9)},
        ),
        (
            "case_termination_right_corner",
            {"start_coord": (1, 8), "ball_dir": 3, "pos": 9, "next_coord": (0, 9)},
        ),
    )
    def test_termination(
        self,
        expected,
    ):
        env = Breakout()
        state = EnvState(
            ball_x=expected["start_coord"][0],
            last_x=expected["start_coord"][0],
            ball_y=expected["start_coord"][1],
            last_y=expected["start_coord"][1],
            ball_dir=expected["ball_dir"],
            pos=expected["pos"],
            brick_map=jnp.zeros((10, 10)).at[1:4, :].set(1),
            strike=False,
            time=0,
            done=False,
        )
        state, new_x, new_y = self.variant(env._agent_step)(state, action=0)
        chex.assert_trees_all_equal((new_x, new_y), expected["next_coord"])
        state, reward = self.variant(env._step_ball_brick)(state, new_x, new_y)
        assert state.done == jnp.array(True, dtype=jnp.bool_)
        assert reward == 0.0
