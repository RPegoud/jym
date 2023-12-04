import chex
import jax.numpy as jnp
from absl.testing import parameterized
from jax import random

from jym.utils import Experience, PrioritizedExperienceReplay


class SumTreeTests(chex.TestCase, parameterized.TestCase):
    def test_initialization(self):
        buffer_size = 16
        batch_size = 8
        alpha, beta = 0.5, 0.5
        per = PrioritizedExperienceReplay(buffer_size, batch_size, alpha, beta)

        assert per.alpha == alpha
        assert per.beta == beta
        assert per.buffer_size == buffer_size
        assert per.batch_size == batch_size

    @chex.variants(with_jit=True, without_jit=True)
    def test_add_update(self):
        buffer_size = 16
        batch_size = 8
        state_shape = (4,)

        alpha, beta = 0.5, 0.5
        tree_state = jnp.zeros(2 * buffer_size - 1)
        buffer_state = {
            "state": jnp.empty((buffer_size, *state_shape), dtype=jnp.float32),
            "action": jnp.empty((buffer_size,), dtype=jnp.int32),
            "reward": jnp.empty((buffer_size,), dtype=jnp.int32),
            "next_state": jnp.empty((buffer_size, *state_shape), dtype=jnp.float32),
            "done": jnp.empty((buffer_size,), dtype=jnp.bool_),
            "priority": jnp.empty((buffer_size), dtype=jnp.float32),
        }
        per = PrioritizedExperienceReplay(buffer_size, batch_size, alpha, beta)

        exp = Experience(
            state=jnp.ones(4),
            action=jnp.int32(3),
            reward=jnp.float32(1),
            next_state=jnp.ones(4) * 2,
            done=jnp.bool_(False),
        )

        exp2 = Experience(
            state=jnp.ones(4),
            action=jnp.int32(3),
            reward=jnp.float32(1),
            next_state=jnp.ones(4) * 2,
            done=jnp.bool_(False),
            priority=2.0,
        )

        # test priority initialization
        assert exp.priority == 0.0
        assert exp2.priority == 2.0

        buffer_state, tree_state = self.variant(per.add)(
            tree_state, buffer_state, 0, exp
        )

        # test tree_state update propagation
        chex.assert_trees_all_equal(
            jnp.nonzero(tree_state)[0], jnp.array([0, 1, 3, 7, 15])
        )
        # ensure the priority is set to 1.0 for an empty buffer
        assert buffer_state["priority"][0] == 1.0

        buffer_state, tree_state = self.variant(per.add)(
            tree_state, buffer_state, 1, exp2
        )
        # ensure priority is clipped to max(buffer.priority)
        assert buffer_state["priority"][1] == max(buffer_state["priority"])

        # test update
        td_error = 5.0
        tree_state = self.variant(per.update)(tree_state, td_error, 1)
        assert tree_state[buffer_size] == td_error**alpha
        assert tree_state[0] == 1.0 + td_error**alpha

    @chex.variants(with_jit=True, without_jit=True)
    def test_sample(self):
        buffer_size = 16
        batch_size = 8
        state_shape = (4,)

        tree_state = jnp.zeros(2 * buffer_size - 1)
        buffer_state = {
            "state": jnp.empty((buffer_size, *state_shape), dtype=jnp.float32),
            "action": jnp.empty((buffer_size,), dtype=jnp.int32),
            "reward": jnp.empty((buffer_size,), dtype=jnp.int32),
            "next_state": jnp.empty((buffer_size, *state_shape), dtype=jnp.float32),
            "done": jnp.empty((buffer_size,), dtype=jnp.bool_),
            "priority": jnp.empty((buffer_size), dtype=jnp.float32),
        }
        per = PrioritizedExperienceReplay(buffer_size, batch_size, 0.5, 0.5)

        experiences = {
            i: Experience(
                state=jnp.ones(4),
                action=jnp.int32(3),
                reward=jnp.float32(1),
                next_state=jnp.ones(4) * 2,
                done=jnp.bool_(False),
                priority=i,  # priorities from 0 to 9
            )
            for i in range(buffer_size)
        }
        for i in range(buffer_size):
            buffer_state, tree_state = per.add(
                tree_state, buffer_state, i, experiences[i]
            )

        key = random.PRNGKey(0)
        experience_batch, _, _, key = self.variant(per.sample)(
            key, buffer_state, tree_state
        )

        expected_shape = {
            "action": jnp.zeros(batch_size),
            "done": jnp.zeros(batch_size),
            "next_state": jnp.zeros((batch_size, *state_shape)),
            "priority": jnp.zeros(batch_size),
            "reward": jnp.zeros(batch_size),
            "state": jnp.zeros((batch_size, *state_shape)),
        }

        chex.assert_trees_all_equal_shapes(experience_batch, expected_shape)
