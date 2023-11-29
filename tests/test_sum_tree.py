import chex
import jax.numpy as jnp
from absl.testing import parameterized
from jax import random

from src.utils import SumTree


class SumTreeTests(chex.TestCase, parameterized.TestCase):
    @chex.variants(without_jit=True)
    def test_initialization(self):
        capacity = 4
        sum_tree = self.variant(SumTree)(capacity)

        assert sum_tree.capacity == capacity

    @chex.variants(with_jit=True, without_jit=True)
    def test_add_update(self):
        """
        Add a priority and test if it's updated correctly
        """
        capacity = 4
        sum_tree = SumTree(capacity)
        tree = jnp.zeros(2 * capacity - 1)

        priority = 1.5
        tree, cursor = self.variant(sum_tree.add)(tree, priority, 0)

        assert tree[-capacity + cursor - 1] == priority
        assert tree[0] == priority  # check if the root is updated

    @chex.variants(with_jit=True, without_jit=True)
    def test_propagation(self):
        capacity = 4
        sum_tree = SumTree(capacity)
        tree = jnp.zeros(2 * capacity - 1)

        # add a priority and test propagation
        priority = 2.0
        tree, _ = self.variant(sum_tree.add)(tree, priority, 0)

        assert tree[0] == priority  # root should be updated
        assert tree[1] == priority  # first internal node should be updated

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("seed_0", 0),
        ("seed_1", 1),
        ("seed_2", 2),
        ("seed_3", 3),
    )
    def test_get_leaf(self, seed: int):
        capacity = 4
        sum_tree = SumTree(capacity)
        tree = jnp.zeros(2 * capacity - 1)

        # Add priorities and test leaf retrieval
        for i in range(capacity):
            tree, _ = self.variant(sum_tree.add)(tree, float(i + 1), i)

        value = random.uniform(random.PRNGKey(seed), (1,), minval=0, maxval=tree[0])[0]
        _, sample_idx, leaf_val = self.variant(sum_tree.get_leaf)(tree, value)

        self.assertTrue(0 <= sample_idx < capacity)
        self.assertEqual(leaf_val, tree[capacity - 1 + sample_idx])

    @chex.variants(with_jit=True, without_jit=True)
    def test_sample_batch(self):
        capacity = 4
        sum_tree = SumTree(capacity)
        tree = jnp.zeros(2 * capacity - 1)

        # Add priorities
        for i in range(capacity):
            tree, _ = self.variant(sum_tree.add)(tree, float(i + 1), i)

        batch_size = 8
        values = random.uniform(random.PRNGKey(0), (batch_size,))
        sampled_idxs = self.variant(sum_tree.sample_batch)(tree, values)

        assert jnp.max(jnp.array(sampled_idxs)) < capacity
        assert jnp.max(jnp.array(sampled_idxs)) >= 0

    @chex.variants(with_jit=True, without_jit=True)
    def test_sum_tree_addition_and_structure(self):
        """
        Test that adding values to the SumTree updates its structure correctly.
        Specifically, it tests if the tree's internal nodes correctly sum the
        priorities of their child nodes.
        """
        capacity = 5
        tree = jnp.zeros(2 * capacity - 1)
        sum_tree = SumTree(capacity)

        expected = jnp.array([70.0, 48.0, 22.0, 30.0, 18.0, 10.0, 12.0, 14.0, 16.0])

        tree, cursor = self.variant(sum_tree.add)(tree, 0.0, 0)
        values = jnp.arange(10) * 2
        for i in range(10):
            tree, cursor = self.variant(sum_tree.add)(tree, values[i], cursor)

        assert jnp.all(tree == expected)
