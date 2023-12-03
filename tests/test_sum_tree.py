import chex
import jax.numpy as jnp
from absl.testing import parameterized
from jax import random

from jym.utils import SumTree


class SumTreeTests(chex.TestCase, parameterized.TestCase):
    @chex.variants(without_jit=True)
    def test_initialization(self):
        capacity = 4
        batch_size = 2

        sum_tree = self.variant(SumTree)(capacity, batch_size)

        assert sum_tree.capacity == capacity

    @chex.variants(with_jit=True, without_jit=True)
    def test_add_update_propagation(self):
        """
        Add a priority and test if it's updated correctly
        """
        capacity = 10
        batch_size = 2

        sum_tree = SumTree(capacity, batch_size)
        tree = jnp.zeros(2 * capacity - 1)

        idx = 0
        priority = 1.5
        tree = self.variant(sum_tree.add)(tree, idx, priority)

        assert tree[idx + capacity - 1] == priority  # the leaf should be updated
        assert tree[0] == priority  # the root should be updated
        assert tree[1] == priority  # the first internal node should be updated

        idx = 2
        priority = 2.0
        tree = self.variant(sum_tree.update)(tree, idx, priority)
        assert tree[idx + capacity - 1] == priority

    @chex.variants(with_jit=True, without_jit=True)
    @parameterized.named_parameters(
        ("seed_0", 0),
        ("seed_1", 1),
        ("seed_2", 2),
        ("seed_3", 3),
    )
    def test_get_leaf(self, seed: int):
        capacity = 4
        batch_size = 2

        sum_tree = SumTree(capacity, batch_size)
        tree = jnp.zeros(2 * capacity - 1)

        # Add priorities and test leaf retrieval
        for i in range(capacity):
            tree = self.variant(sum_tree.add)(tree, i, float(i + 1))

        value = random.uniform(random.PRNGKey(seed), (1,), minval=0, maxval=tree[0])[0]
        _, sample_idx, leaf_val = self.variant(sum_tree.get_leaf)(tree, value)

        self.assertTrue(0 <= sample_idx < capacity)
        self.assertEqual(leaf_val, tree[capacity - 1 + sample_idx])

    @chex.variants(with_jit=True, without_jit=True)
    def test_sample_batch(self):
        capacity = 4
        batch_size = 2

        sum_tree = SumTree(capacity, batch_size)
        tree = jnp.zeros(2 * capacity - 1)

        # Add priorities
        for i in range(capacity):
            tree = self.variant(sum_tree.add)(tree, i, float(i + 1))

        batch_size = 8
        values = random.uniform(random.PRNGKey(0), (batch_size,))
        sampled_idxs = self.variant(sum_tree.sample_idx_batch)(tree, values)

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
        batch_size = 2

        tree = jnp.zeros(2 * capacity - 1)
        sum_tree = SumTree(capacity, batch_size)

        expected = jnp.array([25.0, 10.0, 15.0, 7.0, 3.0, 8.0, 7.0, 3.0, 4.0])

        values = jnp.arange(10)
        for i in range(10):
            tree = self.variant(sum_tree.add)(tree, values[i], i)

        assert jnp.all(tree == expected)
