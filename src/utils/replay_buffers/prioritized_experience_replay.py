from typing import Tuple

import jax.numpy as jnp
from jax import lax

from .base_buffer import BaseReplayBuffer


class PrioritizedExperienceReplay(BaseReplayBuffer):
    def __init__(self, buffer_size: int, batch_size: int) -> None:
        super().__init__(buffer_size, batch_size)
        self.sum_tree = SumTree(buffer_size)


class SumTree:
    def __init__(self, capacity: int) -> None:
        """
        Args:
            capacity (int): The maximum number of leaves (priorities/experiences)
            the tree can hold.
        """
        self.capacity = capacity

    def add(
        self, tree: jnp.ndarray, priority: float, cursor: int
    ) -> Tuple[jnp.ndarray, int]:
        """
        Add a new priority to the tree and update the cursor position.

        Args:
            tree (jnp.ndarray): The current state of the sum tree.
            priority (float): The priority value of the new experience.
            cursor (int): The current write cursor in the tree.

        Returns:
            Tuple[jnp.ndarray, int]: The updated tree and cursor.
        """
        idx = cursor + self.capacity - 1
        tree = self.update(tree, idx, priority)
        cursor = lax.select(cursor + 1 >= self.capacity, 0, cursor + 1)
        return tree, cursor

    def update(self, tree: jnp.ndarray, idx: int, priority: float) -> jnp.ndarray:
        """
        Update a priority in the tree at a specific index and propagate the change.

        Args:
            tree (jnp.ndarray): The current state of the sum tree.
            idx (int): The index in the tree where the priority is to be updated.
            priority (float): The new priority value.

        Returns:
            jnp.ndarray: The updated tree after the priority change.
        """
        change = priority - tree.at[idx].get()
        tree = tree.at[idx].set(priority)
        return self._propagate(tree, idx, change)

    @staticmethod
    def _propagate(tree: jnp.ndarray, idx: int, change: float) -> jnp.ndarray:
        """
        Propagate the changes in priority up the tree from a given index.

        Args:
            tree (jnp.ndarray): The current state of the sum tree.
            idx (int): The index of the tree where the priority was updated.
            change (float): The amount of change in priority.

        Returns:
            jnp.ndarray: The updated tree after propagation.
        """

        def _cond_fn(val: tuple):
            idx, _ = val
            return idx != 0

        def _while_body(val: tuple):
            idx, tree = val
            parent_idx = (idx - 1) // 2
            tree = tree.at[parent_idx].add(change)
            return parent_idx, tree

        val_init = (idx, tree)
        _, tree = lax.while_loop(_cond_fn, _while_body, val_init)
        return tree

    def get_leaf(self, tree: jnp.ndarray, value: float) -> Tuple[int, int, float]:
        """
        Retrieve the index and value of a leaf based on a given value.

        Args:
            tree (jnp.ndarray): The current state of the sum tree.
            value (float): A value to query the tree with.

        Returns:
            Tuple[int, int, float]: The index of the tree, index of the sample, and value of the leaf.
        """
        idx = self._retrieve(tree, value)
        sample_idx = idx - self.capacity + 1
        return idx, sample_idx, tree[idx]

    def _retrieve(self, tree: jnp.ndarray, value: float):
        """
        Recursively search the tree to find a leaf node based on a given value.

        Args:
            tree (jnp.ndarray): The current state of the sum tree.
            value (float): The value used to find a leaf node.

        Returns:
            int: The index of the leaf node that matches the given value.
        """

        def _cond_fn(val: tuple):
            idx, _ = val
            left = 2 * idx + 1
            # continue until a leaf node is reached
            return left < len(tree)

        def _while_body(val: tuple):
            idx, value = val
            left = 2 * idx + 1
            right = left + 1
            new_idx = lax.select(value <= tree[left], left, right)
            new_value = lax.select(value <= tree[left], value, value - tree[left])
            return new_idx, new_value

        val_init = (0, value)
        idx, _ = lax.while_loop(_cond_fn, _while_body, val_init)
        return idx
