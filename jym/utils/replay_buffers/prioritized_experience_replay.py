from functools import partial
from typing import Tuple

import jax.numpy as jnp
from jax import lax, vmap

from .base_buffer import BaseReplayBuffer


class PrioritizedExperienceReplay(BaseReplayBuffer):
    def __init__(self, buffer_size: int, batch_size: int) -> None:
        super().__init__(buffer_size, batch_size)
        self.sum_tree = SumTree(buffer_size)


class SumTree:
    """
    SumTree utilities used to manipulate an external tree state
    """

    def __init__(self, capacity: int, batch_size: int) -> None:
        """
        Args:
            capacity (int): The maximum number of leaves (priorities/experiences)
            the tree can hold.
            batch_size (int): The number of experiences to sample in a minibatch
        """
        self.capacity = capacity
        self.batch_size = batch_size

    def add(self, tree_state: jnp.ndarray, idx: int, priority: float) -> jnp.ndarray:
        """
        Add a new priority to the tree and update the cursor position.

        Args:
            tree_state (jnp.ndarray): The current state of the sum tree.
            priority (float): The priority value of the new experience.
            idx (int): The current write cursor in the tree.

        Returns:
            jnp.ndarray: The updated tree_state and cursor.
        """
        idx = idx + self.capacity - 1
        tree_state = self.update(tree_state, idx, priority)
        return tree_state

    def update(self, tree_state: jnp.ndarray, idx: int, priority: float) -> jnp.ndarray:
        """
        Update a priority in the tree at a specific index and propagate the change.

        Args:
            tree_state (jnp.ndarray): The current state of the sum tree.
            idx (int): The index in the tree where the priority is to be updated.
            priority (float): The new priority value.

        Returns:
            jnp.ndarray: The updated tree after the priority change.
        """
        # ensures only leaf nodes are updated manually
        idx = lax.select(idx >= self.capacity - 1, idx, idx + self.capacity - 1)
        change = priority - tree_state.at[idx].get()
        tree_state = tree_state.at[idx].set(priority)
        return self._propagate(tree_state, idx, change)

    @staticmethod
    def _propagate(tree_state: jnp.ndarray, idx: int, change: float) -> jnp.ndarray:
        """
        Propagate the changes in priority up the tree from a given index.

        Args:
            tree_state (jnp.ndarray): The current state of the sum tree.
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

        val_init = (idx, tree_state)
        _, tree_state = lax.while_loop(_cond_fn, _while_body, val_init)
        return tree_state

    @partial(vmap, in_axes=(None, None, 0))
    def sample_idx_batch(self, tree_state, value) -> Tuple[int, int, float]:
        """
        Applies the get_leaf function to a batch of values,
        used for sampling from the replay buffer.
        """
        return self.get_leaf(tree_state, value)

    def get_leaf(self, tree_state: jnp.ndarray, value: float) -> Tuple[int, int, float]:
        """
        Retrieve the index and value of a leaf based on a given value.

        Args:
            tree_state (jnp.ndarray): The current state of the sum tree.
            value (float): A value to query the tree with.

        Returns:
            Tuple[int, int, float]: The index of the tree, index of the sample, and value of the leaf.
        """
        idx = self._retrieve(tree_state, value)
        sample_idx = idx - self.capacity + 1
        return idx, sample_idx, tree_state[idx]

    def _retrieve(self, tree_state: jnp.ndarray, value: float):
        """
        Recursively search the tree to find a leaf node based on a given value.

        Args:
            tree_state (jnp.ndarray): The current state of the sum tree.
            value (float): The value used to find a leaf node.

        Returns:
            int: The index of the leaf node that matches the given value.
        """

        def _cond_fn(val: tuple):
            idx, _ = val
            left = 2 * idx + 1
            # continue until a leaf node is reached
            return left < len(tree_state)

        def _while_body(val: tuple):
            idx, value = val
            left = 2 * idx + 1
            right = left + 1
            new_idx = lax.select(value <= tree_state[left], left, right)
            new_value = lax.select(
                value <= tree_state[left], value, value - tree_state[left]
            )
            return new_idx, new_value

        val_init = (0, value)
        idx, _ = lax.while_loop(_cond_fn, _while_body, val_init)
        return idx
