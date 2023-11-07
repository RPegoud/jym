from abc import ABC, abstractmethod
from dataclasses import dataclass

import jax.numpy as jnp
from jax import lax


@dataclass
class Experience:
    state: jnp.ndarray
    action: int
    reward: float
    next_state: jnp.ndarray
    done: bool


class BaseReplayBuffer(ABC):
    def __init__(
        self,
        size: int,
    ) -> None:
        self.size = size
        self.buffer = {}
        self.idx = 0

    def add(self, experience: Experience):
        self.buffer[self.idx] = experience
        # conditionally reset the index
        self.idx = lax.cond(
            self.idx < self.size - 1,
            lambda _: self.idx + 1,
            lambda _: 0,
            operand=None,
        ).item()

    @abstractmethod
    def sample(self):
        pass
