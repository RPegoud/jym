from abc import ABC, abstractmethod
from functools import partial

from jax import jit


class BaseReplayBuffer(ABC):
    def __init__(
        self,
        buffer_size: int,
        batch_size: int,
    ) -> None:
        self.buffer_size = buffer_size
        self.batch_size = batch_size

    @partial(jit, static_argnums=(0))
    def add(
        self,
        buffer: dict,
        experience: tuple,
        idx: int,
    ):
        state, action, reward, next_state, done = experience
        idx = idx % self.buffer_size

        buffer["states"] = buffer["states"].at[idx].set(state)
        buffer["actions"] = buffer["actions"].at[idx].set(action)
        buffer["rewards"] = buffer["rewards"].at[idx].set(reward)
        buffer["next_states"] = buffer["next_states"].at[idx].set(next_state)
        buffer["dones"] = buffer["dones"].at[idx].set(done)

        return buffer

    @abstractmethod
    def sample(self):
        pass
