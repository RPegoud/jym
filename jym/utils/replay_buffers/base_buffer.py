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
        buffer_state: dict,
        experience: tuple,
        idx: int,
    ):
        state, action, reward, next_state, done = experience
        idx = idx % self.buffer_size

        buffer_state["states"] = buffer_state["states"].at[idx].set(state)
        buffer_state["actions"] = buffer_state["actions"].at[idx].set(action)
        buffer_state["rewards"] = buffer_state["rewards"].at[idx].set(reward)
        buffer_state["next_states"] = (
            buffer_state["next_states"].at[idx].set(next_state)
        )
        buffer_state["dones"] = buffer_state["dones"].at[idx].set(done)

        return buffer_state

    @abstractmethod
    def sample(self):
        pass
