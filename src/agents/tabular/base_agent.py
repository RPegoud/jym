from abc import ABC, abstractmethod

import haiku as hk


class BaseAgent(ABC):
    def __init__(
        self,
        key,
        n_states,
        n_actions,
        discount,
    ) -> None:
        self.prng = hk.PRNGSequence(key)
        self.n_states = n_states
        self.n_actions = n_actions
        self.discount = discount

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def act(self):
        pass
