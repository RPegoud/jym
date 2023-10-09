from abc import ABC, abstractmethod
from typing import Any


class BasePolicy(ABC):
    def __init__(self) -> None:
        pass

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.call(*args, **kwargs)

    @abstractmethod
    def call(self, key, q_values):
        pass
