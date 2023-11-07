from abc import ABC, abstractmethod


class BaseAgent(ABC):
    def __init__(
        self,
        discount,
        learning_rate,
    ) -> None:
        self.discount = discount
        self.learning_rate = learning_rate

    @abstractmethod
    def update(self):
        pass
