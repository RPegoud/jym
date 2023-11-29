from abc import ABC, abstractmethod


class BaseTabularAgent(ABC):
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
