from abc import ABC, abstractmethod


class BaseDeepRLAgent(ABC):
    def __init__(
        self,
        discount,
        learning_rate,
        num_actions,
    ) -> None:
        self.discount = discount
        self.learning_rate = learning_rate
        self.num_actions = num_actions

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def store(self):
        pass

    @abstractmethod
    def act(self):
        pass
