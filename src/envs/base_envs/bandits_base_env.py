from abc import ABC, abstractmethod


class BanditsBaseEnv(ABC):
    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_reward(self, key, action):
        pass
