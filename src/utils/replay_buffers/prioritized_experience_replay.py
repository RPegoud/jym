from .base_buffer import BaseReplayBuffer


class PrioritizedExperienceReplay(BaseReplayBuffer):
    def __init__(self, buffer_size: int, batch_size: int) -> None:
        super().__init__(buffer_size, batch_size)
