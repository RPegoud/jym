import jax.numpy as jnp

from .base_agent import BaseDeepRLAgent


class DQN(BaseDeepRLAgent):
    def __init__(
        self,
        discount: float,
        learning_rate: float,
        num_actions: int,
        model,
    ) -> None:
        super(DQN, self).__init__(
            discount,
            learning_rate,
            num_actions,
        )

        self.model = model

    def update(self):
        raise NotImplementedError

    def store(self):
        raise NotImplementedError

    def act(self, params: dict, state: jnp.ndarray):
        q_values = self.model.apply(params, state)
        return jnp.argmax(q_values)
