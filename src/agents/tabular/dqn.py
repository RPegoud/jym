from base_agent import BaseAgent

# from haiku import linea


class DQN(BaseAgent):
    def __init__(self, key, n_states, n_actions, discount) -> None:
        self.epsilon = 0.1
        self.learning_rate = 0.1
        self.discount = 0.9

        # self.net =
