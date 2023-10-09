from jax import lax, random


class SkeletonEnv:
    """
    env_state (tuple): state, key
    """

    def __init__(self):
        self.random_limit = 0.05

    def _get_obs(self, state):
        """
        Any processing of the state should go here
        """
        return state

    def _reset(self, key):
        new_state = random.uniform(
            key,
            minval=-self.random_limit,
            max=self.random_limit,
            shape=(4,),
        )
        new_key = random.split(key)[0]
        return new_state, new_key

    def _maybe_reset(self, env_state, done):
        key = env_state[1]
        return lax.cond(
            done,
            self._reset,
            lambda key: env_state,
            key,
        )

    def step(self, env_state, action):
        state, key = env_state
        new_state = state + action

        reward, done, info = 1.0, False, None

        env_state = new_state, key
        env_state = self._maybe_reset(env_state, done)
        new_state = env_state[0]

        return env_state, self._get_obs(new_state), reward, done, info

    def reset(self, key):
        env_state = self._reset(key)
        new_state = env_state[0]
        return env_state, self._get_obs(new_state)


if __name__ == "__main__":
    key = random.PRNGKey(0)
    env = SkeletonEnv()
    env_state, initial_obs = env.reset(key)
    action = 1
    env_state, obs, reward, done, info = env.step(env_state, 1)
