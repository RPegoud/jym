from functools import partial

import jax.numpy as jnp
from jax import jit, random, vmap

from .bandits_base_env import BanditsBaseEnv


class K_armed_bandits(BanditsBaseEnv):
    def __init__(self, K: int, SEED: int) -> None:
        super(K_armed_bandits, self).__init__()
        self.K = K
        self.actions = jnp.arange(K)
        self.init_key = random.PRNGKey(SEED)
        self.bandits_q = random.normal(self.init_key, shape=(K,))

    def render(self):
        """
        Renders a violin plot of the reward distribution of each bandit
        """
        import pandas as pd
        import plotly.express as px

        samples = random.normal(self.init_key, (self.K, 1000))
        samples_df = pd.DataFrame(samples).T
        shifted = samples_df + pd.Series(self.bandits_q)
        melted = shifted.melt()
        melted["mean"] = melted.variable.apply(lambda x: self.bandits_q[x])
        fig = px.violin(
            melted,
            x="variable",
            y="value",
            color="variable",
            title=f"{self.K}-armed Bandits testbed",
        )
        fig.update_traces(meanline_visible=True)
        fig.show()

    def __repr__(self) -> str:
        return str(self.__dict__)

    def get_reward(self, key, action):
        key, subkey = random.split(key)
        return random.normal(subkey) + self.bandits_q[action], subkey

    @partial(jit, static_argnums=(0,))
    def get_batched_reward(self, key, action):
        return vmap(
            K_armed_bandits.get_reward,
            in_axes=(None, 0, 0),
        )(self, key, action)

    @partial(jit, static_argnums=(0,))
    def multi_run_batched_reward(self, key, action):
        return vmap(
            K_armed_bandits.get_batched_reward,
            in_axes=(None, 1, -1),
            out_axes=(1, 1),
        )(self, key, action)
