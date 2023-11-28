import jax.numpy as jnp
from jax import jit, lax, random
from jax_tqdm import loop_tqdm

from ...agents import SimpleBandit
from ...envs import BanditsBaseEnv


def bandits_rollout(
    key: random.PRNGKey,
    timesteps: int,
    bandits_q: list,
    epsilon: float,
    env: BanditsBaseEnv,
    agent: SimpleBandit,
    policy,
):
    """
    Trains a single bandit agent.
    """

    @jit
    @loop_tqdm(timesteps)
    def fori_body(i: int, val: tuple):
        q_values, pulls, key, rewards = val
        action, key = policy(key, env.K, q_values, epsilon)
        reward, key = env.get_reward(key, action, bandits_q)
        q_values, pulls = agent(action, reward, pulls, q_values)
        rewards = rewards.at[i].set(reward)

        val = (q_values, pulls, key, rewards)

        return val

    q_values = jnp.zeros(env.K)
    pulls = jnp.zeros(env.K)
    rewards = jnp.zeros(timesteps)

    val_init = (q_values, pulls, key, rewards)
    val = lax.fori_loop(0, timesteps, fori_body, val_init)

    return val


def bandits_parallel_rollout(
    key: random.PRNGKey,
    timesteps: int,
    n_env: int,
    bandits_q: list,
    epsilons: list,
    env: BanditsBaseEnv,
    agent: SimpleBandit,
    policy,
):
    """
    Trains multiple bandit agents in parallel, each with an independent
    epsilon parameter.
    """

    @jit
    @loop_tqdm(timesteps)
    def fori_body(i: int, val: tuple):
        q_values, pulls, keys, rewards = val
        action, keys = policy.batched_call(keys, env.K, q_values, epsilons)
        reward, keys = env.get_batched_reward(keys, action, bandits_q)
        q_values, pulls = agent.batch_update(action, reward, pulls, q_values)
        rewards = rewards.at[i].set(reward)

        val = (q_values, pulls, keys, rewards)

        return val

    keys = random.split(key, (n_env,))
    q_values = jnp.zeros([env.K, n_env])
    pulls = jnp.zeros([env.K, n_env])
    rewards = jnp.zeros([timesteps, n_env])

    val_init = (q_values, pulls, keys, rewards)
    val = lax.fori_loop(0, timesteps, fori_body, val_init)

    return val


def bandits_multi_run_parallel_rollout(
    key: random.PRNGKey,
    timesteps: int,
    n_env: int,
    n_runs: int,
    bandits_q: jnp.array,
    epsilons: list,
    env: BanditsBaseEnv,
    agent: SimpleBandit,
    policy,
):
    """
    Trains multiple bandit agents for multiple runs in parallel,
    each with an independent epsilon parameter.
    """

    @jit
    @loop_tqdm(timesteps)
    def fori_body(i: int, val: tuple):
        q_values, pulls, keys, rewards = val
        action, keys = policy.multi_run_batched_call(keys, env.K, q_values, epsilons)
        reward, keys = env.multi_run_batched_reward(keys, action, bandits_q)
        q_values, pulls = agent.multi_run_batch_update(action, reward, pulls, q_values)
        rewards = rewards.at[i].set(reward)

        val = (q_values, pulls, keys, rewards)

        return val

    keys = random.split(key, (n_env, n_runs))
    q_values = jnp.zeros([env.K, n_env, n_runs])
    pulls = jnp.zeros([env.K, n_env, n_runs])
    rewards = jnp.zeros([timesteps, n_env, n_runs])

    val_init = (q_values, pulls, keys, rewards)
    val = lax.fori_loop(0, timesteps, fori_body, val_init)

    return val
