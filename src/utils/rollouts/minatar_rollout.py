from typing import Callable

import haiku as hk
import jax.numpy as jnp
import optax
from jax import jit, lax, random, vmap
from jax_tqdm import loop_tqdm

from ...agents import BaseDeepRLAgent
from ...envs import BaseEnv
from ..replay_buffers import BaseReplayBuffer


def minatar_rollout(
    timesteps: int,
    random_seed: int,
    target_net_update_freq: int,
    model: hk.Transformed,
    optimizer: optax.GradientTransformation,
    buffer_state: dict,
    agent: BaseDeepRLAgent,
    env: BaseEnv,
    replay_buffer: BaseReplayBuffer,
    state_shape: int,
    buffer_size: int,
    epsilon_decay_fn: Callable,
    decay_params: dict,
):
    @loop_tqdm(timesteps)
    @jit
    def _fori_body(i: int, val: tuple):
        (
            model_params,
            target_net_params,
            optimizer_state,
            buffer_state,
            env_key,
            action_key,
            buffer_key,
            state,
            obs,
            all_actions,
            all_obs,
            all_rewards,
            all_done,
            losses,
        ) = val

        epsilon = epsilon_decay_fn(current_step=i, **decay_params)
        action, action_key = agent.act(action_key, model_params, obs, epsilon)
        state, new_state, reward, done, env_key = env.step(state, env_key, action)
        experience = (env._get_obs(state), action, reward, new_state, done)

        buffer_state = replay_buffer.add(buffer_state, experience, i)
        current_buffer_size = jnp.min(jnp.array([i, buffer_size]))

        experiences_batch, buffer_key = replay_buffer.sample(
            buffer_key,
            buffer_state,
            current_buffer_size,
        )

        model_params, optimizer_state, loss = agent.update(
            model_params,
            target_net_params,
            optimizer,
            optimizer_state,
            experiences_batch,
        )

        # update the target parameters every ``target_net_update_freq`` steps
        target_net_params = lax.cond(
            i % target_net_update_freq == 0,
            lambda _: model_params,
            lambda _: target_net_params,
            operand=None,
        )

        all_actions = all_actions.at[i].set(action)
        all_obs = all_obs.at[i].set(new_state)
        all_rewards = all_rewards.at[i].set(reward)
        all_done = all_done.at[i].set(done)
        losses = losses.at[i].set(loss)

        val = (
            model_params,
            target_net_params,
            optimizer_state,
            buffer_state,
            env_key,
            action_key,
            buffer_key,
            state,
            obs,
            all_actions,
            all_obs,
            all_rewards,
            all_done,
            losses,
        )

        return val

    init_key, action_key, buffer_key = vmap(random.PRNGKey)(jnp.arange(3) + random_seed)
    state, obs, env_key = env.reset(init_key)
    all_actions = jnp.zeros([timesteps])
    all_obs = jnp.zeros([timesteps, *state_shape])
    all_rewards = jnp.zeros([timesteps], dtype=jnp.float32)
    all_done = jnp.zeros([timesteps], dtype=jnp.bool_)
    losses = jnp.zeros([timesteps], dtype=jnp.float32)

    model_params = model.init(init_key, jnp.zeros(state_shape))
    target_net_params = model.init(action_key, jnp.zeros(state_shape))
    optimizer_state = optimizer.init(model_params)

    val_init = (
        model_params,
        target_net_params,
        optimizer_state,
        buffer_state,
        env_key,
        action_key,
        buffer_key,
        state,
        obs,
        all_actions,
        all_obs,
        all_rewards,
        all_done,
        losses,
    )

    vals = lax.fori_loop(0, timesteps, _fori_body, val_init)
    output_dict = {}
    keys = [
        "model_params",
        "target_net_params",
        "optimizer_state",
        "buffer_state",
        "env_key",
        "action_key",
        "buffer_key",
        "state",
        "obs",
        "all_actions",
        "all_obs",
        "all_rewards",
        "all_done",
        "losses",
    ]
    for idx, value in enumerate(vals):
        output_dict[keys[idx]] = value

    return output_dict


def parallel_minatar_rollout(
    timesteps: int,
    random_seeds: list,
    n_env: int,
    target_net_update_freq: int,
    model: hk.Transformed,
    optimizer: optax.GradientTransformation,
    buffer_state: dict,
    agent: BaseDeepRLAgent,
    env: BaseEnv,
    replay_buffer: BaseReplayBuffer,
    state_shape: int,
    buffer_size: int,
    epsilon_decay_fn: Callable,
    decay_params: dict,
):
    @loop_tqdm(timesteps)
    @jit
    def _fori_body(i: int, val: tuple):
        (
            model_params,
            target_net_params,
            optimizer_state,
            buffer_state,
            env_keys,
            action_keys,
            buffer_keys,
            state,
            obs,
            all_actions,
            all_obs,
            all_rewards,
            all_done,
            losses,
        ) = val

        epsilon = epsilon_decay_fn(current_step=i, **decay_params)
        action, action_keys = agent.act(action_keys, model_params, obs, epsilon)
        state, new_state, reward, done, env_keys = env.step(state, env_keys, action)
        experience = (env._get_obs(state), action, reward, new_state, done)

        buffer_state = replay_buffer.add(buffer_state, experience, i)
        current_buffer_size = jnp.min(jnp.array([i, buffer_size]))

        experiences_batch, buffer_keys = replay_buffer.sample(
            buffer_keys,
            buffer_state,
            current_buffer_size,
        )

        model_params, optimizer_state, loss = agent.update(
            model_params,
            target_net_params,
            optimizer,
            optimizer_state,
            experiences_batch,
        )

        # update the target parameters every ``target_net_update_freq`` steps
        target_net_params = lax.cond(
            i % target_net_update_freq == 0,
            lambda _: model_params,
            lambda _: target_net_params,
            operand=None,
        )

        all_actions = all_actions.at[i].set(action)
        all_obs = all_obs.at[i].set(new_state)
        all_rewards = all_rewards.at[i].set(reward)
        all_done = all_done.at[i].set(done)
        losses = losses.at[i].set(loss)

        val = (
            model_params,
            target_net_params,
            optimizer_state,
            buffer_state,
            env_keys,
            action_keys,
            buffer_keys,
            state,
            obs,
            all_actions,
            all_obs,
            all_rewards,
            all_done,
            losses,
        )

        return val

    init_keys, action_keys, buffer_keys = vmap(
        lambda seed: random.split(random.PRNGKey(seed), n_env)
    )(jnp.arange(3))
    state, obs, env_key = env.reset(init_keys)
    all_actions = jnp.zeros([timesteps])
    all_obs = jnp.zeros([timesteps, *state_shape])
    all_rewards = jnp.zeros([timesteps], dtype=jnp.float32)
    all_done = jnp.zeros([timesteps], dtype=jnp.bool_)
    losses = jnp.zeros([timesteps], dtype=jnp.float32)

    model_params = model.init(init_keys, jnp.zeros(state_shape))
    target_net_params = model.init(action_keys, jnp.zeros(state_shape))
    optimizer_state = optimizer.init(model_params)

    val_init = (
        model_params,
        target_net_params,
        optimizer_state,
        buffer_state,
        env_key,
        action_keys,
        buffer_keys,
        state,
        obs,
        all_actions,
        all_obs,
        all_rewards,
        all_done,
        losses,
    )

    vals = lax.fori_loop(0, timesteps, _fori_body, val_init)
    output_dict = {}
    keys = [
        "model_params",
        "target_net_params",
        "optimizer_state",
        "buffer_state",
        "env_key",
        "action_key",
        "buffer_key",
        "state",
        "obs",
        "all_actions",
        "all_obs",
        "all_rewards",
        "all_done",
        "losses",
    ]
    for idx, value in enumerate(vals):
        output_dict[keys[idx]] = value

    return output_dict
