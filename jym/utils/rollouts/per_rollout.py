from functools import partial
from typing import Callable, List

import haiku as hk
import jax.numpy as jnp
import optax
from jax import jit, lax, random, vmap
from jax_tqdm import loop_tqdm

from ...agents import BaseDeepRLAgent
from ...envs import BaseEnv
from ..replay_buffers import Experience, PrioritizedExperienceReplay


@partial(vmap, in_axes=(None, None, None, None))
def compute_td_error(
    model: hk.Transformed,
    online_net_params: dict,
    target_net_params: dict,
    discount: float,
    state: jnp.ndarray,
    action: jnp.ndarray,
    reward: jnp.ndarray,
    next_state: jnp.ndarray,
    done: jnp.ndarray,
    priority: jnp.ndarray,  # unused
) -> List[float]:
    """
    Computes the td errors for a batch of experiences.
    Errors are clipped to [-1, 1] for statibility reasons.
    """
    td_target = (
        (1 - done)
        * discount
        * jnp.max(model.apply(target_net_params, None, next_state))
    )
    prediction = model.apply(online_net_params, None, state)[action]
    return jnp.clip(reward + td_target - prediction, a_min=-1, a_max=1)


def per_rollout(
    timesteps: int,
    random_seed: int,
    target_net_update_freq: int,
    model: hk.Transformed,
    optimizer: optax.GradientTransformation,
    buffer_state: dict,
    tree_state: jnp.ndarray,
    agent: BaseDeepRLAgent,
    env: BaseEnv,
    state_shape: int,
    buffer_size: int,
    batch_size: int,
    alpha: float,
    beta: float,
    discount: float,
    epsilon_decay_fn: Callable,
    decay_params: dict,
) -> dict[jnp.ndarray | dict]:
    @loop_tqdm(timesteps)
    @jit
    def _fori_body(i: int, val: tuple):
        (
            online_net_params,
            target_net_params,
            optimizer_state,
            buffer_state,
            tree_state,
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
        action, action_key = agent.act(action_key, online_net_params, obs, epsilon)
        state, next_state, reward, done, env_key = env.step(state, env_key, action)
        experience = Experience(
            state=env._get_obs(state),
            action=action,
            reward=reward,
            next_state=next_state,
            done=done,
        )

        buffer_state, tree_state = replay_buffer.add(
            tree_state, buffer_state, i, experience
        )

        (
            experiences_batch,
            sample_indexes,
            importance_weights,
            buffer_key,
        ) = replay_buffer.sample(buffer_key, buffer_state, tree_state)

        # compute individual td errors for the sampled batch and
        # update the tree state using the batched absolute td errors
        td_errors = compute_td_error(
            model, online_net_params, target_net_params, discount, **experiences_batch
        )

        tree_state = replay_buffer.sum_tree.batch_update(
            tree_state, sample_indexes, jnp.abs(td_errors)
        )
        online_net_params, optimizer_state, loss = agent.update(
            online_net_params,
            target_net_params,
            optimizer,
            optimizer_state,
            importance_weights,
            experiences_batch,
        )

        # update the target parameters every ``target_net_update_freq`` steps
        target_net_params = lax.cond(
            i % target_net_update_freq == 0,
            lambda _: online_net_params,
            lambda _: target_net_params,
            operand=None,
        )

        all_actions = all_actions.at[i].set(action)
        all_obs = all_obs.at[i].set(next_state)
        all_rewards = all_rewards.at[i].set(reward)
        all_done = all_done.at[i].set(done)
        losses = losses.at[i].set(loss)

        val = (
            online_net_params,
            target_net_params,
            optimizer_state,
            buffer_state,
            tree_state,
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

    online_net_params = model.init(init_key, jnp.zeros(state_shape))
    target_net_params = model.init(action_key, jnp.zeros(state_shape))
    optimizer_state = optimizer.init(online_net_params)
    replay_buffer = PrioritizedExperienceReplay(buffer_size, batch_size, alpha, beta)

    val_init = (
        online_net_params,
        target_net_params,
        optimizer_state,
        buffer_state,
        tree_state,
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
        "online_net_params",
        "target_net_params",
        "optimizer_state",
        "buffer_state",
        "tree_state",
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
