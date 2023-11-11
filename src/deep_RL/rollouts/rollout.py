import haiku as hk
import jax.numpy as jnp
import optax
from jax import jit, lax, random, vmap
from jax_tqdm import loop_tqdm

from ..agents import BaseDeepRLAgent
from ..envs import BaseControlEnv
from ..replay_buffers import BaseReplayBuffer


def DeepRlRollout(
    timesteps: int,
    random_seed: int,
    target_net_update_freq: int,
    model: hk.Transformed,
    optimizer: optax.GradientTransformation,
    buffer_state: dict,
    agent: BaseDeepRLAgent,
    env: BaseControlEnv,
    replay_buffer: BaseReplayBuffer,
    state_shape: int,
    buffer_size: int = 1024,
):
    @loop_tqdm(timesteps)
    @jit
    def _fori_body(i: int, val: tuple):
        (
            model_params,
            target_net_params,
            optimizer_state,
            buffer_state,
            action_key,
            buffer_key,
            env_state,
            all_obs,
            all_rewards,
            all_done,
            losses,
        ) = val

        state, _ = env_state
        action, action_key = agent.act(action_key, model_params, state)
        env_state, obs, reward, done = env.step(env_state, action)
        experience = (state, action, reward, obs, done)

        buffer_state = replay_buffer.add(buffer_state, experience, i)
        current_buffer_size = jnp.min(jnp.array([i, buffer_size]))
        experiences_batch, buffer_key = replay_buffer.sample(
            buffer_key, buffer_state, current_buffer_size
        )

        model_params, optimizer_state, loss = agent.update(
            model_params,
            target_net_params,
            optimizer,
            optimizer_state,
            experiences_batch,
        )

        target_net_params = lax.cond(
            i % target_net_update_freq,
            lambda _: model_params,
            lambda _: target_net_params,
            operand=None,
        )

        all_obs = all_obs.at[i].set(obs)
        all_rewards = all_rewards.at[i].set(reward)
        all_done = all_done.at[i].set(done)
        losses = losses.at[i].set(loss)

        val = (
            model_params,
            target_net_params,
            optimizer_state,
            buffer_state,
            action_key,
            buffer_key,
            env_state,
            all_obs,
            all_rewards,
            all_done,
            losses,
        )

        return val

    init_key, action_key, buffer_key = vmap(random.PRNGKey)(jnp.arange(3) + random_seed)
    env_state, _ = env.reset(init_key)
    all_obs = jnp.zeros([timesteps, state_shape])
    all_rewards = jnp.zeros([timesteps], dtype=jnp.int32)
    all_done = jnp.zeros([timesteps], dtype=jnp.bool_)
    losses = jnp.zeros([timesteps], dtype=jnp.float32)

    model_params = model.init(action_key, jnp.zeros((state_shape,)))
    target_net_params = model.init(action_key, jnp.zeros((state_shape,)))
    optimizer_state = optimizer.init(model_params)

    val_init = (
        model_params,
        target_net_params,
        optimizer_state,
        buffer_state,
        action_key,
        buffer_key,
        env_state,
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
        "action_key",
        "buffer_key",
        "env_state",
        "all_obs",
        "all_rewards",
        "all_done",
        "losses",
    ]
    for idx, value in enumerate(vals):
        output_dict[keys[idx]] = value

    return output_dict
