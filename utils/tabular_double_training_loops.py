import jax.numpy as jnp
from jax import jit, lax, random
from jax_tqdm import loop_tqdm


def double_rollout(
    key,
    TIME_STEPS,
    N_ACTIONS,
    GRID_SIZE,
    env,
    agent,
    policy,
):
    def single_agent_double_rollout(key: random.PRNGKey, timesteps=TIME_STEPS):
        @loop_tqdm(TIME_STEPS)
        @jit
        def fori_body(i: int, val: tuple):
            env_states, action_key, all_obs, all_rewards, all_done, all_q1, all_q2 = val
            states, _ = env_states
            q1 = all_q1[i]
            q2 = all_q2[i]

            # action selection, step and q-update
            actions, action_key = policy(action_key, N_ACTIONS, states, q1, q2)
            env_states, obs, rewards, done = env.step(env_states, actions)

            # selects whether to update Q1 or Q2
            to_update = random.normal(action_key) > 0.5
            q1, q2 = agent.update(
                states, actions, rewards, done, obs, q1, q2, to_update
            )
            # update observations, rewards, done flag and q_values
            all_obs = all_obs.at[i].set(obs)
            all_rewards = all_rewards.at[i].set(rewards)
            all_done = all_done.at[i].set(done)
            all_q1 = all_q1.at[i + 1].set(q1)
            all_q2 = all_q2.at[i + 1].set(q2)

            val = (
                env_states,
                action_key,
                all_obs,
                all_rewards,
                all_done,
                all_q1,
                all_q2,
            )
            return val

        # initialize obs, rewards, done and q_values with an added time index
        all_obs = jnp.zeros([timesteps, 2])
        all_rewards = jnp.zeros([timesteps], dtype=jnp.int32)
        all_done = jnp.zeros([timesteps], dtype=jnp.bool_)
        # q_values have first dimension = timesteps +1, as the update targets time step t+1
        all_q1 = jnp.zeros(
            [timesteps + 1, GRID_SIZE[0], GRID_SIZE[1], N_ACTIONS], dtype=jnp.float32
        )
        all_q2 = jnp.zeros(
            [timesteps + 1, GRID_SIZE[0], GRID_SIZE[1], N_ACTIONS], dtype=jnp.float32
        )

        # random keys used for policy / action selection
        action_key = random.PRNGKey(0)
        env_states, _ = env.reset(key)

        val_init = (
            env_states,
            action_key,
            all_obs,
            all_rewards,
            all_done,
            all_q1,
            all_q2,
        )
        val = lax.fori_loop(0, timesteps, fori_body, val_init)

        return val

    return single_agent_double_rollout(key, TIME_STEPS)


def parallel_double_rollout(
    keys, TIME_STEPS, N_ACTIONS, GRID_SIZE, N_ENV, v_policy, v_update, v_step, v_reset
):
    def parallel_agent_double_rollout(keys, timesteps, n_env):
        @loop_tqdm(TIME_STEPS)
        @jit
        def fori_body(i: int, val: tuple):
            (
                env_states,
                action_keys,
                all_obs,
                all_rewards,
                all_done,
                all_q1,
                all_q2,
            ) = val
            states, _ = env_states
            q1 = all_q1[i]
            q2 = all_q2[i]
            actions, action_keys = v_policy(action_keys, N_ACTIONS, states, q1, q2)

            env_states, obs, rewards, done = v_step(env_states, actions)
            q1, q2 = v_update(states, actions, rewards, done, obs, q1, q2)

            all_obs = all_obs.at[i].set(obs)
            all_rewards = all_rewards.at[i].set(rewards)
            all_done = all_done.at[i].set(done)
            all_q1 = all_q1.at[i + 1].set(q1)
            all_q2 = all_q2.at[i + 1].set(q2)

            val = (
                env_states,
                action_keys,
                all_obs,
                all_rewards,
                all_done,
                all_q1,
                all_q2,
            )
            return val

        all_obs = jnp.zeros([timesteps, n_env, 2])
        all_rewards = jnp.zeros([timesteps, n_env], dtype=jnp.int32)
        all_done = jnp.zeros([timesteps, n_env], dtype=jnp.bool_)
        # q_values has first dimension = timesteps +1, as the update targets time step t+1
        all_q1 = jnp.zeros(
            [timesteps + 1, GRID_SIZE[0], GRID_SIZE[1], N_ACTIONS, n_env],
            dtype=jnp.float32,
        )
        all_q2 = jnp.zeros(
            [timesteps + 1, GRID_SIZE[0], GRID_SIZE[1], N_ACTIONS, n_env],
            dtype=jnp.float32,
        )

        action_keys = random.split(random.PRNGKey(0), n_env)
        env_states, _ = v_reset(keys)

        val_init = (
            env_states,
            action_keys,
            all_obs,
            all_rewards,
            all_done,
            all_q1,
            all_q2,
        )
        val = lax.fori_loop(0, timesteps, fori_body, val_init)
        env_states, action_keys, all_obs, all_reward, all_done, all_q_values = val

        return all_obs, all_reward, all_done, all_q_values

    return parallel_agent_double_rollout(keys, TIME_STEPS, N_ENV)