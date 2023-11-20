import jax.numpy as jnp
from jax import jit, lax, random
from jax_tqdm import loop_tqdm


def tabular_double_rollout(
    key,
    TIME_STEPS,
    N_ACTIONS,
    GRID_SIZE,
    env,
    agent,
    policy,
):
    def _single_agent_double_rollout(key: random.PRNGKey, timesteps=TIME_STEPS):
        @loop_tqdm(TIME_STEPS)
        @jit
        def _fori_body(i: int, val: tuple):
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
        val = lax.fori_loop(0, timesteps, _fori_body, val_init)

        return val

    return _single_agent_double_rollout(key, TIME_STEPS)
