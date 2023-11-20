from jax import jit, vmap


class SimpleBandit:
    @staticmethod
    @jit
    def __call__(action, reward, pulls, q_values):
        q_value = q_values[action]

        pull_update = pulls[action] + 1
        # incremental mean update
        # new_estimate = old_estimate + step_size(target - olde_estimate)
        q_update = q_value + (1 / pull_update) * (reward - q_value)
        return q_values.at[action].set(q_update), pulls.at[action].set(pull_update)

    @staticmethod
    @jit
    def batch_update(actions, rewards, pulls, q_values):
        return vmap(
            SimpleBandit.__call__,
            # in_axes =-1 => loop over the batch dimension
            in_axes=(0, 0, -1, -1),
            # out_axes =-1 => the batch dimensions appears last
            out_axes=-1,
        )(actions, rewards, pulls, q_values)

    @staticmethod
    @jit
    def multi_run_batch_update(actions, rewards, pulls, q_values):
        return vmap(
            SimpleBandit.batch_update,
            in_axes=(-1, -1, -1, -1),
            out_axes=(-1, -1),
        )(actions, rewards, pulls, q_values)
