class SimpleBandit:
    @staticmethod
    def __call__(action, reward, pulls, q_values):
        q_value = q_values[action]

        pull_update = pulls[action] + 1
        # incremental mean update
        # new_estimate = old_estimate + step_size(target - olde_estimate)
        q_update = q_value + (1 / pull_update) * (reward - q_value)
        return q_values.at[action].set(q_update), pulls.at[action].set(pull_update)
