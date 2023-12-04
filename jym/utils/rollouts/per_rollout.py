import haiku as hk
import jax.numpy as jnp

from ..replay_buffers import Experience


def _compute_td_error(
    model: hk.Transformed,
    online_net_params: dict,
    target_net_params: dict,
    discount: float,
    experience: Experience,
) -> float:
    state, action, reward, next_state, done = experience
    td_target = (
        (1 - done)
        * discount
        * jnp.max(model.apply(target_net_params, None, next_state))
    )
    prediction = model.apply(online_net_params, None, state)[action]
    return reward + td_target - prediction


def per_rollout():
    pass
