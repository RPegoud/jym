from functools import partial

import haiku as hk
import jax.numpy as jnp
import optax
from jax import jit, lax, random, value_and_grad, vmap

from .base_agent import BaseDeepRLAgent


class DQN(BaseDeepRLAgent):
    def __init__(
        self,
        model: hk.Transformed,
        discount: float,
        n_actions: int,
    ) -> None:
        super(DQN, self).__init__(
            discount,
        )
        self.model = model
        self.n_actions = n_actions

    @partial(jit, static_argnums=(0))
    def act(
        self,
        key: random.PRNGKey,
        online_net_params: dict,
        state: jnp.ndarray,
        epsilon: float,
    ):
        """
        Epsilon-Greedy policy with respect to the estimated Q-values.
        """

        def _random_action(subkey):
            return random.choice(subkey, jnp.arange(self.n_actions))

        def _forward_pass(_):
            q_values = self.model.apply(online_net_params, None, state)
            return jnp.argmax(q_values)

        explore = random.uniform(key) < epsilon
        key, subkey = random.split(key)
        action = lax.cond(
            explore,
            _random_action,
            _forward_pass,
            operand=subkey,
        )
        return action, subkey

    @partial(jit, static_argnames=("self", "optimizer"))
    def update(
        self,
        online_net_params: dict,
        target_net_params: dict,
        optimizer: optax.GradientTransformation,
        optimizer_state: jnp.ndarray,
        experiences: dict[
            str : jnp.ndarray
        ],  # states, actions, next_states, dones, rewards
    ):
        @jit
        def _batch_loss_fn(
            online_net_params: dict,
            target_net_params: dict,
            states: jnp.ndarray,
            actions: jnp.ndarray,
            rewards: jnp.ndarray,
            next_states: jnp.ndarray,
            dones: jnp.ndarray,
        ):
            # vectorize the loss over states, actions, rewards, next_states and done flags
            @partial(vmap, in_axes=(None, None, 0, 0, 0, 0, 0))
            def _loss_fn(
                online_net_params,
                target_net_params,
                state,
                action,
                reward,
                next_state,
                done,
            ):
                target = reward + (1 - done) * self.discount * jnp.max(
                    self.model.apply(target_net_params, None, next_state),
                )
                prediction = self.model.apply(online_net_params, None, state)[action]
                return jnp.square(target - prediction)

            return jnp.mean(
                _loss_fn(
                    online_net_params,
                    target_net_params,
                    states,
                    actions,
                    rewards,
                    next_states,
                    dones,
                ),
                axis=0,
            )

        loss, grads = value_and_grad(_batch_loss_fn)(
            online_net_params, target_net_params, **experiences
        )
        updates, optimizer_state = optimizer.update(grads, optimizer_state)
        online_net_params = optax.apply_updates(online_net_params, updates)

        return online_net_params, optimizer_state, loss
