from functools import partial

import haiku as hk
import jax.numpy as jnp
import optax
from jax import jit, lax, random, value_and_grad, vmap

from .base_agents import BaseDeepRLAgent


class DQN_PER(BaseDeepRLAgent):
    """
    Deep Q-Network using Prioritized Experience Replay.
    """

    def __init__(
        self,
        model: hk.Transformed,
        discount: float,
        n_actions: int,
    ) -> None:
        super(DQN_PER, self).__init__(
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
        importance_weights: jnp.ndarray,
        experiences: dict[
            str : jnp.ndarray
        ],  # states, actions, next_states, dones, rewards
    ):
        """
        DQN update with importance sampling.
        """

        @jit
        def _batch_loss_fn(
            online_net_params: dict,
            target_net_params: dict,
            state: jnp.ndarray,
            action: jnp.ndarray,
            reward: jnp.ndarray,
            next_state: jnp.ndarray,
            done: jnp.ndarray,
            priority: jnp.ndarray,
        ):
            # vectorize the loss over states, actions, rewards, next_states and done flags
            @partial(vmap, in_axes=(None, None, 0, 0, 0, 0, 0, 0))
            def _loss_fn(
                online_net_params,
                target_net_params,
                state,
                action,
                reward,
                next_state,
                done,
                priority,
            ):
                target = reward + (1 - done) * self.discount * jnp.max(
                    self.model.apply(target_net_params, None, next_state),
                )
                prediction = self.model.apply(online_net_params, None, state)[action]
                return jnp.square(target - prediction)

            loss = (
                _loss_fn(
                    online_net_params,
                    target_net_params,
                    state,
                    action,
                    reward,
                    next_state,
                    done,
                    priority,
                )
                * importance_weights
            )

            return jnp.mean(loss, axis=0)

        loss, grads = value_and_grad(_batch_loss_fn)(
            online_net_params, target_net_params, **experiences
        )
        updates, optimizer_state = optimizer.update(grads, optimizer_state)
        online_net_params = optax.apply_updates(online_net_params, updates)

        return online_net_params, optimizer_state, loss

    @partial(jit, static_argnums=(0))
    def batch_act(
        self,
        key: random.PRNGKey,
        online_net_params: dict,
        state: jnp.ndarray,
        epsilon: float,
    ):
        return vmap(
            DQN_PER.act,
            in_axes=(None, 0, 0, 0, 0),
        )(self, key, online_net_params, state, epsilon)

    @partial(jit, static_argnames=("self", "optimizer"))
    def batch_update(
        self,
        online_net_params: dict,
        target_net_params: dict,
        optimizer: optax.GradientTransformation,
        optimizer_state: jnp.ndarray,
        importance_weights: jnp.ndarray,
        experiences: dict[
            str : jnp.ndarray
        ],  # states, actions, next_states, dones, rewards
    ):
        return vmap(
            DQN_PER.update,
            in_axes=(0, 0, None, None, 0, 0),
        )(
            self,
            online_net_params,
            target_net_params,
            optimizer,
            optimizer_state,
            importance_weights,
            experiences,
        )
