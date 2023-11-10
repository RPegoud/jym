from functools import partial

import haiku as hk
import jax.numpy as jnp
import optax
from jax import grad, jit, lax, random, vmap

from .base_agent import BaseDeepRLAgent


class DQN(BaseDeepRLAgent):
    def __init__(
        self,
        discount: float,
        learning_rate: float,
        model: hk.Transformed,
        epsilon: float,
    ) -> None:
        super(DQN, self).__init__(
            discount,
            learning_rate,
        )
        self.model = model
        self.epsilon = epsilon

    @partial(jit, static_argnums=(0))
    def act(self, key: random.PRNGKey, model_params: dict, state: jnp.ndarray):
        """
        Epsilon-Greedy policy with respect to the estimated Q-values.
        """

        def _random_action(subkey):
            return random.choice(subkey, jnp.arange(state.shape[-1]))

        def _forward_pass(_):
            q_values = self.model.apply(model_params, None, state)
            return jnp.argmax(q_values)

        explore = random.uniform(key) < self.epsilon
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
        model_params: dict,
        target_net_params: dict,
        optimizer: optax.GradientTransformation,
        optimizer_state: jnp.ndarray,
        experiences: dict[str : jnp.ndarray],
    ):
        @jit
        def batch_loss_fn(
            model_params: dict,
            target_net_params: dict,
            states: jnp.ndarray,
            actions: jnp.ndarray,
            next_states: jnp.ndarray,
            dones: jnp.ndarray,
            rewards: jnp.ndarray,
        ):
            # vectorize the loss over states, actions, next_states, done flags and rewards
            @partial(vmap, in_axes=(None, None, 0, 0, 0, 0, 0))
            def _loss_fn(
                model_params, target_net_params, state, action, next_state, done, reward
            ):
                target = lax.cond(
                    jnp.all(done is True),
                    lambda _: 0.0,
                    lambda _: self.discount
                    * jnp.max(
                        self.model.apply(target_net_params, None, next_state),
                    ),
                    operand=None,
                )
                prediction = self.model.apply(model_params, None, state)[action]
                return jnp.square(reward + target - prediction)

            return jnp.mean(
                _loss_fn(
                    model_params,
                    target_net_params,
                    states,
                    actions,
                    next_states,
                    dones,
                    rewards,
                ),
                axis=0,
            )

        grad_fn = grad(batch_loss_fn)
        grads = grad_fn(model_params, target_net_params, **experiences)
        updates, optimizer_state = optimizer.update(grads, optimizer_state)
        model_params = optax.apply_updates(model_params, updates)

        return model_params, optimizer_state
