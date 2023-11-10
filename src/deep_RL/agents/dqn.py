from functools import partial
from typing import Callable

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

    # def update(
    #     self,
    #     x: jnp.ndarray,
    #     y: jnp.ndarray,
    #     model_params: dict,
    #     target_net_params: dict,
    #     optimizer: optax.GradientTransformation,
    #     optimizer_state: jnp.ndarray,
    #     loss_fn: Callable,
    #     n_updates: int,
    #     target_updates: int,
    # ):
    #     @loop_tqdm(n_updates)
    #     @jit
    #     def _update_loop(i: int, val: tuple):
    #         x, y, model_params, target_net_params, optimizer_state, losses = val
    #         loss_val, grads = loss_grad_fn(model_params, x, y)
    #         updates, optimizer_state = optimizer(grads, optimizer_state)
    #         model_params = optax.apply_updates(model_params, updates)
    #         losses = losses.at[i].set(loss_val)

    #         # update the target parameters every `target_updates` steps
    #         target_net_params = lax.cond(
    #             i % target_updates == 0,
    #             lambda _: model_params,
    #             lambda _: target_net_params,
    #             operand=None,
    #         )
    #         return (x, y, model_params, target_net_params, optimizer_state, losses)

    #     losses = jnp.empty((n_updates), dtype=jnp.float32)

    #     loss_grad_fn = value_and_grad(loss_fn)
    #     val_init = (x, y, model_params, target_net_params, optimizer_state, losses)

    #     return lax.fori_loop(0, n_updates, _update_loop, val_init)

    @jit
    def update(
        self,
        model_params: dict,
        optimizer: optax.GradientTransformation,
        optimizer_state: jnp.ndarray,
        loss_fn: Callable,
    ):
        loss_grad_fn = grad(loss_fn)
        grads = loss_grad_fn(model_params)
        updates, optimizer_state = optimizer.update(grads, optimizer_state)
        model_params = optax.apply_updates(model_params, updates)

        return model_params, optimizer_state

    @partial(jit, static_argnums=(0))
    def act(self, key: random.PRNGKey, model_params: dict, state: jnp.ndarray):
        def _random_action(subkey):
            return random.choice(subkey, jnp.arange(state.shape[-1]))

        def _forward_pass(_):
            q_values = self.model.apply(model_params, state)
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

    @partial(jit, static_argnums=(0))
    def batch_loss_fn(
        self,
        model_params: dict,
        target_net_params: dict,
        states: jnp.ndarray,
        actions: jnp.ndarray,
        next_states: jnp.ndarray,
        dones: jnp.ndarray,
        rewards: jnp.ndarray,
    ):
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
            )
        )
