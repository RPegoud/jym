import jax.numpy as jnp
from jax import jit, random, vmap

from src.policies import EpsilonGreedy

SEED = 0
N_ENV = 10
N_ACTIONS = 4
policy = EpsilonGreedy(0.1)
q_values = jnp.zeros(4)


def test_unique_action_selection():
    dims = [8, 8, 4]
    states = (jnp.array([0, 0]), jnp.array([8, 8]), jnp.array([4, 4]))
    q_values = (jnp.zeros(dims), jnp.ones(dims), random.normal(random.PRNGKey(0), dims))
    expected = (3, 3, 3, 2, 0, 3, 3, 3, 3, 2, 0, 3, 0, 3, 3)
    for i in range(len(expected)):
        assert (
            policy(random.PRNGKey(i), N_ACTIONS, states[i % 3], q_values[i % 3])[0]
            == expected[i]
        )


def test_batch_action_selection():
    v_policy = jit(
        vmap(
            policy.call,
            in_axes=(0, None, 0, -1),  # (keys, n_actions, state, q_values)
            axis_name="batch_axis",
        ),
        static_argnums=(1,),
    )
    dims = [8, 8, 4, 10]

    keys = random.split(random.PRNGKey(0), 10)
    states = jnp.arange(20).reshape(-1, 2)
    expected = [
        [1, 3, 0, 0, 1, 1, 1, 1, 2, 1],
        [3, 3, 3, 3, 1, 3, 3, 1, 2, 3],
        [3, 3, 3, 1, 1, 3, 3, 1, 0, 1],
        [3, 1, 0, 2, 1, 0, 0, 1, 0, 1],
        [0, 0, 0, 3, 1, 1, 2, 1, 0, 3],
    ]

    for i in range(5):
        test_q = random.normal(random.PRNGKey(i), dims)
        assert jnp.all(
            jnp.equal(
                v_policy(keys, N_ACTIONS, states, test_q)[0], jnp.array(expected[i])
            )
        )
