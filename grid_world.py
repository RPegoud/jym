import jax
import jax.numpy as jnp
from jax import jit, lax, random

GRID_SIZE = jnp.array([8, 8], dtype=jnp.int32)
GOAL_POSITION = jnp.array([6, 1], dtype=jnp.int32)
INITIAL_POSITION = jnp.array([1, 6], dtype=jnp.int32)
OUT_OF_BOUNDS_RESET = True
NUM_ACTIONS = 4
RANDOM_SEED = 0
REWARD = -1
epsilon = 0.1
learning_rate = 0.25
discount_factor = 0.9
movements = jnp.array([(0, -1), (1, 0), (0, 1), (-1, 0)], dtype=jnp.int32)


def reset(initial_position=INITIAL_POSITION) -> jnp.array:
    return initial_position


def is_done(state: jnp.array, goal_position: jnp.array = GOAL_POSITION) -> bool:
    return jnp.all(state == goal_position)


# @jit
def clip_state(state, grid_size=GRID_SIZE):
    """if new positions are out of bound, clip the coordinates"""
    return jnp.clip(
        state,
        jnp.array([0, 0], dtype=jnp.int32),
        grid_size,
    )


# @jit
def random_policy(
    key: jax.random.PRNGKey,
) -> (jnp.array, jax.random.PRNGKey):
    """
    Picks an action according to the random uniform policy
    @key: a PRNGkey
    """
    key, subkey = jax.random.split(key)
    return (
        jax.random.randint(
            subkey,
            shape=(1,),
            minval=0,
            maxval=NUM_ACTIONS,
        ),
        key,
    )


@jit
def update_state(
    state: jnp.array, action: jnp.array, movements: jnp.array
) -> jnp.array:
    """
    Updates the state of the agent based on its current state and the
    selected action. In case of invalid action, 0 is selected by default
    @state: the (x, y) coordinates of the agent
    @action: integer between 0 and 3 corresponding to the direction of
        the agent (counterclockwise)
    """
    action_int = action[0]
    movement = jax.lax.dynamic_index_in_dim(movements, action_int, 0)[0]
    dx, dy = movement[0], movement[1]
    new_state = state + jnp.array([dx, dy], dtype=jnp.int32)

    return clip_state(new_state)


def epsilon_greedy(state, q_values, epsilon, key):
    """
    Args:
        q_values: jnp.array(n_actions) - the q-values for the current state
        epsilon: float - the probability of selecting a random action
        rng_key: random.PRNGKey
    """
    key, subkey = jax.random.split(key)
    state_q_values = jnp.array(q_values[state[0], state[1]])
    random_number = random.uniform(key, (1,))

    def exploration(rng_key):
        """Random Policy"""
        return random.randint(rng_key, (1,), 0, q_values.shape[-1])

    def exploitation(rng_key, max_indices):
        """Greedy action selection with random tie-break"""
        return max_indices[random.randint(rng_key, (1,), 0, max_indices.shape[-1])]

    max_indices = jnp.flatnonzero(state_q_values == jnp.max(state_q_values))
    rand_action = random_number < epsilon
    action = lax.cond(
        rand_action.item(),
        lambda k: exploration(k),
        lambda k: exploitation(k, max_indices),
        operand=key,
    )
    return action, key


def update_q_values(state, action, reward, next_state, q_values, discount_factor):
    @jit
    def update_fn():
        update = q_values[state[0], state[1], action]
        update += learning_rate * (
            reward
            + discount_factor * jnp.max(q_values[next_state[0], next_state[1]])
            - update
        )
        return q_values.at[state[0], state[1], action].set(update)

    def terminal_update_fn():
        return q_values.at[next_state[0], next_state[1]].set(0)

    if is_done(next_state):
        return terminal_update_fn()
    else:
        return update_fn()
