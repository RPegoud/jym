import jax.numpy as jnp
from agents import Q_learning
from envs import GridWorld
from jax import jit, random, vmap
from policies import EpsilonGreedy

SEED = 2
N_ENV = 10
GRID_DIM = jnp.array([8, 8])
INITIAL_STATE = jnp.array([8, 8])
GOAL_STATE = jnp.array([0, 0])
GRID_SIZE = jnp.array([8, 8])
N_STATES = jnp.prod(GRID_DIM)
N_ACTIONS = 4
DISCOUNT = 0.9
LEARNING_RATE = 0.1

key = random.PRNGKey(SEED)

env = GridWorld(INITIAL_STATE, GOAL_STATE, GRID_SIZE)
policy = EpsilonGreedy(0.1)
agent = Q_learning(
    key,
    N_STATES,
    N_ACTIONS,
    DISCOUNT,
    LEARNING_RATE,
    policy,
)

movements = jnp.array([[0, 1], [1, 0], [0, -1], [-1, 0]])

v_reset = jit(
    vmap(env.reset, out_axes=((0, 0), 0), axis_name="batch_axis")  # ((env_state), obs)
)

v_step = jit(
    vmap(
        env.step,
        in_axes=((0, 0), 0),  # ((env_state), action)
        out_axes=((0, 0), 0, 0, 0),  # ((env_state), obs, reward, done)
        axis_name="batch_axis",
    )
)

v_policy = jit(
    vmap(
        policy.call,
        in_axes=(0, None, 0, -1),  # (keys, n_actions, state, q_values)
        axis_name="batch_axis",
    ),
    static_argnums=(1,),
)

v_update = jit(
    vmap(
        agent.update,
        in_axes=(0, 0, 0, 0, 0, -1),
        axis_name="batch_axis",
    ),
)


keys = random.split(key, N_ENV)
env_state, obs = v_reset(keys)
states, keys = env_state
q_values = jnp.zeros([GRID_SIZE[0], GRID_SIZE[1], N_ACTIONS, N_ENV], dtype=jnp.float32)

action, keys = v_policy(keys, N_ACTIONS, states, q_values)

env_state, obs, reward, done = v_step(env_state, action)
states, keys = env_state

q_values = v_update(states, action, reward, done, obs, q_values)

print(states.shape, action.shape, reward.shape)
