import jax.numpy as jnp
from jax import random
from tqdm.auto import tqdm

from agents import Q_learning
from envs import GridWorld
from policies import EpsilonGreedy

SEED = 2
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

q_values = jnp.zeros(
    [GRID_SIZE[0], GRID_SIZE[1], N_ACTIONS],
    dtype=jnp.float32,
)
movements = jnp.array([[0, 1], [1, 0], [0, -1], [-1, 0]])

env_state, obs = env.reset(key)
done = False
steps = []

for _ in tqdm(range(400)):
    done = False
    n_steps = 0
    while not done:
        state, _ = env_state
        action, key = policy(key, N_ACTIONS, q_values[tuple(state)])
        movement = movements[action]
        env_state, obs, reward, done = env.step(env_state, movement)

        q_values = agent.update(state, action, reward, done, obs, q_values)
        n_steps += 1
