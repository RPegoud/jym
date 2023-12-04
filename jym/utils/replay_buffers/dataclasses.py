import jax.numpy as jnp
from chex import dataclass


@dataclass
class Experience:
    state: jnp.ndarray
    action: int
    reward: float
    next_state: jnp.ndarray
    done: bool
    priority: float = jnp.float32(0.0)
