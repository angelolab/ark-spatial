from jax import vmap, pmap, jit, grad
import jax.numpy as jnp
import equinox as eqx

@jit
@grad
def loss_fn(model, x, y):
    pred_y = vmap(model)(x)
    return jnp.mean((y - pred_y) ** 2)

batch_size, in_size, out_size = 32, 2, 3
model = eqx.nn.Linear(in_size, out_size, key=jnp.random.PRNGKey(0))
x = jnp.numpy.zeros((batch_size, in_size))
y = jnp.numpy.zeros((batch_size, out_size))
grads = loss_fn(model, x, y)


"""
XPU Accelerated FlowSOM.
"""
