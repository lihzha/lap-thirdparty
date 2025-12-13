import jax
import jax.numpy as jnp

from openpi_cot.models.backbones.gemma import Attention
from openpi_cot.models.backbones.gemma import Config


def _build_inputs():
    batch = 1
    width = 4
    t0, t1 = 60, 28
    x0 = jnp.reshape(jnp.arange(batch * t0 * width, dtype=jnp.float32) / 10.0, (batch, t0, width))
    x1 = jnp.reshape(jnp.arange(batch * t1 * width, dtype=jnp.float32) / 7.0 + 0.5, (batch, t1, width))
    positions = jnp.arange(t0 + t1)[None, :]
    mask = jnp.ones((batch, 1, t0 + t1, t0 + t1), dtype=bool)  # allow all attention, including cross-expert
    return x0, x1, positions, mask


def _grad_wrt_expert0_input(stop_cross: bool):
    x0, x1, positions, mask = _build_inputs()
    config0 = Config(
        width=4, depth=1, mlp_dim=8, num_heads=4, num_kv_heads=2, head_dim=2, stop_action_to_vlm_grad=stop_cross
    )
    config1 = Config(width=4, depth=1, mlp_dim=8, num_heads=4, num_kv_heads=2, head_dim=2)
    attn = Attention(configs=(config0, config1))
    params = attn.init(jax.random.PRNGKey(0), [x0, x1], positions, mask, kv_cache=None)["params"]

    def loss_fn(x0_input):
        out, _ = attn.apply({"params": params}, [x0_input, x1], positions, mask, None)
        return jnp.sum(out[1])

    return jax.grad(loss_fn)(x0)


def test_stop_grad_blocks_cross_to_expert0():
    grad_enabled = _grad_wrt_expert0_input(stop_cross=True)
    grad_disabled = _grad_wrt_expert0_input(stop_cross=False)

    assert jnp.allclose(grad_enabled, 0.0, atol=1e-6)
    assert jnp.linalg.norm(grad_disabled) > 1e-6
