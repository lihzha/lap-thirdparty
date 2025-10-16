import jax
import jax.numpy as jnp
import pprint  # Import the pretty-print library
from gemma.gm.nn import _config, _modules
from gemma.multimodal import vision as gemma_vision

# Import your custom MoE model classes
from gemma3 import Gemma3MoEModel# Or your actual file name


_NUM_LAYERS_GEMMA_2B = 18
_NUM_LAYERS_GEMMA_7B = 28
_NUM_LAYERS_GEMMA2_2B = 26
_NUM_LAYERS_GEMMA2_9B = 42
_NUM_LAYERS_GEMMA2_27B = 46
_NUM_LAYERS_GEMMA3_270M = 18
_NUM_LAYERS_GEMMA3_1B = 26
_NUM_LAYERS_GEMMA3_4B = 34
_NUM_LAYERS_GEMMA3_12B = 48
_NUM_LAYERS_GEMMA3_27B = 62


GEMMA3_ATTENTION_PATTERN = (
    _modules.AttentionType.LOCAL_SLIDING,
    _modules.AttentionType.LOCAL_SLIDING,
    _modules.AttentionType.LOCAL_SLIDING,
    _modules.AttentionType.LOCAL_SLIDING,
    _modules.AttentionType.LOCAL_SLIDING,
    _modules.AttentionType.GLOBAL,
)


def main():
    # --- Example Usage ---

    # 1. Define the number of experts
    NUM_EXPERTS = 2

    # 2. Get the base configuration for a Gemma 3 model
    base_config = _config.TransformerConfig(
        final_logit_softcap=None,
      num_embed=262_144,
      embed_dim=2560,
      hidden_dim=2560 * 8 // 2,
      num_heads=8,
      head_dim=256,
      num_kv_heads=4,
      use_post_attn_norm=True,
      use_post_ffw_norm=True,
      use_qk_norm=True,
      attention_types=_config.make_attention_layers_types(
          GEMMA3_ATTENTION_PATTERN, num_layers=_NUM_LAYERS_GEMMA3_4B
      ),
      query_pre_attn_norm=_config.QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM,
      attn_logits_soft_cap=None,
      sliding_window_size=1024,
      transpose_gating_einsum=True,
      local_base_frequency=10_000,
      global_base_frequency=1_000_000,
      global_scale_factor=8.0,
      vision_encoder=gemma_vision.SigLiPFromPatches(),
    )

    print("--- Base Config (Nicely Printed) ---")
    pprint.pprint(base_config)
    print("-" * 40)
    print("\n") # Add some space

    # 3. Instantiate your MoE model
    moe_model = Gemma3MoEModel(config=base_config, num_experts=NUM_EXPERTS)

    print("--- MoE Model Definition (Uninitialized) ---")
    # Printing the uninitialized model just shows its constructor arguments
    print(moe_model)
    print("-" * 40)
    print("\n")

    # --- To print the model's structure, we must first initialize it ---
    # This requires a JAX random key and dummy input data that matches the
    # shape of the model's __call__ method.

    print("--- MoE Model Structure (Tabulated) ---")
    
    # a. Create a random key
    key = jax.random.PRNGKey(0)
    
    # b. Create dummy input tensors with realistic shapes (e.g., batch size 1)
    batch_size = 1
    prefix_len = 10
    suffix_len = 5
    full_len = prefix_len + suffix_len
    
    dummy_prefix_tok = jnp.zeros((batch_size, prefix_len), dtype=jnp.int32)
    dummy_suffix_tok = jnp.zeros((batch_size, suffix_len), dtype=jnp.int32)
    dummy_images = None  # For a text-only test run
    dummy_mask = jnp.ones((batch_size, full_len, full_len), dtype=jnp.bool_)
    dummy_positions = jnp.arange(full_len)[None, :]
    dummy_adarms_cond = None

    # c. Use the `tabulate` method to generate and print the model summary
    # This is the best way to visualize the model's architecture.
    tabulated_summary = moe_model.tabulate(
        key,
        prefix_tok=dummy_prefix_tok,
        suffix_tok=dummy_suffix_tok,
        images=dummy_images,
        mask=dummy_mask,
        positions=dummy_positions,
        adarms_cond=dummy_adarms_cond,
    )
    
    print(tabulated_summary)
    print("-" * 40)


if __name__ == "__main__":
    main()