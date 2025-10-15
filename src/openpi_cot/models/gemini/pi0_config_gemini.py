import dataclasses
from typing import TYPE_CHECKING  # This is the guard we will use

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from mediapy import _config
from typing_extensions import override

# Imports from the official Gemma repository for config creation
from gemma.gm.nn import _config as _gemma3_config
from gemma.gm.nn import _modules as gemma_modules
from gemma.multimodal import vision as gemma_vision

from openpi.models import model as _model
from openpi.shared import array_typing as at

# --- THE FIX IS HERE ---
# This block is only True for static type checkers (like in VSCode/PyCharm)
# but is False when you run the code. This breaks the import circle at runtime.
if TYPE_CHECKING:
    from openpi_cot.models.gemini.pi0_gemini import Pi0Gemma3

# Define Gemma 3 constants needed for the configuration.
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
    gemma_modules.AttentionType.LOCAL_SLIDING,
    gemma_modules.AttentionType.LOCAL_SLIDING,
    gemma_modules.AttentionType.LOCAL_SLIDING,
    gemma_modules.AttentionType.LOCAL_SLIDING,
    gemma_modules.AttentionType.LOCAL_SLIDING,
    gemma_modules.AttentionType.GLOBAL,
)


@dataclasses.dataclass(frozen=True)
class Pi0Config(_model.BaseModelConfig):
    """Configuration for the Pi0 model using a Gemma 3 backbone."""
    dtype: str = "bfloat16"
    model_variant: str = "gemma3_test"

    # Core architectural parameters
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = 256

    # If true, enables AdaRMSNorm and removes the explicit state token.
    pi05: bool = False
    
    # This is not used by the model but can be used by data loaders.
    @property
    def discrete_state_input(self) -> bool:
        return self.pi05

    @property
    @override
    def model_type(self) -> _model.ModelType:
        return _model.ModelType.PI05 if self.pi05 else _model.ModelType.PI0

    def get_gemma3_config(self) -> _gemma3_config.TransformerConfig:
        """Builds and returns the Gemma 3 TransformerConfig object."""

        
        """
        4B checks
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
        """
        if self.model_variant == "gemma3_4b":
            return _gemma3_config.TransformerConfig(
                num_embed=262_144, # Standard vocab size
                embed_dim=2560,
                hidden_dim=2560 * 8 // 2,
                num_heads=8,
                head_dim=256,
                num_kv_heads=4,
                use_post_attn_norm=True,
                use_post_ffw_norm=True,
                use_qk_norm=True,
                attention_types=_gemma3_config.make_attention_layers_types(
                    GEMMA3_ATTENTION_PATTERN, num_layers=_NUM_LAYERS_GEMMA3_4B
                ),
                query_pre_attn_norm=_gemma3_config.QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM,
                attn_logits_soft_cap=None,
                sliding_window_size=1024,
                transpose_gating_einsum=True,
                local_base_frequency=10_000.0,
                global_base_frequency=1_000_000.0,
                vision_encoder=gemma_vision.SigLiPFromPatches(),
                final_logit_softcap=None, # Not used for prediction
            )
        elif self.model_variant == "gemma3_1b":
            return _gemma3_config.TransformerConfig(
                final_logit_softcap=None,
                num_embed=262144,
                embed_dim=1152,
                hidden_dim=6 * 1152,
                num_heads=4,
                head_dim=256,
                num_kv_heads=1,
                use_post_attn_norm=True,
                use_post_ffw_norm=True,
                use_qk_norm=True,
                attention_types=_gemma3_config.make_attention_layers_types(
                    GEMMA3_ATTENTION_PATTERN, num_layers=_NUM_LAYERS_GEMMA3_1B
                ),
                query_pre_attn_norm=_gemma3_config.QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM,
                attn_logits_soft_cap=None,
                sliding_window_size=512,
                transpose_gating_einsum=True,
                local_base_frequency=10_000.0,
                global_base_frequency=1_000_000.0,
                vision_encoder=None,
            )
        
        elif self.model_variant == "gemma3_test":
            return _gemma3_config.TransformerConfig(
                num_embed=262_144, # Standard vocab size
                embed_dim=1024,
                hidden_dim=1024 * 8 // 2,
                num_heads=8,
                head_dim=128,
                num_kv_heads=4,
                use_post_attn_norm=True,
                use_post_ffw_norm=True,
                use_qk_norm=True,
                attention_types=_gemma3_config.make_attention_layers_types(
                    GEMMA3_ATTENTION_PATTERN, num_layers=_NUM_LAYERS_GEMMA3_4B
                ),
                query_pre_attn_norm=_gemma3_config.QueryPreAttentionNormalisation.BY_ONE_OVER_SQRT_HEAD_DIM,
                attn_logits_soft_cap=None,
                sliding_window_size=1024,
                transpose_gating_einsum=True,
                local_base_frequency=10_000.0,
                global_base_frequency=1_000_000.0,
                vision_encoder=gemma_vision.SigLiPFromPatches(),
                final_logit_softcap=None, # Not used for prediction
            )
        raise ValueError(f"Unknown model_variant: {self.model_variant}")

    @override
    def create(self, rng: at.KeyArrayLike) -> "Pi0Gemma3":
        """Factory method to create the Pi0 model instance."""
        from openpi_cot.models.gemini.pi0_gemini import Pi0Gemma3
        return Pi0Gemma3(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[_model.Observation, _model.Actions]:
        """Defines the shape and dtype of the model's raw inputs."""
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        # --- THIS IS THE FIX ---
        # Define a more realistic, multi-camera dummy observation,
        # matching the original OpenPI structure.
        with at.disable_typechecking():
            observation_spec = _model.Observation(
                images={
                    "base_0_rgb": image_spec,
                    "left_wrist_0_rgb": image_spec,
                    "right_wrist_0_rgb": image_spec,
                },
                image_masks={
                    "base_0_rgb": image_mask_spec,
                    "left_wrist_0_rgb": image_mask_spec,
                    "right_wrist_0_rgb": image_mask_spec,
                },
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """
        Returns a filter to freeze parts of the model during training.
        By default, nothing is frozen.
        """
        # Example: To freeze the vision encoder, you would inspect the model's
        # parameter structure and create a filter like:
        # return nnx.MultiFilter(
        #     nnx.PathContains("vision_encoder"),
        #     nnx.Not(nnx.PathContains("lora")) # Exclude LoRA params from freezing
        # )
        return nnx.Nothing