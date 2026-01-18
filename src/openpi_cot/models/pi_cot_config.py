import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from openpi.models import model as _model
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils
from typing_extensions import override

import openpi_cot.models.backbones.gemma as _gemma
import openpi_cot.models.backbones.gemma3 as _gemma3
from openpi_cot.models.model_adapter import CoTObservation
from openpi_cot.models.model_adapter import ExtendedModelType

if TYPE_CHECKING:
    from openpi_cot.models.pi_cot import PiCoT
    from openpi_cot.models.pi_cot_gemma3 import PiCoTGemma3


@dataclasses.dataclass(frozen=True)
class PiCoTConfig(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant | _gemma3.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant | _gemma3.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 7
    action_horizon: int = 16
    max_token_len: int = 220

    # if verbose_mode=True, log per sample metrics
    verbose_mode: bool = False

    pi05: bool = True
    discrete_state_input: bool = True
    prompt_format: str = "pi05_notime"
    prediction_format: str = "default"
    use_fast: bool = False

    aug_wrist_image: bool = True
    # When False, disables all image augmentation during training
    enable_image_augmentation: bool = True
    # Whether to use bimanual (3 cameras) or single-arm (2 cameras) setup
    # When False, only uses base_0_rgb and left_wrist_0_rgb to save memory
    use_bimanual: bool = False
    # Whether to use Pan & Scan for Gemma3 (multi-crop image processing)
    # When False, uses standard single-crop processing even with Gemma3
    use_pan_and_scan: bool = False
    # Override image resolution for Gemma3. If None, uses default (896, 896).
    # Set to (224, 224) to use smaller images with resized positional embeddings.
    # Note: When using non-default resolution, you must also set weight_loader.target_pos_emb_grid_size
    # to match (e.g., (16, 16) for 224x224 images with 14x14 patches).
    gemma3_image_resolution: tuple[int, int] | None = None

    # Enable/disable individual loss components
    # When True, enables training on raw actions (diffusion suffix) in addition to language tokens.
    enable_action_training: bool = False
    # When True, enables training on language (reasoning) tokens with cross-entropy.
    enable_langact_training: bool = True
    # When True, enables prediction loss (predicting movement between current and future frame).
    enable_prediction_training: bool = False
    # When True, enables VQA weighted loss.
    enable_vqa_training: bool = False
    # Scalar weights to combine losses when multiple are enabled
    language_loss_weight: float = 1.0
    action_loss_weight: float = 1.0
    prediction_loss_weight: float = 0.2
    vqa_loss_weight: float = 0.1
    # Per-dataset VQA loss weights. If provided, overrides vqa_loss_weight for specific datasets.
    # Format: dict[str, float] mapping dataset names to weights.
    # Unspecified datasets will use vqa_loss_weight as default.
    # Example: {"droid_bbox": 0.2, "coco_captions": 0.15}
    vqa_loss_weights: dict[str, float] | None = None

    state_dropout: float = 0.0
    reasoning_mask_prob: float = 0.0
    aggresive_aug: bool = False

    # When True, stops gradients produced by the action expert from flowing back
    # into the VLM expert through cross-attention.
    stop_action_to_vlm_grad: bool = False

    def __post_init__(self):
        if self.max_token_len is None:
            object.__setattr__(self, "max_token_len", 200 if self.pi05 else 48)
        if self.discrete_state_input is None:
            object.__setattr__(self, "discrete_state_input", self.pi05)

    @property
    def image_keys(self) -> tuple[str, ...]:
        """Returns the image keys to use based on bimanual setting."""
        if self.use_bimanual:
            return ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")
        return ("base_0_rgb", "left_wrist_0_rgb")

    @property
    def image_resolution(self) -> tuple[int, int]:
        """Returns image resolution based on model variant.
        
        For Gemma3, can be overridden via gemma3_image_resolution config.
        Supported resolutions for Gemma3 (with 14x14 patches):
        - (896, 896): 64x64 grid -> 4x4 pool -> 256 tokens (default)
        - (448, 448): 32x32 grid -> 2x2 pool -> 256 tokens
        - (224, 224): 16x16 grid -> no pool -> 256 tokens
        """
        # if "gemma3" in self.paligemma_variant:
        #     if self.gemma3_image_resolution is not None:
        #         return self.gemma3_image_resolution
        #     return (896, 896)
        return (224, 224)

    @property
    @override
    def model_type(self) -> ExtendedModelType:
        if self.use_fast:
            return ExtendedModelType.PI_FAST
        return ExtendedModelType.PI_COT

    @override
    def create(self, rng: at.KeyArrayLike) -> "PiCoT | PiCoTGemma3":
        """Create the appropriate model based on the variant."""
        if "gemma3" in self.paligemma_variant:
            from openpi_cot.models.pi_cot_gemma3 import PiCoTGemma3
            return PiCoTGemma3(self, rngs=nnx.Rngs(rng))
        else:
            from openpi_cot.models.pi_cot import PiCoT
            return PiCoT(self, rngs=nnx.Rngs(rng))

    @override
    def inputs_spec(self, *, batch_size: int = 1) -> tuple[CoTObservation, _model.Actions]:
        image_spec = jax.ShapeDtypeStruct([batch_size, *_model.IMAGE_RESOLUTION, 3], jnp.float32)
        image_mask_spec = jax.ShapeDtypeStruct([batch_size], jnp.bool_)

        with at.disable_typechecking():
            observation_spec = CoTObservation(
                images=dict.fromkeys(self.image_keys, image_spec),
                image_masks=dict.fromkeys(self.image_keys, image_mask_spec),
                state=jax.ShapeDtypeStruct([batch_size, self.action_dim], jnp.float32),
                tokenized_prompt=jax.ShapeDtypeStruct([batch_size, self.max_token_len], jnp.int32),
                tokenized_prompt_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                tokenized_langact_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
                critical_token_mask=jax.ShapeDtypeStruct([batch_size, self.max_token_len], bool),
            )
        action_spec = jax.ShapeDtypeStruct([batch_size, self.action_horizon, self.action_dim], jnp.float32)

        return observation_spec, action_spec

    def get_freeze_filter(self) -> nnx.filterlib.Filter:
        """Returns the freeze filter based on the model config."""
        filters = []
        has_lora = False
        gemma_params_filter = nnx_utils.PathRegex(".*llm.*")
        action_expert_params_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        if "lora" in self.paligemma_variant:
            filters.append(
                gemma_params_filter,
            )
            if "lora" not in self.action_expert_variant:
                # If only freeze gemma params, exclude action expert params.
                filters.append(
                    nnx.Not(action_expert_params_filter),
                )
            has_lora = True
        elif "lora" in self.action_expert_variant:
            filters.append(
                action_expert_params_filter,
            )
            has_lora = True

        if has_lora:
            # If any lora is used, exclude all lora params.
            filters.append(
                nnx.Not(nnx_utils.PathRegex(".*lora.*")),
            )
        # Always freeze the Gemma input embedding table (embedder/input_embedding).
        # input_embedding_filter = nnx_utils.PathRegex(".*input_embedding.*")

        if not filters:
            # If no other freeze rules, just freeze the input embedding.
            return nnx.Nothing

        # Union existing freeze rules with input embedding freeze.
        combined = nnx.All(*filters)
        # return nnx.Any(combined, input_embedding_filter)
        return combined

    def get_vlm_freeze_filter(self) -> nnx.filterlib.Filter:
        """Freeze the VLM (language model + image encoder), keep action expert trainable.

        This returns a filter that matches:
          - All params under `llm` EXCEPT the action-expert branch (identified by suffix `_1`)
          - All params under the image encoder `img`
        """
        # Match any parameter path containing "llm"
        llm_filter = nnx_utils.PathRegex(".*llm.*")
        # The action expert branch is identified with a trailing "_1" in its path per ModuleWithDecode
        action_expert_filter = nnx_utils.PathRegex(".*llm.*_1.*")
        # Image encoder lives under `img`
        img_filter = nnx_utils.PathRegex(".*img.*")

        # Freeze (llm minus action expert) OR (img)
        return nnx.Any(
            nnx.All(llm_filter, nnx.Not(action_expert_filter)),
            img_filter,
        )
