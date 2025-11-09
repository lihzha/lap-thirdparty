import dataclasses
from typing import TYPE_CHECKING

import flax.nnx as nnx
import jax
import jax.numpy as jnp
from openpi.models import model as _model
import openpi.models.gemma as _gemma
from openpi.shared import array_typing as at
import openpi.shared.nnx_utils as nnx_utils
from typing_extensions import override

from openpi_cot.models.adapters.model_adapter import CoTObservation
from openpi_cot.models.adapters.model_adapter import ExtendedModelType
import openpi_cot.models.gemma2 as _gemma2
import openpi_cot.models.gemma3 as _gemma3

if TYPE_CHECKING:
    from openpi_cot.models.pi_cot import PiCoT

from typing import Literal


@dataclasses.dataclass(frozen=True)
class PiCoTConfig(_model.BaseModelConfig):
    dtype: str = "bfloat16"
    paligemma_variant: _gemma.Variant | _gemma2.Variant | _gemma3.Variant = "gemma_2b"
    action_expert_variant: _gemma.Variant | _gemma2.Variant | _gemma3.Variant = "gemma_300m"

    # Set the model specific defaults.
    action_dim: int = 32
    action_horizon: int = 50
    max_token_len: int = None

    # if verbose_mode=True, log per sample metrics 
    verbose_mode: bool = False

    pi05: bool = False
    discrete_state_input: bool = None
    prompt_format: Literal[
        "pi05",
        "pi0",
        "vqa",
        "coordinate_system",
        "schema_compact",
        "schema_compact_with_rotation",
        "schema_compact_bimanual",
        "schema_compact_bimanual_with_rotation",
        "schema_compact_named_params",
        "verbose_state",
    ] = "pi05"

    aug_wrist_image: bool = True
    # Whether to use bimanual (3 cameras) or single-arm (2 cameras) setup
    # When False, only uses base_0_rgb and left_wrist_0_rgb to save memory
    use_bimanual: bool = False

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
    @override
    def model_type(self) -> ExtendedModelType:
        return ExtendedModelType.PI_COT

    @override
    def create(self, rng: at.KeyArrayLike) -> "PiCoT":
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
