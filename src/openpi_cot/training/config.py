"""See _CONFIGS for the list of available configs."""

import abc
import dataclasses
import difflib
import logging
import pathlib
from typing import Literal, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
import openpi.models.model as _model
import openpi.training.config as upstream_config
import openpi.training.optimizer as _optimizer
import openpi.transforms as upstream_transforms
from typing_extensions import override
import tyro

from openpi_cot.datasets.utils.helpers import ActionEncoding
from openpi_cot.datasets.utils.helpers import NormalizationType
from openpi_cot.datasets.utils.helpers import StateEncoding
import openpi_cot.models.model_adapter as _model_adapter
import openpi_cot.models.pi_cot_config as pi_cot_config
from openpi_cot.models.tokenizer import FASTTokenizer
from openpi_cot.models.tokenizer import PaligemmaCoTTokenizer, Gemma3CoTTokenizer
import openpi_cot.policies.cot_policy as cot_policy
import openpi_cot.policies.libero_finetune_policy as libero_finetune_policy
import openpi_cot.policies.libero_policy as libero_policy
import openpi_cot.policies.planning_policy as planning_policy
import openpi_cot.policies.raw_policy as raw_action_policy
import openpi_cot.policies.vqa_policy as vqa_policy
import openpi_cot.shared.adapters.normalize_adapter as _normalize_adapter
from openpi_cot.shared.download import maybe_download
import openpi_cot.training.weight_loaders as weight_loaders
from openpi_cot.transforms import DetokenizeReasoning
from openpi_cot.transforms import ExtractFASTActions
from openpi_cot.transforms import TokenizeFASTCoTInputs
from openpi_cot.transforms import TokenizePromptAndReasoning

ModelType: TypeAlias = _model_adapter.ExtendedModelType
UpstreamModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class DeviceConfig:
    """Configuration for a specific device/environment.

    Centralizes device-specific paths and settings to avoid repetition.
    """

    name: str
    rlds_data_dir: str
    checkpoint_base_dir: str
    language_action_dir: str | None = None
    cache_dir: str | None = None
    fsdp_devices: int = 1
    default_batch_size: int = 256

    def get_language_action_dir(self) -> str:
        """Get language action directory, deriving from rlds_data_dir if not set."""
        if self.language_action_dir is not None:
            return self.language_action_dir
        return self.rlds_data_dir.replace("OXE", "droid-base-lang-actions")

    def get_cache_dir(self) -> str:
        """Get cache directory, deriving from checkpoint_base_dir if not set."""
        if self.cache_dir is not None:
            return self.cache_dir
        # Extract base bucket from checkpoint_base_dir
        base = (
            self.checkpoint_base_dir.rsplit("/", 1)[0] if "/" in self.checkpoint_base_dir else self.checkpoint_base_dir
        )
        return f"{base}/cache"


# Centralized device registry
DEVICE_CONFIGS = {
    "v4": DeviceConfig(
        name="v4",
        rlds_data_dir="gs://pi0-cot/OXE",
        checkpoint_base_dir="gs://pi0-cot/checkpoints",
        fsdp_devices=4,
        default_batch_size=256,
    ),
    "v5": DeviceConfig(
        name="v5",
        rlds_data_dir="gs://v5_central1_a/OXE",
        checkpoint_base_dir="gs://v5_central1_a/checkpoints",
        fsdp_devices=8,
        default_batch_size=256,
    ),
    "v5europe": DeviceConfig(
        name="v5",
        rlds_data_dir="gs://v5_europewest4/OXE",
        checkpoint_base_dir="gs://v5_europewest4/checkpoints",
        fsdp_devices=8,
        default_batch_size=256,
    ),
    "v6": DeviceConfig(
        name="v6",
        rlds_data_dir="gs://v6_east1d/OXE",
        checkpoint_base_dir="gs://v6_east1d/checkpoints",
        fsdp_devices=8,
        default_batch_size=256,
    ),
    "v6europe": DeviceConfig(
        name="v6europe",
        rlds_data_dir="gs://v6_europe_west4a/OXE",
        checkpoint_base_dir="gs://v6_europe_west4a/checkpoints",
        fsdp_devices=4,
        default_batch_size=256,
    ),
    "local": DeviceConfig(
        name="local",
        rlds_data_dir="/n/fs/robot-data/data/",
        checkpoint_base_dir="/n/fs/robot-data/pi0-cot/checkpoints",
        language_action_dir="/n/fs/robot-data/vlm-syn/droid-lang-actions",
        fsdp_devices=1,
        default_batch_size=4,
    ),
}


def _to_path(base: str | pathlib.Path, *extra: str) -> pathlib.Path | epath.Path:
    """
    Join `base` with any `extra` segments, returning:
      • `pathlib.Path` for normal file-system paths
      • `epath.Path`   for `gs://` URIs
    """
    base = str(base)  # in case the attr is already a Path object
    if base.startswith("gs://"):
        # epath.Path already mimics pathlib semantics (`/`, `.joinpath`, etc.)
        return epath.Path(base).joinpath(*extra)  # no `.resolve()` on GCS
    return (pathlib.Path(base).joinpath(*extra)).resolve()


def build_picot_model(
    *,
    action_horizon: int = 32,
    max_token_len: int = 110,
    pi05: bool = True,
    discrete_state_input: bool = True,
) -> _model.BaseModelConfig:
    """Convenience helper for common PiCoT model instantiations."""
    return pi_cot_config.PiCoTConfig(
        action_horizon=action_horizon,
        max_token_len=max_token_len,
        pi05=pi05,
        discrete_state_input=discrete_state_input,
    )


def build_cosine_lr(
    *,
    warmup_steps: int = 5_000,
    peak_lr: float = 1e-4,
    decay_steps: int = 40_000,
    decay_lr: float = 1e-4,
) -> _optimizer.LRScheduleConfig:
    """Shared cosine LR schedule used by most experiments."""
    return _optimizer.CosineDecaySchedule(
        warmup_steps=warmup_steps,
        peak_lr=peak_lr,
        decay_steps=decay_steps,
        decay_lr=decay_lr,
    )


@dataclasses.dataclass(frozen=True)
class CoTDataConfig(upstream_config.DataConfig):
    shuffle_buffer_size: int = 250_000
    # Optional cap on number of unique flattened samples for overfitting tests
    max_samples: int | None = None
    # Validation controls for RLDS-CoT dataset splitting/visualization
    val_max_samples: int | None = None
    val_fraction: float | None = 0.025
    use_wrist_image: bool = True
    wrist_image_dropout_prob: float = 0.0
    # One of {"droid", "oxe", "combined"}; used by the RLDS loader switch.
    dataset_type: Literal["droid", "oxe", "combined"] = "oxe"
    state_encoding: StateEncoding = StateEncoding.POS_EULER
    action_encoding: ActionEncoding = ActionEncoding.EEF_POS
    # Normalization type for actions and proprioceptive state.
    # CLI: --data.action_proprio_normalization_type {normal|bounds|bounds_q99}
    action_proprio_normalization_type: NormalizationType = NormalizationType.BOUNDS
    resize_resolution: tuple[int, int] = (224, 224)
    # Aggressive augmentation in data pipeline BEFORE padding (more effective cropping)
    # This mirrors preprocess_observation_aggressive but applies in TF data pipeline
    aggressive_aug: bool = False
    aug_wrist_image: bool = True

    # Language action format
    language_action_format_name: str = "verbose_eef_with_rotation"
    filter_all_1s_actions: bool = False
    stateless_gripper: bool = True
    filter_large_actions: bool = False
    random_base_prob: float = 0.0
    random_mask_prob: float = 0.2
    not_rotate_wrist_prob: float = 0.0
    use_rough_scale: bool = False
    horizon_seconds: list[float] = dataclasses.field(default_factory=lambda: [1.0])

    # Prediction training parameters
    max_prediction_horizon: int = 30
    pred_prob: float = 0.2  # Probability of converting a frame to prediction sample (after flattening)
    primary_pred_prob: float = 0.5  # Probability of using primary camera (vs wrist) for prediction training

    ### DROID fields (used when dataset_type == "droid")
    vis_dataset: bool = False
    language_action_dir: str | None = None
    # Optional path when DROID path is different from OXE path
    droid_rlds_data_dir: str | None = None
    # support using droid_subset for debugging
    droid_dataset_name: Literal["droid", "droid_100"] = "droid"
    force_recompute_stats: bool = False

    want_full_determinism: bool = False

    ### OXE fields (used when dataset_type == "oxe" or "combined")
    data_mix: str | None = "oxe_magic_soup"


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(upstream_config.ModelTransformFactory):
    """Creates model transforms for standard pi0 models."""

    prompt_format: str = "pi05"
    prediction_format: str = "default"
    include_outputs: bool = True  # Toggle output transforms (e.g., detokenization)
    fast_tokenizer_path: str = "physical-intelligence/fast"  # KarlP/fast_droid_specialist

    def __call__(self, model_config: _model.BaseModelConfig) -> upstream_transforms.Group:
        if model_config.model_type == ModelType.PI_COT:
            assert isinstance(model_config, pi_cot_config.PiCoTConfig)
            outputs = []
            tok_cls = PaligemmaCoTTokenizer if "gemma3" not in model_config.paligemma_variant else Gemma3CoTTokenizer
            if self.include_outputs:
                outputs = [
                    DetokenizeReasoning(
                        tok_cls(
                            model_config.max_token_len,
                            prompt_format=self.prompt_format,
                            prediction_format=self.prediction_format,
                            reasoning_mask_prob=0,
                        )
                    )
                ]
            return upstream_transforms.Group(
                inputs=[
                    upstream_transforms.InjectDefaultPrompt(self.default_prompt),
                    # upstream_transforms.ResizeImages(224, 224),
                    TokenizePromptAndReasoning(
                        tok_cls(
                            model_config.max_token_len,
                            prompt_format=self.prompt_format,
                            prediction_format=self.prediction_format,
                            reasoning_mask_prob=model_config.reasoning_mask_prob,
                        ),
                        discrete_state_input=model_config.discrete_state_input,
                        verbose_mode=model_config.verbose_mode,
                        state_dropout=model_config.state_dropout,
                    ),
                    upstream_transforms.PadStatesAndActions(model_config.action_dim),
                ],
                outputs=outputs,
            )
        if model_config.model_type in (ModelType.PI_FAST, UpstreamModelType.PI0_FAST):
            return upstream_transforms.Group(
                inputs=[
                    upstream_transforms.InjectDefaultPrompt(self.default_prompt),
                    # upstream_transforms.ResizeImages(224, 224),
                    TokenizeFASTCoTInputs(
                        FASTTokenizer(
                            fast_tokenizer_path=self.fast_tokenizer_path,
                            max_len=model_config.max_token_len,
                            prompt_format=self.prompt_format,
                            prediction_format=self.prediction_format,
                        ),
                        discrete_state_input=model_config.discrete_state_input,
                        state_dropout=model_config.state_dropout,
                    ),
                    # PadStates(model_config.action_dim),
                ],
                outputs=[
                    ExtractFASTActions(
                        FASTTokenizer(
                            fast_tokenizer_path=self.fast_tokenizer_path,
                            max_len=model_config.max_token_len,
                            prompt_format=self.prompt_format,
                            prediction_format=self.prediction_format,
                        ),
                        action_horizon=model_config.action_horizon,
                        action_dim=model_config.action_dim,
                    )
                ],
            )

        return super().__call__(model_config)


@dataclasses.dataclass(frozen=True)
class BaseCoTDataConfigFactory(CoTDataConfig, upstream_config.DataConfigFactory, abc.ABC):
    """Base class for all CoT data config factories.

    Provides common implementations for:
    - create_base_config: Extract CoT fields and set up base configuration
    - _load_norm_stats: Load normalization statistics from assets directory

    Subclasses must implement:
    - _create_data_transforms: Policy-specific data transformations
    - _create_model_transforms: Model-specific transformations
    """

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> CoTDataConfig:
        """Create base CoT config with common fields."""
        cot_fields = CoTDataConfig.__dataclass_fields__.keys()
        data = {k: getattr(self, k) for k in cot_fields}
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        data.update(
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=None,  # Note: Normalization is handled on dataset level
        )
        return CoTDataConfig(**data)

    def _load_norm_stats(
        self, assets_dir: epath.Path, asset_id: str | None
    ) -> dict[str, upstream_transforms.NormStats] | None:
        """Load normalization statistics from assets directory."""
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize_adapter.load(maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None

    @abc.abstractmethod
    def _create_data_transforms(
        self, base_cfg: CoTDataConfig, model_config: _model.BaseModelConfig
    ) -> upstream_transforms.Group:
        """Create policy-specific data transforms. Must be implemented by subclasses."""

    @abc.abstractmethod
    def _create_model_transforms(
        self, base_cfg: CoTDataConfig, model_config: _model.BaseModelConfig
    ) -> upstream_transforms.Group:
        """Create model-specific transforms. Must be implemented by subclasses."""

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> CoTDataConfig:
        """Template method that orchestrates config creation."""
        base_cfg = self.create_base_config(assets_dirs, model_config)
        data_transforms = self._create_data_transforms(base_cfg, model_config)
        model_transforms = self._create_model_transforms(base_cfg, model_config)

        return dataclasses.replace(
            base_cfg,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


@dataclasses.dataclass(frozen=True)
class RLDSCoTDataConfig(BaseCoTDataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    @override
    def _create_data_transforms(
        self, base_cfg: CoTDataConfig, model_config: _model.BaseModelConfig
    ) -> upstream_transforms.Group:
        return upstream_transforms.Group(
            inputs=[
                cot_policy.CoTInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                    wrist_image_dropout_prob=base_cfg.wrist_image_dropout_prob,
                    action_encoding=base_cfg.action_encoding,
                    language_action_format=base_cfg.language_action_format_name,
                    filter_all_1s_actions=base_cfg.filter_all_1s_actions,
                    random_mask_prob=base_cfg.random_mask_prob,
                    stateless_gripper=base_cfg.stateless_gripper,
                    random_base_prob=base_cfg.random_base_prob,
                    filter_large_actions=base_cfg.filter_large_actions,
                    use_rough_scale=base_cfg.use_rough_scale,
                    enable_langact_training=model_config.enable_langact_training,
                )
            ],
            outputs=[cot_policy.CoTOutputs(language_action_format=base_cfg.language_action_format_name)],
        )

    @override
    def _create_model_transforms(
        self, base_cfg: CoTDataConfig, model_config: _model.BaseModelConfig
    ) -> upstream_transforms.Group:
        return ModelTransformFactory(
            prompt_format=model_config.prompt_format,
            prediction_format=model_config.prediction_format,
        )(model_config)


@dataclasses.dataclass(frozen=True)
class RawActionDataConfig(BaseCoTDataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    @override
    def _create_data_transforms(
        self, base_cfg: CoTDataConfig, model_config: _model.BaseModelConfig
    ) -> upstream_transforms.Group:
        return upstream_transforms.Group(
            inputs=[
                raw_action_policy.RawActionInputs(
                    model_type=model_config.model_type,
                )
            ],
            outputs=[raw_action_policy.RawActionOutputs()],
        )

    @override
    def _create_model_transforms(
        self, base_cfg: CoTDataConfig, model_config: _model.BaseModelConfig
    ) -> upstream_transforms.Group:
        return ModelTransformFactory(
            prompt_format=model_config.prompt_format,
            prediction_format=model_config.prediction_format,
        )(model_config)


@dataclasses.dataclass(frozen=True)
class VQADataConfig(BaseCoTDataConfigFactory):
    """
    Config for VQA evaluation.
    """

    @override
    def _create_data_transforms(
        self, base_cfg: CoTDataConfig, model_config: _model.BaseModelConfig
    ) -> upstream_transforms.Group:
        return upstream_transforms.Group(
            inputs=[vqa_policy.VQAInputs(model_type=model_config.model_type)],
            outputs=[vqa_policy.VQAOutputs()],
        )

    @override
    def _create_model_transforms(
        self, base_cfg: CoTDataConfig, model_config: _model.BaseModelConfig
    ) -> upstream_transforms.Group:
        return ModelTransformFactory(
            prompt_format=model_config.prompt_format,
            prediction_format=model_config.prediction_format,
        )(model_config)


@dataclasses.dataclass(frozen=True)
class LiberoFinetuneDataConfig(BaseCoTDataConfigFactory):
    """
    Config for fine-tuning on Libero dataset.
    """

    @override
    def _create_data_transforms(
        self, base_cfg: CoTDataConfig, model_config: _model.BaseModelConfig
    ) -> upstream_transforms.Group:
        return upstream_transforms.Group(
            inputs=[
                libero_finetune_policy.LiberoFinetuneInputs(
                    model_type=model_config.model_type,
                    action_dim=model_config.action_dim,
                )
            ],
            outputs=[libero_finetune_policy.LiberoFinetuneOutputs()],
        )

    @override
    def _create_model_transforms(
        self, base_cfg: CoTDataConfig, model_config: _model.BaseModelConfig
    ) -> upstream_transforms.Group:
        return ModelTransformFactory(
            prompt_format=model_config.prompt_format,
            prediction_format=model_config.prediction_format,
            include_outputs=False,  # Libero doesn't need output detokenization
        )(model_config)


@dataclasses.dataclass(frozen=True)
class LiberoCoTDataConfig(BaseCoTDataConfigFactory):
    """
    Config for fine-tuning on Libero dataset.
    """

    @override
    def _create_data_transforms(
        self, base_cfg: CoTDataConfig, model_config: _model.BaseModelConfig
    ) -> upstream_transforms.Group:
        return upstream_transforms.Group(
            inputs=[
                libero_policy.LiberoInputs(
                    model_type=model_config.model_type,
                    action_dim=model_config.action_dim,
                )
            ],
            outputs=[libero_policy.LiberoOutputs(language_action_format_name=base_cfg.language_action_format_name)],
        )

    @override
    def _create_model_transforms(
        self, base_cfg: CoTDataConfig, model_config: _model.BaseModelConfig
    ) -> upstream_transforms.Group:
        return ModelTransformFactory(
            prompt_format=model_config.prompt_format,
            prediction_format=model_config.prediction_format,
            include_outputs=True,  # Libero doesn't need output detokenization
        )(model_config)


@dataclasses.dataclass(frozen=True)
class PlanningDataConfig(BaseCoTDataConfigFactory):
    """
    Config for training on planning dataset, using RLDS format loaded from TFDS.
    """

    @override
    def _create_data_transforms(
        self, base_cfg: CoTDataConfig, model_config: _model.BaseModelConfig
    ) -> upstream_transforms.Group:
        return upstream_transforms.Group(
            inputs=[
                planning_policy.PlanningInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                )
            ],
            outputs=[planning_policy.PlanningOutputs()],
        )

    @override
    def _create_model_transforms(
        self, base_cfg: CoTDataConfig, model_config: _model.BaseModelConfig
    ) -> upstream_transforms.Group:
        return ModelTransformFactory(
            prompt_format=model_config.prompt_format,
            prediction_format=model_config.prediction_format,
            include_outputs=False,  # Planning doesn't need output detokenization
        )(model_config)


@dataclasses.dataclass(frozen=True)
class TrainingStage:
    """Defines training configuration for a specific step range.

    Each stage can independently control:
    - Which losses to compute (langact, action, prediction)
    - Weight for each loss component
    - Probability of computing each loss (for stochastic curriculum learning)
    """

    start_step: int
    end_step: int | None = None  # None means until training ends

    # Loss enables
    enable_langact_training: bool = True
    enable_action_training: bool = False
    enable_prediction_training: bool = False

    # Loss weights
    language_loss_weight: float = 1.0
    action_loss_weight: float = 1.0
    prediction_loss_weight: float = 1.0

    # Stochastic loss probabilities (1.0 = always compute, 0.0 = never compute)
    langact_prob: float = 1.0
    action_prob: float = 1.0
    prediction_prob: float = 1.0

    def validate(self):
        """Validate stage configuration."""
        if self.start_step < 0:
            raise ValueError(f"start_step must be >= 0, got {self.start_step}")
        if self.end_step is not None and self.end_step <= self.start_step:
            raise ValueError(f"end_step ({self.end_step}) must be > start_step ({self.start_step})")

        # Validate probabilities are in [0, 1]
        for prob_name in ["langact_prob", "action_prob", "prediction_prob"]:
            prob = getattr(self, prob_name)
            if not 0.0 <= prob <= 1.0:
                raise ValueError(f"{prob_name} must be in [0.0, 1.0], got {prob}")

        # Validate weights are non-negative
        for weight_name in ["language_loss_weight", "action_loss_weight", "prediction_loss_weight"]:
            weight = getattr(self, weight_name)
            if weight < 0:
                raise ValueError(f"{weight_name} must be >= 0, got {weight}")


@dataclasses.dataclass(frozen=True)
class TrainingSchedule:
    """Manages multiple training stages with different loss configurations.

    Enables flexible curriculum learning with:
    - Stage-based loss configuration
    - Smooth or abrupt transitions between stages
    - Stochastic loss masking
    """

    stages: tuple[TrainingStage, ...]

    def __post_init__(self):
        """Validate the training schedule on initialization."""
        if not self.stages:
            raise ValueError("TrainingSchedule must have at least one stage")

        # Validate each stage
        for stage in self.stages:
            stage.validate()

        # Validate stages are ordered and non-overlapping
        for i in range(len(self.stages) - 1):
            current_stage = self.stages[i]
            next_stage = self.stages[i + 1]

            if current_stage.end_step is None:
                raise ValueError(
                    f"Stage {i} (starting at {current_stage.start_step}) has end_step=None but is not the last stage"
                )

            if next_stage.start_step < current_stage.end_step:
                raise ValueError(
                    f"Stage {i + 1} (starting at {next_stage.start_step}) overlaps with "
                    f"stage {i} (ending at {current_stage.end_step})"
                )

    def get_stage_for_step(self, step: int) -> TrainingStage:
        """Returns the appropriate stage configuration for the given training step.

        Args:
            step: Current training step

        Returns:
            TrainingStage configuration for this step

        Raises:
            ValueError: If no stage covers this step
        """
        for stage in self.stages:
            if stage.start_step <= step:
                if stage.end_step is None or step < stage.end_step:
                    return stage

        # If we reach here, no stage covers this step
        raise ValueError(
            f"No training stage covers step {step}. "
            f"Available stages: {[(s.start_step, s.end_step) for s in self.stages]}"
        )

    def get_stage_config_for_step(self, step):
        """JAX-compatible method to get stage configuration for a given step.

        This method returns stage parameters as a dictionary that can be used
        inside JIT-compiled functions without concretization errors.

        Args:
            step: Current training step (can be a JAX traced value)

        Returns:
            Dictionary with stage configuration parameters
        """
        import jax.numpy as jnp

        # Initialize with first stage as default
        result = {
            "enable_langact_training": self.stages[0].enable_langact_training,
            "enable_action_training": self.stages[0].enable_action_training,
            "enable_prediction_training": self.stages[0].enable_prediction_training,
            "language_loss_weight": self.stages[0].language_loss_weight,
            "action_loss_weight": self.stages[0].action_loss_weight,
            "prediction_loss_weight": self.stages[0].prediction_loss_weight,
            "langact_prob": self.stages[0].langact_prob,
            "action_prob": self.stages[0].action_prob,
            "prediction_prob": self.stages[0].prediction_prob,
        }

        # Iterate through stages and use jnp.where to select appropriate values
        # This avoids Python control flow that would fail with traced values
        for stage in self.stages:
            in_range = step >= stage.start_step
            if stage.end_step is not None:
                in_range = in_range & (step < stage.end_step)
            else:
                in_range = in_range & (step >= stage.start_step)

            # Use jnp.where to conditionally select values from this stage
            result["enable_langact_training"] = jnp.where(
                in_range, stage.enable_langact_training, result["enable_langact_training"]
            )
            result["enable_action_training"] = jnp.where(
                in_range, stage.enable_action_training, result["enable_action_training"]
            )
            result["enable_prediction_training"] = jnp.where(
                in_range, stage.enable_prediction_training, result["enable_prediction_training"]
            )
            result["language_loss_weight"] = jnp.where(
                in_range, stage.language_loss_weight, result["language_loss_weight"]
            )
            result["action_loss_weight"] = jnp.where(in_range, stage.action_loss_weight, result["action_loss_weight"])
            result["prediction_loss_weight"] = jnp.where(
                in_range, stage.prediction_loss_weight, result["prediction_loss_weight"]
            )
            result["langact_prob"] = jnp.where(in_range, stage.langact_prob, result["langact_prob"])
            result["action_prob"] = jnp.where(in_range, stage.action_prob, result["action_prob"])
            result["prediction_prob"] = jnp.where(in_range, stage.prediction_prob, result["prediction_prob"])

        return result

    def validate_for_training(self, num_train_steps: int):
        """Validate that the schedule covers all training steps.

        Args:
            num_train_steps: Total number of training steps
        """
        # Check that step 0 is covered
        try:
            self.get_stage_for_step(0)
        except ValueError:
            raise ValueError("Training schedule must cover step 0")

        # Check that the last step is covered
        try:
            self.get_stage_for_step(num_train_steps - 1)
        except ValueError:
            logging.warning(
                f"Training schedule does not cover final step {num_train_steps - 1}. "
                f"Last stage ends at {self.stages[-1].end_step}"
            )


@dataclasses.dataclass(frozen=True)
class TrainingScheduleChoice:
    """Choice of pre-specified training schedule configuration.

    Similar to WeightLoaderChoice, this provides common training schedules
    that can be selected via command line or programmatically.

    Available schedules:
    - lang_act_only: Only language action training
    - lang_act_and_pred: Language action + prediction training (weight=0.2, prob=0.2)
    - raw_act_only: Only raw action training

    Example:
        # From command line:
        python -m openpi_cot.training.train --config pi_droid_cot_v4 \\
            --training_schedule_choice.kind lang_act_and_pred

        # From Python:
        schedule = TrainingScheduleChoice(kind="lang_act_and_pred").build()
    """

    kind: Literal["lang_act_only", "lang_act_and_pred", "raw_act_only", None] = None

    # Optional overrides for schedule parameters
    langact_weight: float = 1.0
    prediction_weight: float = 0.2
    prediction_prob: float = 0.2
    action_weight: float = 1.0

    def build(self) -> TrainingSchedule:
        """Build the training schedule based on the selected kind."""
        if self.kind == "lang_act_only":
            return TrainingSchedule(
                stages=(
                    TrainingStage(
                        start_step=0,
                        end_step=None,
                        enable_langact_training=True,
                        enable_action_training=False,
                        enable_prediction_training=False,
                        language_loss_weight=self.langact_weight,
                        action_loss_weight=1.0,
                        prediction_loss_weight=1.0,
                        langact_prob=1.0,
                        action_prob=1.0,
                        prediction_prob=1.0,
                    ),
                )
            )
        if self.kind == "lang_act_and_pred":
            return TrainingSchedule(
                stages=(
                    TrainingStage(
                        start_step=0,
                        end_step=None,
                        enable_langact_training=True,
                        enable_action_training=False,
                        enable_prediction_training=True,
                        language_loss_weight=self.langact_weight,
                        action_loss_weight=1.0,
                        prediction_loss_weight=self.prediction_weight,
                        langact_prob=1.0,
                        action_prob=1.0,
                        prediction_prob=self.prediction_prob,
                    ),
                )
            )
        if self.kind == "raw_act_only":
            return TrainingSchedule(
                stages=(
                    TrainingStage(
                        start_step=0,
                        end_step=None,
                        enable_langact_training=False,
                        enable_action_training=True,
                        enable_prediction_training=False,
                        language_loss_weight=1.0,
                        action_loss_weight=self.action_weight,
                        prediction_loss_weight=1.0,
                        langact_prob=1.0,
                        action_prob=1.0,
                        prediction_prob=1.0,
                    ),
                )
            )
        if self.kind is None:
            return None


@dataclasses.dataclass(frozen=True)
class EmaStage:
    """Defines EMA decay for a specific step range."""

    start_step: int
    end_step: int | None = None  # None means until training ends
    decay: float | None = None  # None disables EMA updates in this range

    def validate(self):
        """Validate stage configuration."""
        if self.start_step < 0:
            raise ValueError(f"start_step must be >= 0, got {self.start_step}")
        if self.end_step is not None and self.end_step <= self.start_step:
            raise ValueError(f"end_step ({self.end_step}) must be > start_step ({self.start_step})")
        if self.decay is not None and not 0.0 < self.decay < 1.0:
            raise ValueError(f"decay must be in (0.0, 1.0), got {self.decay}")


@dataclasses.dataclass(frozen=True)
class EmaSchedule:
    """Manages EMA decay across multiple step ranges."""

    stages: tuple[EmaStage, ...]

    def __post_init__(self):
        if not self.stages:
            raise ValueError("EmaSchedule must have at least one stage")

        for stage in self.stages:
            stage.validate()

        for i in range(len(self.stages) - 1):
            current_stage = self.stages[i]
            next_stage = self.stages[i + 1]

            if current_stage.end_step is None:
                raise ValueError(
                    f"Stage {i} (starting at {current_stage.start_step}) has end_step=None but is not the last stage"
                )

            if next_stage.start_step < current_stage.end_step:
                raise ValueError(
                    f"Stage {i + 1} (starting at {next_stage.start_step}) overlaps with "
                    f"stage {i} (ending at {current_stage.end_step})"
                )

    def get_stage_for_step(self, step: int) -> EmaStage:
        for stage in self.stages:
            if stage.start_step <= step:
                if stage.end_step is None or step < stage.end_step:
                    return stage
        raise ValueError(
            f"No EMA stage covers step {step}. Available stages: {[(s.start_step, s.end_step) for s in self.stages]}"
        )

    def get_decay_for_step(self, step):
        """JAX-compatible method to get EMA decay and enable flag for a given step."""
        import jax.numpy as jnp

        decay = jnp.asarray(0.0, dtype=jnp.float32)
        enabled = jnp.asarray(False)

        for stage in self.stages:
            in_range = step >= stage.start_step
            if stage.end_step is not None:
                in_range = in_range & (step < stage.end_step)
            else:
                in_range = in_range & (step >= stage.start_step)

            stage_decay = 0.0 if stage.decay is None else stage.decay
            stage_enabled = stage.decay is not None
            decay = jnp.where(in_range, stage_decay, decay)
            enabled = jnp.where(in_range, stage_enabled, enabled)

        return decay, enabled

    def has_ema(self) -> bool:
        return any(stage.decay is not None for stage in self.stages)

    def default_decay(self) -> float | None:
        for stage in self.stages:
            if stage.decay is not None:
                return stage.decay
        return None


@dataclasses.dataclass(frozen=True)
class EmaScheduleChoice:
    """Choice of pre-specified EMA schedules.

    Available schedules:
    - disabled: EMA off
    - constant: EMA on from step 0 with fixed decay
    - delayed: EMA off until start_step, then fixed decay
    - cosine_delayed: EMA off until start_step, then cosine ramp to max decay

    Example:
        # From command line:
        python -m openpi_cot.training.train --config pi_droid_cot_v4 \\
            --ema_schedule_choice.kind delayed \\
            --ema_schedule_choice.start_step 10000 \\
            --ema_schedule_choice.decay 0.999
    """

    kind: Literal["disabled", "constant", "delayed", "cosine_delayed"] = "delayed"

    start_step: int = 10000

    def build(self, *, decay: float | None) -> EmaSchedule | None:
        if self.kind == "disabled":
            return None

        if self.kind == "constant":
            if decay is None:
                return None
            return EmaSchedule(stages=(EmaStage(start_step=0, end_step=None, decay=decay),))

        if self.kind == "delayed":
            if decay is None:
                return None
            if self.start_step <= 0:
                return EmaSchedule(stages=(EmaStage(start_step=0, end_step=None, decay=decay),))
            return EmaSchedule(
                stages=(
                    EmaStage(start_step=0, end_step=self.start_step, decay=None),
                    EmaStage(start_step=self.start_step, end_step=None, decay=decay),
                )
            )

        if self.kind == "cosine_delayed":
            return None

        raise ValueError(f"Unsupported EMA schedule kind: {self.kind}")


def create_multi_device_configs(
    base_name: str,
    devices: list[str],
    model: _model.BaseModelConfig,
    data_config_class: type[upstream_config.DataConfigFactory],
    data_config_kwargs: dict,
    **train_config_kwargs,
) -> list["TrainConfig"]:
    """Create multiple TrainConfig instances for different devices.

    This replaces the ConfigBuilder pattern with a simpler functional approach
    that directly creates configs with device-specific settings.

    Args:
        base_name: Base name for configs (device suffix will be added)
        devices: List of device names from DEVICE_CONFIGS
        model: Model configuration
        data_config_class: Data config factory class
        data_config_kwargs: Arguments for data config (device paths auto-filled)
        **train_config_kwargs: Additional TrainConfig arguments

    Returns:
        List of TrainConfig instances, one per device

    Example:
        configs = create_multi_device_configs(
            base_name="pi_droid_cot",
            devices=["v4", "v5", "v6", "local"],
            model=build_picot_model(),
            data_config_class=RLDSCoTDataConfig,
            data_config_kwargs={"repo_id": "droid", "dataset_type": "droid"},
            weight_loader=WeightLoaderChoice(kind="paligemma"),
        )
    """
    configs = []
    for device in devices:
        if device not in DEVICE_CONFIGS:
            raise ValueError(f"Unknown device '{device}'. Available: {list(DEVICE_CONFIGS.keys())}")

        device_cfg = DEVICE_CONFIGS[device]
        config_name = f"{base_name}_{device}"

        # Build data config kwargs with device-specific paths
        data_kwargs = {**data_config_kwargs}
        if "rlds_data_dir" not in data_kwargs:
            data_kwargs["rlds_data_dir"] = device_cfg.rlds_data_dir
        if "language_action_dir" not in data_kwargs:
            data_kwargs["language_action_dir"] = device_cfg.get_language_action_dir()

        # Build train config with device-specific settings
        train_kwargs = {
            "name": config_name,
            "model": model,
            "data": data_config_class(**data_kwargs),
            "fsdp_devices": device_cfg.fsdp_devices,
            "checkpoint_base_dir": device_cfg.checkpoint_base_dir,
            **train_config_kwargs,
        }

        # Handle device-specific weight loader params_path
        if "weight_loader" in train_kwargs:
            weight_loader = train_kwargs["weight_loader"]
            # Check if weight_loader has params_path that needs to be device-specific
            if hasattr(weight_loader, "params_path") and weight_loader.params_path:
                params_path = weight_loader.params_path
                # Extract the relative path after /cache/
                if "/cache/" in params_path:
                    cache_suffix = params_path.split("/cache/", 1)[1]
                    device_cache_dir = device_cfg.get_cache_dir()
                    device_params_path = f"{device_cache_dir}/{cache_suffix}"
                    # Create new weight loader with device-specific params_path
                    train_kwargs["weight_loader"] = dataclasses.replace(weight_loader, params_path=device_params_path)

        # Set batch_size from device default if not specified
        if "batch_size" not in train_config_kwargs:
            train_kwargs["batch_size"] = device_cfg.default_batch_size

        configs.append(TrainConfig(**train_kwargs))

    return configs


@dataclasses.dataclass(frozen=True)
class TrainConfig(upstream_config.TrainConfig):
    # Overide
    project_name: str = "openpi-cot"
    weight_loader: weight_loaders.WeightLoaderChoice = dataclasses.field(
        default_factory=weight_loaders.WeightLoaderChoice
    )
    model: _model.BaseModelConfig = dataclasses.field(default_factory=build_picot_model)
    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=build_cosine_lr)
    num_train_steps: int = 100_000
    save_interval: int = 2500
    log_interval: int = 50
    keep_period: int | None = 10000
    resume: bool = True
    ema_decay: float | None = 0.999
    ema_schedule_choice: EmaScheduleChoice = dataclasses.field(default_factory=EmaScheduleChoice)
    # New field
    use_validation: bool = True
    val_interval: int = 2000
    checkpoint_async_timeout_secs: int | None = 7200
    checkpoint_async_enable: bool = True
    checkpoint_max_retries: int = 1
    checkpoint_retry_delay_secs: float = 30.0
    checkpoint_retry_backoff: float = 2.0
    checkpoint_fallback_to_sync: bool = True
    allow_partial_weights: bool = True
    # Evaluation fields
    use_eval: bool = True
    eval_checkpoint_step: int | None = None
    num_eval_batches: int | None = 50
    eval_mode: Literal["token_accuracy", "rollout", "both", "token_visualization", "train_loss"] = "rollout"
    eval_use_ema: bool = False
    eval_split: Literal["val", "train"] = "val"
    # Multi-stage training schedule choice
    training_schedule_choice: TrainingScheduleChoice = dataclasses.field(default_factory=TrainingScheduleChoice)

    @property
    def training_schedule(self) -> TrainingSchedule:
        """Build training schedule from the choice configuration."""
        return self.training_schedule_choice.build()

    @property
    def ema_schedule(self) -> EmaSchedule | None:
        """Build EMA schedule from the choice configuration."""
        return self.ema_schedule_choice.build(decay=self.ema_decay)

    def get_ema_init(self) -> tuple[float | None, bool]:
        """Return the initial EMA decay and whether EMA params should be initialized."""
        if self.ema_schedule_choice.kind == "cosine_delayed":
            if self.ema_decay is None:
                return None, False
            return 0.0, True
        schedule = self.ema_schedule
        if schedule is None:
            return self.ema_decay, self.ema_decay is not None
        stage0 = schedule.get_stage_for_step(0)
        return stage0.decay, schedule.has_ema()

    def get_ema_decay_for_step(self, step):
        """Return EMA decay and enabled flag for a given step (JAX-compatible)."""
        if self.ema_schedule_choice.kind == "cosine_delayed":
            import jax.numpy as jnp

            max_decay = self.ema_decay
            if max_decay is None:
                return jnp.asarray(0.0, dtype=jnp.float32), jnp.asarray(False)
            start_step = self.ema_schedule_choice.start_step
            max_step = self.num_train_steps
            duration = jnp.maximum(max_step - start_step, 1)
            progress = (step - start_step) / duration
            progress = jnp.clip(progress, 0.0, 1.0)
            decay = max_decay * (1.0 - jnp.cos(jnp.pi * progress)) / 2.0
            enabled = step >= start_step
            return decay, enabled

        schedule = self.ema_schedule
        if schedule is not None:
            return schedule.get_decay_for_step(step)

        if self.ema_decay is None:
            import jax.numpy as jnp

            return jnp.asarray(0.0, dtype=jnp.float32), jnp.asarray(False)

        import jax.numpy as jnp

        return jnp.asarray(self.ema_decay, dtype=jnp.float32), jnp.asarray(True)

    @property
    @override
    def assets_dirs(self) -> pathlib.Path | epath.Path:
        """Assets directory (works for local paths and gs://…)."""
        return _to_path(self.assets_base_dir, self.name)

    @property
    @override
    def checkpoint_dir(self) -> pathlib.Path | epath.Path:
        """Checkpoint directory (local or Cloud Storage)."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return _to_path(self.checkpoint_base_dir, self.name, self.exp_name)


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    *create_multi_device_configs(
        base_name="pi_combined_cot",
        devices=["v6", "v6europe", "v4", "local", "v5", "v5europe"],
        model=pi_cot_config.PiCoTConfig(
            action_dim=7,
            action_horizon=16,
            max_token_len=220,
        ),
        data_config_class=RLDSCoTDataConfig,
        data_config_kwargs={
            "repo_id": "combined",
            "asset_id": "combined",
            "dataset_type": "combined",
            "droid_dataset_name": "droid",
            "data_mix": "oxe_magic_soup",
            "shuffle_buffer_size": 400_000,
            "action_proprio_normalization_type": NormalizationType.BOUNDS_Q99,
        },
        weight_loader=weight_loaders.WeightLoaderChoice(
            kind="paligemma"
        ),
        ema_schedule_choice=EmaScheduleChoice(kind="cosine_delayed", start_step=5000),
        optimizer=_optimizer.AdamW(weight_decay=0.0001),
        save_interval=1000,
        keep_period=5000,
        val_interval=2000,
        num_train_steps=40000,
        seed=0,
        resume=True,
    ),
    *create_multi_device_configs(
        base_name="gemma3_combined_cot",
        devices=["v6", "v6europe", "v4", "local", "v5", "v5europe"],
        model=pi_cot_config.PiCoTConfig(
            action_dim=7,
            action_horizon=16,
            enable_action_training=False,
            enable_langact_training=True,
            max_token_len=800,  # Gemma3 needs ~600+ tokens (512 image + prompt + reasoning)
            paligemma_variant="gemma3_4b",
            action_expert_variant="gemma3_300m",
            use_pan_and_scan=False,
        ),
        data_config_class=RLDSCoTDataConfig,
        data_config_kwargs={
            "repo_id": "combined",
            "asset_id": "combined",
            "dataset_type": "combined",
            "droid_dataset_name": "droid",
            "data_mix": "oxe_magic_soup",
            "shuffle_buffer_size": 400_000,
            "action_proprio_normalization_type": NormalizationType.BOUNDS_Q99,
            
        },
        weight_loader=weight_loaders.WeightLoaderChoice(kind="gemma3", params_path="gs://pi0-cot/cache/gemma3-4b-it"),
        ema_schedule_choice=EmaScheduleChoice(kind="cosine_delayed", start_step=5000),
        optimizer=_optimizer.AdamW(weight_decay=0.0001),
        save_interval=1000,
        keep_period=5000,
        val_interval=2000,
        num_train_steps=40000,
        seed=0,
        resume=True,
    ),
    # Combined dataset configs for v4, v6, v6europe
    *create_multi_device_configs(
        base_name="pi_combined_fast_cot",
        devices=["v6", "v6europe", "v4", "local", "v5", "v5europe"],
        model=pi_cot_config.PiCoTConfig(
            action_dim=7,
            action_horizon=16,
            max_token_len=220,
            use_fast=True,
        ),
        data_config_class=RLDSCoTDataConfig,
        data_config_kwargs={
            "repo_id": "combined",
            "asset_id": "combined",
            "dataset_type": "combined",
            "droid_dataset_name": "droid",
            "data_mix": "oxe_magic_soup",
            "shuffle_buffer_size": 400_000,
            "action_proprio_normalization_type": NormalizationType.BOUNDS_Q99,
        },
        weight_loader=weight_loaders.WeightLoaderChoice(kind="paligemma"),
        ema_schedule_choice=EmaScheduleChoice(kind="cosine_delayed", start_step=5000),
        optimizer=_optimizer.AdamW(weight_decay=0.0001),
        save_interval=1000,
        keep_period=5000,
        val_interval=2000,
        num_train_steps=40000,
        seed=0,
        resume=True,
    ),
    # VLA-0 Baseline: Actions as discretized integers (no language descriptions)
    # Reference: "VLA-0: Building State-of-the-Art VLAs with Zero Modification"
    *create_multi_device_configs(
        base_name="pi_combined_vla0",
        devices=["v6", "v6europe", "v4", "local", "v5", "v5europe"],
        model=pi_cot_config.PiCoTConfig(
            action_dim=7,
            action_horizon=10,  # 10-step action chunking
            max_token_len=390,  # VLA0 format is more compact
            pi05=True,
            discrete_state_input=True,
            enable_action_training=False,  # VLA0 uses language modeling loss only
            enable_langact_training=True,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            prompt_format="vla0_chunked",  # VLA-0 specific prompt format
        ),
        data_config_class=RLDSCoTDataConfig,
        data_config_kwargs={
            "repo_id": "combined",
            "asset_id": "combined",
            "dataset_type": "combined",
            "droid_dataset_name": "droid",
            "data_mix": "oxe_magic_soup",
            "shuffle_buffer_size": 400_000,
            "language_action_format_name": "vla0_chunked",  # VLA-0 format with action chunking
            "action_proprio_normalization_type": NormalizationType.BOUNDS_Q99,
        },
        weight_loader=weight_loaders.WeightLoaderChoice(kind="paligemma"),
        ema_schedule_choice=EmaScheduleChoice(kind="cosine_delayed", start_step=5000),
        optimizer=_optimizer.AdamW(weight_decay=0.0001),
        save_interval=1000,
        keep_period=5000,
        val_interval=2000,
        num_train_steps=40000,
        seed=0,
        resume=True,
    ),
    TrainConfig(
        name="paligemma_boundsq99",
        model=pi_cot_config.PiCoTConfig(
            action_horizon=16,
            max_token_len=180,
            pi05=True,
            discrete_state_input=True,
            enable_action_training=True,
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            prompt_format="pi05_notime",
        ),
        data=RLDSCoTDataConfig(
            action_proprio_normalization_type=NormalizationType.BOUNDS_Q99,
            repo_id="combined",
            asset_id="combined",
            dataset_type="combined",
        ),
    ),
    TrainConfig(
        name="pi05_vqa_v4",
        model=pi_cot_config.PiCoTConfig(
            pi05=True,
            discrete_state_input=False,
            max_token_len=600,
            prompt_format="vqa",
        ),
        data=VQADataConfig(
            repo_id="droid",
            asset_id="droid",
            dataset_type="droid",
            rlds_data_dir="gs://pi0-cot/OXE",
            language_action_dir="gs://pi0-cot/droid-base-lang-actions",
            droid_dataset_name="droid",
            droid_rlds_data_dir="gs://pi0-cot/OXE",
        ),
        fsdp_devices=1,
        batch_size=1,
        checkpoint_base_dir="gs://pi0-cot/checkpoints",
        weight_loader=weight_loaders.WeightLoaderChoice(
            kind="checkpoint",
            params_path="gs://openpi-assets/checkpoints/pi05_base/params",
        ),
    ),
    *upstream_config._CONFIGS,  # noqa: SLF001
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str, device: str | None = None) -> TrainConfig:
    """Get a config by name, optionally specifying device.

    Args:
        config_name: Name of the config to retrieve
        device: Optional device name to append (e.g., "v4", "v5", "local")

    Returns:
        The requested TrainConfig

    Examples:
        get_config("pi_droid_cot_v4")  # Direct lookup
        get_config("pi_droid_cot", device="v4")  # Lookup with device parameter
    """
    # Try exact match first
    if config_name in _CONFIGS_DICT:
        return _CONFIGS_DICT[config_name]

    # Try with device suffix if provided
    if device is not None:
        full_name = f"{config_name}_{device}"
        if full_name in _CONFIGS_DICT:
            return _CONFIGS_DICT[full_name]

    # Config not found - provide helpful error message
    search_name = f"{config_name}_{device}" if device else config_name
    closest = difflib.get_close_matches(search_name, _CONFIGS_DICT.keys(), n=3, cutoff=0.0)
    closest_str = f" Did you mean one of: {', '.join(repr(c) for c in closest)}?" if closest else ""

    available_devices = list(DEVICE_CONFIGS.keys())
    device_hint = f"\nAvailable devices: {', '.join(available_devices)}" if not device else ""

    raise ValueError(f"Config '{search_name}' not found.{closest_str}{device_hint}")
