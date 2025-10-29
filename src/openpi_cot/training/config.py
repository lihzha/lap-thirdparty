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

from openpi_cot.dataloader.helpers import ActionEncoding
from openpi_cot.dataloader.helpers import StateEncoding
import openpi_cot.models.adapters.model_adapter as _model_adapter
from openpi_cot.models.adapters.tokenizer_adapter import PaligemmaCoTTokenizer
import openpi_cot.models.pi_cot_config as pi_cot_config
import openpi_cot.policies.cot_policy as cot_policy
import openpi_cot.policies.libero_finetune_policy as libero_finetune_policy
import openpi_cot.policies.planning_policy as planning_policy
import openpi_cot.policies.vqa_policy as vqa_policy
import openpi_cot.shared.adapters.normalize_adapter as _normalize_adapter
from openpi_cot.shared.download import maybe_download
import openpi_cot.training.weight_loaders as weight_loaders
from openpi_cot.transforms import DetokenizeReasoning
from openpi_cot.transforms import TokenizePromptAndReasoning

ModelType: TypeAlias = _model_adapter.ExtendedModelType
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
    fsdp_devices: int = 1
    default_batch_size: int = 256

    def get_language_action_dir(self) -> str:
        """Get language action directory, deriving from rlds_data_dir if not set."""
        if self.language_action_dir is not None:
            return self.language_action_dir
        return self.rlds_data_dir.replace("OXE", "droid-base-lang-actions")


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
    action_horizon: int = 10,
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
    warmup_steps: int = 1_000,
    peak_lr: float = 1e-4,
    decay_steps: int = 1_000_000,
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
    # TODO: remove the cot argument
    cot: bool = False
    shuffle_buffer_size: int = 250_000
    # Optional cap on number of unique flattened samples for overfitting tests
    max_samples: int | None = None
    # Validation controls for RLDS-CoT dataset splitting/visualization
    val_max_samples: int | None = None
    val_fraction: float | None = 0.01
    use_wrist_image: bool = True
    wrist_image_dropout_prob: float = 0.0
    # One of {"droid", "oxe", "combined"}; used by the RLDS loader switch.
    dataset_type: Literal["droid", "oxe", "combined"] = "droid"
    state_encoding: StateEncoding = StateEncoding.POS_EULER
    action_encoding: ActionEncoding = ActionEncoding.EEF_POS
    resize_resolution: tuple[int, int] = (224, 224)

    # Language action configuration
    language_action_config_name: str = "compact"
    decoding_schema: str = "verbose"

    # Prediction training parameters
    max_prediction_horizon: int = 30
    prediction_prompt: str = "What is the robot's movement between two frames?"

    ### DROID fields (used when dataset_type == "droid")
    vis_dataset: bool = False
    language_action_dir: str | None = None
    # Optional path when DROID path is different from OXE path
    droid_rlds_data_dir: str | None = None
    # support using droid_subset for debugging
    droid_dataset_name: Literal["droid", "droid_subset"] = "droid"
    use_json_actions: bool = False
    force_recompute_stats: bool = False

    ### OXE fields (used when dataset_type == "oxe" or "combined")
    data_mix: str | None = "oxe_pi_magic_soup_with_other_states_with_bimanual"


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(upstream_config.ModelTransformFactory):
    """Creates model transforms for standard pi0 models."""

    prompt_format: Literal[
        "pi05",
        "pi0",
        "vqa",
        "coordinate_system",
        "schema_compact",
        "schema_compact_with_rotation",
        "schema_compact_bimanual",
        "schema_compact_bimanual_with_rotation",
    ] = "schema_compact"
    prediction_prompt: str = "What is the robot's movement between two frames?"
    tokenizer_type: Literal["gemma3", "paligemma"] = "paligemma"
    include_outputs: bool = True  # Toggle output transforms (e.g., detokenization)

    def __call__(self, model_config: _model.BaseModelConfig) -> upstream_transforms.Group:
        if model_config.model_type == ModelType.PI_COT:
            assert isinstance(model_config, pi_cot_config.PiCoTConfig)
            outputs = []
            if self.include_outputs:
                outputs = [
                    DetokenizeReasoning(
                        PaligemmaCoTTokenizer(
                            model_config.max_token_len,
                            prompt_format=self.prompt_format,
                            tokenizer_type=self.tokenizer_type,
                        )
                    )
                ]
            return upstream_transforms.Group(
                inputs=[
                    upstream_transforms.InjectDefaultPrompt(self.default_prompt),
                    # upstream_transforms.ResizeImages(224, 224),
                    TokenizePromptAndReasoning(
                        PaligemmaCoTTokenizer(
                            model_config.max_token_len,
                            prompt_format=self.prompt_format,
                            tokenizer_type=self.tokenizer_type,
                        ),
                        discrete_state_input=model_config.discrete_state_input,
                        prediction_prompt=self.prediction_prompt,
                    ),
                    upstream_transforms.PadStatesAndActions(model_config.action_dim),
                ],
                outputs=outputs,
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
            use_quantile_norm=model_config.model_type != ModelType.PI0,
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
            use_quantile_norm=model_config.model_type == ModelType.PI0_FAST,
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
                    language_action_config=cot_policy.get_language_action_config(base_cfg.language_action_config_name),
                    enable_prediction_training=model_config.enable_prediction_training,
                    prediction_prompt=base_cfg.prediction_prompt,
                )
            ],
            outputs=[cot_policy.CoTOutputs(decoding_schema=base_cfg.decoding_schema)],
        )

    @override
    def _create_model_transforms(
        self, base_cfg: CoTDataConfig, model_config: _model.BaseModelConfig
    ) -> upstream_transforms.Group:
        return ModelTransformFactory(
            prompt_format=model_config.prompt_format,
            prediction_prompt=base_cfg.prediction_prompt,
            tokenizer_type="gemma3" if "gemma3" in model_config.paligemma_variant else "paligemma",
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
            prediction_prompt=base_cfg.prediction_prompt,
            tokenizer_type="gemma3" if "gemma3" in model_config.paligemma_variant else "paligemma",
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
            prediction_prompt=base_cfg.prediction_prompt,
            tokenizer_type="gemma3" if "gemma3" in model_config.paligemma_variant else "paligemma",
            include_outputs=False,  # Libero doesn't need output detokenization
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
            prediction_prompt=base_cfg.prediction_prompt,
            prompt_format=model_config.prompt_format,
            tokenizer_type="gemma3" if "gemma3" in model_config.paligemma_variant else "paligemma",
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


def build_langact_prediction_schedule(
    *,
    num_train_steps: int = 100_000,
    langact_weight: float = 1.0,
    prediction_weight: float = 0.2,
    prediction_prob: float = 0.2,
    enable_prediction_training: bool = False,
) -> TrainingSchedule:
    """Build training schedule with language action and prediction losses.

    This schedule enables both language action and prediction training from the start,
    with configurable loss weights and stochastic masking.

    Args:
        num_train_steps: Total number of training steps
        langact_weight: Weight for language action loss (default: 1.0)
        prediction_weight: Weight for prediction loss (default: 0.2)
        prediction_prob: Probability of computing prediction loss per sample (default: 0.2)

    Returns:
        TrainingSchedule configured with the specified parameters

    Example:
        # From Python code:
        schedule = build_langact_prediction_schedule(num_train_steps=50000)
        config = TrainConfig(..., training_schedule=schedule)

        # From command line:
        python -m openpi_cot.training.train \\
            --config pi_droid_cot_v4 \\
            --training_schedule "$(python -c 'from openpi_cot.training.config import build_langact_prediction_schedule; print(build_langact_prediction_schedule())')"
    """
    return TrainingSchedule(
        stages=(
            TrainingStage(
                start_step=0,
                end_step=None,  # Until end of training
                enable_langact_training=True,
                enable_action_training=False,
                enable_prediction_training=enable_prediction_training,
                language_loss_weight=langact_weight,
                action_loss_weight=1.0,  # Not used when enable_action_training=False
                prediction_loss_weight=prediction_weight,
                langact_prob=1.0,  # Always compute language action loss
                action_prob=1.0,
                prediction_prob=prediction_prob,  # Stochastically compute prediction loss
            ),
        )
    )


class ConfigBuilder:
    """Builder for creating TrainConfig with device-specific overrides.

    Enables programmatic generation of configs across multiple devices while
    avoiding repetitive configuration code.

    Example:
        # Generate configs for multiple devices
        configs = (ConfigBuilder("pi_droid_cot")
            .with_model(build_picot_model())
            .with_data(RLDSCoTDataConfig, repo_id="droid", dataset_type="droid")
            .build_for_devices(["v4", "v5", "v6", "local"]))
    """

    def __init__(self, base_name: str):
        """Initialize builder with a base config name.

        Args:
            base_name: Base name for the config (device suffix will be added)
        """
        self.base_name = base_name
        self._model_config = None
        self._data_config_class = None
        self._data_config_kwargs = {}
        self._train_config_kwargs = {}

    def with_model(self, model_config: _model.BaseModelConfig) -> "ConfigBuilder":
        """Set model configuration."""
        self._model_config = model_config
        return self

    def with_data(
        self,
        data_config_class: type[upstream_config.DataConfigFactory],
        **kwargs,
    ) -> "ConfigBuilder":
        """Set data configuration class and its parameters.

        Args:
            data_config_class: Class to instantiate for data config
            **kwargs: Parameters to pass to data config constructor
        """
        self._data_config_class = data_config_class
        self._data_config_kwargs = kwargs
        return self

    def with_training(self, **kwargs) -> "ConfigBuilder":
        """Set additional training config parameters.

        Args:
            **kwargs: Parameters to pass to TrainConfig constructor
        """
        self._train_config_kwargs.update(kwargs)
        return self

    def build(self, name: str | None = None, **overrides) -> "TrainConfig":
        """Build a single TrainConfig.

        Args:
            name: Optional name override (defaults to base_name)
            **overrides: Additional TrainConfig parameters to override

        Returns:
            Configured TrainConfig instance
        """
        config_name = name or self.base_name

        # Merge all train config parameters
        train_params = {
            "name": config_name,
            **self._train_config_kwargs,
            **overrides,
        }

        # Add model if specified
        if self._model_config is not None:
            train_params["model"] = self._model_config

        # Add data config if specified
        if self._data_config_class is not None:
            train_params["data"] = self._data_config_class(**self._data_config_kwargs)

        return TrainConfig(**train_params)

    def build_for_devices(self, devices: list[str]) -> list["TrainConfig"]:
        """Generate config variants for multiple devices.

        Args:
            devices: List of device names (must exist in DEVICE_CONFIGS)

        Returns:
            List of TrainConfig instances, one per device
        """
        configs = []
        for device in devices:
            if device not in DEVICE_CONFIGS:
                raise ValueError(f"Unknown device '{device}'. Available devices: {list(DEVICE_CONFIGS.keys())}")

            device_cfg = DEVICE_CONFIGS[device]
            config_name = f"{self.base_name}_{device}"

            # Build data config kwargs with device-specific paths
            data_kwargs = {**self._data_config_kwargs}
            if "rlds_data_dir" not in data_kwargs:
                data_kwargs["rlds_data_dir"] = device_cfg.rlds_data_dir
            if "language_action_dir" not in data_kwargs:
                data_kwargs["language_action_dir"] = device_cfg.get_language_action_dir()

            # Build train config with device-specific settings
            train_overrides = {
                "fsdp_devices": device_cfg.fsdp_devices,
                "checkpoint_base_dir": device_cfg.checkpoint_base_dir,
            }

            # Set batch_size from device default if not explicitly specified
            if "batch_size" not in self._train_config_kwargs:
                train_overrides["batch_size"] = device_cfg.default_batch_size

            # Create a builder copy with device-specific data config
            device_builder = ConfigBuilder(self.base_name)
            device_builder._model_config = self._model_config
            device_builder._data_config_class = self._data_config_class
            device_builder._data_config_kwargs = data_kwargs
            device_builder._train_config_kwargs = {**self._train_config_kwargs, **train_overrides}

            configs.append(device_builder.build(name=config_name))

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
    save_interval: int = 500
    log_interval: int = 50
    keep_period: int | None = 10000
    resume: bool = True
    # New field
    do_val: bool = True
    val_interval: int = 2000
    checkpoint_async_timeout_secs: int | None = 7200
    checkpoint_async_enable: bool = True
    checkpoint_max_retries: int = 1
    checkpoint_retry_delay_secs: float = 30.0
    checkpoint_retry_backoff: float = 2.0
    checkpoint_fallback_to_sync: bool = True
    # Evaluation fields
    eval_checkpoint_step: int | None = None
    num_eval_batches: int | None = None
    eval_mode: Literal["token_accuracy", "rollout", "both"] = "token_accuracy"
    # Multi-stage training
    training_schedule: TrainingSchedule | None = None

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
    # Multi-device configs: Generated programmatically for v4, v5, v6, local
    *ConfigBuilder("pi_droid_cot")
    .with_model(build_picot_model())
    .with_data(
        RLDSCoTDataConfig,
        repo_id="droid",
        asset_id="droid",
        dataset_type="droid",
        droid_dataset_name="droid",
    )
    .with_training(
        weight_loader=weight_loaders.WeightLoaderChoice(kind="paligemma"),
    )
    .build_for_devices(["v4", "v5", "v6", "local"]),
    # Combined dataset configs for v4, v6, v6europe
    *ConfigBuilder("pi_combined_cot")
    .with_model(build_picot_model())
    .with_data(
        RLDSCoTDataConfig,
        repo_id="combined",
        asset_id="combined",
        dataset_type="combined",
        droid_dataset_name="droid",
        data_mix="oxe_pi_magic_soup_with_other_states_with_bimanual",
        shuffle_buffer_size=400_000,
    )
    .with_training(
        weight_loader=weight_loaders.WeightLoaderChoice(
            kind="checkpoint", params_path="gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        save_interval=500,
        keep_period=5000,
        resume=True,
        training_schedule=build_langact_prediction_schedule(
            num_train_steps=100_000,
            langact_weight=1.0,
            prediction_weight=0.0,
            prediction_prob=0.2,
            enable_prediction_training=False,
        ),
    )
    .build_for_devices(["v6", "v6europe", "v4", "local"]),
    # Gemma3 configs
    ConfigBuilder("gemma3_combined_cot")
    .with_model(
        pi_cot_config.PiCoTConfig(
            pi05=True,
            discrete_state_input=False,
            max_token_len=600,
            paligemma_variant="gemma3_4b",
            action_expert_variant="gemma3_300m",
            prompt_format="pi05",
        )
    )
    .with_data(
        RLDSCoTDataConfig,
        repo_id="combined",
        asset_id="combined",
        dataset_type="combined",
        droid_dataset_name="droid",
        shuffle_buffer_size=400_000,
    )
    .with_training(
        fsdp_devices=1,
        batch_size=1,
        weight_loader=weight_loaders.WeightLoaderChoice(kind="gemma3", params_path="gs://pi0-cot/cache/gemma3-4b-it"),
        save_interval=500,
        keep_period=10000,
        resume=True,
    )
    .build(
        name="gemma3_combined_cot_v4", rlds_data_dir="gs://pi0-cot/OXE", checkpoint_base_dir="gs://pi0-cot/checkpoints"
    ),
    ConfigBuilder("gemma3_droid_cot_lora")
    .with_model(
        pi_cot_config.PiCoTConfig(
            pi05=True,
            discrete_state_input=False,
            max_token_len=600,
            paligemma_variant="gemma3_4b_lora",
            action_expert_variant="gemma3_300m",
            prompt_format="pi05",
        )
    )
    .with_data(
        RLDSCoTDataConfig,
        repo_id="droid",
        asset_id="droid",
        dataset_type="droid",
        droid_dataset_name="droid",
        shuffle_buffer_size=400_000,
    )
    .with_training(
        fsdp_devices=1,
        batch_size=1,
        save_interval=500,
        keep_period=10000,
        resume=True,
    )
    .build(
        name="gemma3_droid_cot_lora_v5",
        rlds_data_dir="gs://v5_central1_a/OXE",
        language_action_dir="gs://v5_central1_a/droid-base-lang-actions",
        checkpoint_base_dir="gs://v5_central1_a/checkpoints",
        weight_loader=weight_loaders.WeightLoaderChoice(
            kind="gemma3", params_path="gs://v5_central1_a/cache/gemma3-4b-it"
        ),
    ),
    # Evaluation and special single-instance configs
    TrainConfig(
        name="pi0_eval",
        model=pi_cot_config.PiCoTConfig(
            action_horizon=10,
            max_token_len=110,
            pi05=False,
            discrete_state_input=False,
        ),
        data=RLDSCoTDataConfig(
            repo_id="droid",
            asset_id="droid",
            dataset_type="droid",
        ),
    ),
    TrainConfig(
        name="pi05_eval",
        model=pi_cot_config.PiCoTConfig(
            action_horizon=10,
            max_token_len=200,
            pi05=True,
            discrete_state_input=True,
        ),
        data=RLDSCoTDataConfig(
            repo_id="droid",
            asset_id="droid",
            dataset_type="droid",
        ),
    ),
    TrainConfig(
        name="paligemma2_eval",
        model=pi_cot_config.PiCoTConfig(
            action_horizon=10,
            max_token_len=150,
            pi05=True,
            discrete_state_input=True,
            paligemma_variant="gemma2_2b",
            action_expert_variant="gemma2_300m",
        ),
        data=RLDSCoTDataConfig(
            repo_id="droid",
            asset_id="droid",
            dataset_type="droid",
        ),
    ),
    TrainConfig(
        name="paligemma2_eval_compact",
        model=pi_cot_config.PiCoTConfig(
            action_horizon=10,
            max_token_len=150,
            pi05=True,
            discrete_state_input=True,
            paligemma_variant="gemma2_2b",
            action_expert_variant="gemma2_300m",
        ),
        data=RLDSCoTDataConfig(repo_id="droid", asset_id="droid", dataset_type="droid", decoding_schema="compact"),
    ),
    TrainConfig(
        name="pi05_vqa",
        model=pi_cot_config.PiCoTConfig(pi05=True, discrete_state_input=False, max_token_len=600),
        data=VQADataConfig(
            repo_id="droid",
            asset_id="droid",
            dataset_type="droid",
            rlds_data_dir="/n/fs/robot-data/data/",
            language_action_dir="/n/fs/robot-data/vlm-syn/droid-lang-actions",
            droid_dataset_name="droid",
            droid_rlds_data_dir="/n/fs/robot-data/data/",
        ),
        fsdp_devices=1,
        batch_size=1,
        checkpoint_base_dir="/n/fs/robot-data/pi0-cot/checkpoints",
        weight_loader=weight_loaders.WeightLoaderChoice(kind="paligemma2", params_path="tbd"),
    ),
    TrainConfig(
        name="pi05_libero_finetune_v4",
        model=pi_cot_config.PiCoTConfig(
            action_horizon=10,
            max_token_len=180,
            pi05=True,
            discrete_state_input=True,
            enable_action_training=True,
            enable_langact_training=False,
            paligemma_variant="gemma2_2b",
            action_expert_variant="gemma2_300m",
        ),
        data=LiberoFinetuneDataConfig(
            repo_id="libero",
            asset_id="libero",
            dataset_type="combined",
            data_mix="libero_finetune",
            rlds_data_dir="gs://pi0-cot/OXE",  # Update this path
            language_action_config_name="default",
            decoding_schema="default",
        ),
        fsdp_devices=1,
        batch_size=32,
        num_train_steps=50000,
        save_interval=500,
        log_interval=100,
        keep_period=500,
        checkpoint_base_dir="gs://pi0-cot/checkpoints",  # Update this path
        weight_loader=weight_loaders.WeightLoaderChoice(
            kind="checkpoint",
            params_path="gs://pi0-cot/checkpoints/pi_combined_cot_v4/oxe_no_galaxea_v4_fixed/30000/params",
        ),
    ),
    TrainConfig(
        name="pi05_libero_finetune_v4_freezevlm",
        model=pi_cot_config.PiCoTConfig(
            action_horizon=10,
            max_token_len=180,
            pi05=True,
            discrete_state_input=True,
            enable_action_training=True,
            enable_langact_training=False,
            paligemma_variant="gemma2_2b",
            action_expert_variant="gemma2_300m",
        ),
        data=LiberoFinetuneDataConfig(
            repo_id="libero",
            asset_id="libero",
            dataset_type="combined",
            data_mix="libero_finetune",
            rlds_data_dir="gs://pi0-cot/OXE",  # Update this path
            language_action_config_name="default",
            decoding_schema="default",
        ),
        fsdp_devices=1,
        batch_size=32,
        num_train_steps=50000,
        save_interval=500,
        log_interval=100,
        keep_period=500,
        checkpoint_base_dir="gs://pi0-cot/checkpoints",  # Update this path
        weight_loader=weight_loaders.WeightLoaderChoice(
            kind="checkpoint",
            params_path="gs://pi0-cot/checkpoints/pi_combined_cot_v4/oxe_no_galaxea_v4_fixed/30000/params",
        ),
        freeze_filter=pi_cot_config.PiCoTConfig(
            paligemma_variant="gemma2_2b",
            action_expert_variant="gemma2_300m",
        ).get_vlm_freeze_filter(),
    ),
    TrainConfig(
        name="pi05_libero_finetune_local",
        model=pi_cot_config.PiCoTConfig(
            action_horizon=10,
            max_token_len=180,
            pi05=True,
            discrete_state_input=True,
            enable_action_training=True,
            enable_langact_training=True,
        ),
        data=LiberoFinetuneDataConfig(
            repo_id="libero_10_no_noops",
            asset_id="libero",
            dataset_type="oxe",
            data_mix="libero_10_no_noops",
            rlds_data_dir="/n/fs/robot-data/libero/tfds",
            language_action_config_name="compact",
            decoding_schema="compact",
        ),
        fsdp_devices=1,
        batch_size=8,
        num_train_steps=50000,
        save_interval=1000,
        log_interval=100,
        keep_period=5000,
        checkpoint_base_dir="/n/fs/robot-data/pi0-cot/checkpoints",
        weight_loader=weight_loaders.WeightLoaderChoice(kind="paligemma"),
    ),
    TrainConfig(
        name="paligemma2_vqa",
        model=pi_cot_config.PiCoTConfig(
            pi05=True,
            discrete_state_input=False,
            max_token_len=600,
            paligemma_variant="gemma2_2b",
            action_expert_variant="gemma2_300m",
            prompt_format="vqa",
        ),
        data=VQADataConfig(
            repo_id="droid",
            asset_id="droid",
            dataset_type="droid",
            rlds_data_dir="/n/fs/robot-data/data/",
            language_action_dir="/n/fs/robot-data/vlm-syn/droid-lang-actions",
            droid_dataset_name="droid",
            droid_rlds_data_dir="/n/fs/robot-data/data/",
        ),
        fsdp_devices=1,
        batch_size=1,
        checkpoint_base_dir="/n/fs/robot-data/pi0-cot/checkpoints",
        weight_loader=weight_loaders.WeightLoaderChoice(
            kind="paligemma2", params_path="/n/fs/robot-data/openpi-cot/paligemma2-3b-mix-224.b16.npz"
        ),
    ),
    TrainConfig(
        name="paligemma2_vqa_v4",
        model=pi_cot_config.PiCoTConfig(
            pi05=True,
            discrete_state_input=False,
            max_token_len=600,
            paligemma_variant="gemma2_2b",
            action_expert_variant="gemma2_300m",
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
            kind="paligemma2", params_path="gs://pi0-cot/cache/paligemma2-3b-pt-224.b16.npz"
        ),
    ),
    TrainConfig(
        name="gemma3_vqa_v4",
        model=pi_cot_config.PiCoTConfig(
            pi05=True,
            discrete_state_input=False,
            max_token_len=800,
            paligemma_variant="gemma3_4b",
            action_expert_variant="gemma3_300m",
            prompt_format="vqa",
        ),
        # batch_size=256,
        # weight_loader=weight_loaders.WeightLoaderChoice(
        #     kind="checkpoint", params_path="gs://openpi-assets/checkpoints/pi05_base/params"
        # ),
        # checkpoint_base_dir="gs://pi0-cot/checkpoints",
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
        weight_loader=weight_loaders.WeightLoaderChoice(kind="gemma3", params_path="gs://pi0-cot/cache/gemma3-4b-it"),
    ),
    TrainConfig(
        name="gemma3_vqa_local",
        model=pi_cot_config.PiCoTConfig(
            pi05=True,
            discrete_state_input=False,
            max_token_len=600,
            paligemma_variant="gemma3_4b",
            action_expert_variant="gemma3_300m",
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
        checkpoint_base_dir="/home/ajhancock/Desktop/openpi-cot/src/openpi_cot/ckpts/",
        weight_loader=weight_loaders.WeightLoaderChoice(
            kind="gemma3", params_path="/home/ajhancock/Desktop/openpi-cot/src/openpi_cot/ckpts/gemma3-4b-it"
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
    TrainConfig(
        name="pi05_planning_finetune_v4",
        model=pi_cot_config.PiCoTConfig(
            action_horizon=10,
            max_token_len=180,
            pi05=True,
            discrete_state_input=True,
            enable_action_training=True,
            enable_langact_training=False,
            prompt_format="pi05",
        ),
        data=PlanningDataConfig(
            repo_id="planning_dataset",
            asset_id="planning",
            dataset_type="combined",
            data_mix="planning_dataset",
            rlds_data_dir="gs://pi0-cot/OXE",
        ),
        fsdp_devices=1,
        batch_size=4,
        num_train_steps=100000,
        save_interval=2500,
        log_interval=100,
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
