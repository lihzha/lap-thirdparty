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


import openpi_cot.models.model_adapter as _model_adapter
import openpi_cot.models.pi_cot_config as pi_cot_config
from openpi_cot.models.tokenizer import FASTTokenizer
from openpi_cot.models.tokenizer import PaligemmaCoTTokenizer
import openpi_cot.shared.adapters.normalize_adapter as _normalize_adapter
from openpi_cot.shared.download import maybe_download
from openpi_cot.transforms import DetokenizeReasoning
from openpi_cot.transforms import ExtractFASTActions
from openpi_cot.transforms import TokenizeFASTCoTInputs
from openpi_cot.transforms import TokenizePromptAndReasoning

ModelType: TypeAlias = _model_adapter.ExtendedModelType
UpstreamModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter



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
    # Normalization type for actions and proprioceptive state.
    # CLI: --data.action_proprio_normalization_type {normal|bounds|bounds_q99}
    action_proprio_normalization_type: Literal["normal", "bounds", "bounds_q99"] = "bounds_q99"
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

    # VQA bbox dataset parameters
    direction_prob: float = 0.0  # Probability of using direction caption instead of bbox for bbox VQA datasets

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

    def _create_tokenizer(self, model_config: pi_cot_config.PiCoTConfig, reasoning_mask_prob: float):
        """Create the appropriate tokenizer based on model variant."""
        
        return PaligemmaCoTTokenizer(
            model_config.max_token_len,
            prompt_format=self.prompt_format,
            prediction_format=self.prediction_format,
            reasoning_mask_prob=reasoning_mask_prob,
        )

    def __call__(self, model_config: _model.BaseModelConfig) -> upstream_transforms.Group:
        if model_config.model_type == ModelType.PI_COT:
            assert isinstance(model_config, pi_cot_config.PiCoTConfig)
            outputs = []
            if self.include_outputs:
                outputs = [
                    DetokenizeReasoning(
                        self._create_tokenizer(model_config, reasoning_mask_prob=0)
                    )
                ]
            return upstream_transforms.Group(
                inputs=[
                    upstream_transforms.InjectDefaultPrompt(self.default_prompt),
                    # upstream_transforms.ResizeImages(224, 224),
                    TokenizePromptAndReasoning(
                        self._create_tokenizer(model_config, reasoning_mask_prob=model_config.reasoning_mask_prob),
                        discrete_state_input=model_config.discrete_state_input,
                        verbose_mode=model_config.verbose_mode,
                        state_dropout=model_config.state_dropout,
                    ),
                    upstream_transforms.PadStatesAndActions(model_config.action_dim),
                ],
                outputs=outputs,
            )
        if model_config.model_type in (ModelType.PI_FAST, UpstreamModelType.PI0_FAST):
            assert isinstance(model_config, pi_cot_config.PiCoTConfig)
            fast_tokenizer = FASTTokenizer(
                fast_tokenizer_path=self.fast_tokenizer_path,
                max_len=model_config.max_token_len,
                prompt_format=self.prompt_format,
                prediction_format=self.prediction_format,
            )
            return upstream_transforms.Group(
                inputs=[
                    upstream_transforms.InjectDefaultPrompt(self.default_prompt),
                    # upstream_transforms.ResizeImages(224, 224),
                    TokenizeFASTCoTInputs(
                        fast_tokenizer,
                        discrete_state_input=model_config.discrete_state_input,
                        state_dropout=model_config.state_dropout,
                    ),
                    # PadStates(model_config.action_dim),
                ],
                outputs=[
                    ExtractFASTActions(
                        fast_tokenizer,
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
                    use_rough_scale=False,
                    enable_langact_training=model_config.enable_langact_training,
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
            prediction_format=model_config.prediction_format,
        )(model_config)


@dataclasses.dataclass(frozen=True)
class TrainConfig(upstream_config.TrainConfig):
    # Overide
    project_name: str = "openpi-cot"
    model: _model.BaseModelConfig = dataclasses.field(default_factory=build_picot_model)
    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=build_cosine_lr)
    num_train_steps: int = 100_000
    save_interval: int = 2500
    # Additional steps at which to save checkpoints (beyond save_interval)
    additional_save_steps: tuple[int, ...] | None = None
    log_interval: int = 50
    keep_period: int | None = 10000
    resume: bool = True
    ema_decay: float | None = 0.999
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
    eval_checkpoint_steps: list[int] | None = None  # List of specific checkpoint steps to evaluate
    eval_all_checkpoints: bool = True  # If True, evaluate all available checkpoints sequentially
    num_eval_batches: int | None = 463
    eval_mode: Literal["token_accuracy", "rollout", "both", "token_visualization", "train_loss", "val_loss", "action_prediction_loss"] = "val_loss"
    eval_use_ema: bool = True
    eval_split: Literal["val", "train"] = "val"
    eval_load_params_directly: bool = False

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
            action_proprio_normalization_type="bounds_q99",
            repo_id="combined",
            asset_id="combined",
            dataset_type="combined",
            language_action_format_name="verbose_with_rotation",
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

    raise ValueError(f"Config '{search_name}' not found.{closest_str}")
