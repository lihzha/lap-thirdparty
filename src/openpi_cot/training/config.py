"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, TypeAlias

import etils.epath as epath
import flax.nnx as nnx
import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.tokenizer as _tokenizer
import openpi.training.config as upstream_config
import openpi.training.optimizer as _optimizer
import openpi.transforms as upstream_transforms
from typing_extensions import override
import tyro

import openpi_cot.dataloader.cot_rlds_dataset as cot_rlds_dataset
import openpi_cot.models.adapters.model_adapter as _model_adapter
from openpi_cot.models.adapters.tokenizer_adapter import PaligemmaCoTTokenizer
import openpi_cot.models.pi_cot_config as pi_cot_config
import openpi_cot.policies.droid_cot_policy as droid_cot_policy
import openpi_cot.shared.adapters.normalize_adapter as _normalize
import openpi_cot.shared.download as _download
import openpi_cot.training.weight_loaders as weight_loaders
from openpi_cot.transforms import DetokenizeReasoning
from openpi_cot.transforms import TokenizePromptAndReasoning

ModelType: TypeAlias = _model_adapter.ExtendedModelType
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


@dataclasses.dataclass(frozen=True)
class CoTDataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, upstream_transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: upstream_transforms.Group = dataclasses.field(default_factory=upstream_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: upstream_transforms.Group = dataclasses.field(default_factory=upstream_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: upstream_transforms.Group = dataclasses.field(default_factory=upstream_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)

    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: cot_rlds_dataset.DroidActionSpace | None = None
    # Path to the data filter file for DROID dataset
    filter_dict_path: str | None = None

    # For DROID-CoT (optional; used only when `cot` is True)
    cot: bool = False
    language_action_dir: str | None = None
    shuffle_buffer_size: int = 250_000
    # For CoT-style datasets (e.g., DROID-CoT): number of future steps to sum over for language actions
    summation_steps: int = 15
    # Optional cap on number of unique flattened samples for overfitting tests
    max_samples: int | None = None
    # Tokenization / formatting controls for CoT numeric aggregation
    sum_decimal: str = "2f"
    left_pad: bool = True
    include_decimal_point: bool = True
    # Validation controls for RLDS-CoT dataset splitting/visualization
    val_max_samples: int | None = None
    val_fraction: float | None = None
    validation_mode: str = "easy"
    vis_dataset: bool = False
    use_wrist_image: bool = False
    use_idle_filter: bool = True
    wrist_image_dropout_prob: float = 0.0
    text_state_dropout_prob: float = 0.0
    # If true, will drop samples where projected gripper is outside the resized image bounds.
    drop_gripper_oob: bool = False


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(upstream_config.GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None
    left_pad: bool = True
    include_decimal_point: bool = True

    def __call__(self, model_config: _model.BaseModelConfig) -> upstream_transforms.Group:
        match model_config.model_type:
            case ModelType.PI0:
                return upstream_transforms.Group(
                    inputs=[
                        upstream_transforms.InjectDefaultPrompt(self.default_prompt),
                        upstream_transforms.ResizeImages(224, 224),
                        upstream_transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        upstream_transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return upstream_transforms.Group(
                    inputs=[
                        upstream_transforms.InjectDefaultPrompt(self.default_prompt),
                        upstream_transforms.ResizeImages(224, 224),
                        upstream_transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        upstream_transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return upstream_transforms.Group(
                    inputs=[
                        upstream_transforms.InjectDefaultPrompt(self.default_prompt),
                        upstream_transforms.ResizeImages(224, 224),
                        upstream_transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        upstream_transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )
            case ModelType.PI_COT:
                assert isinstance(model_config, pi_cot_config.PiCoTConfig)
                return upstream_transforms.Group(
                    inputs=[
                        upstream_transforms.InjectDefaultPrompt(self.default_prompt),
                        upstream_transforms.ResizeImages(224, 224),
                        TokenizePromptAndReasoning(
                            PaligemmaCoTTokenizer(
                                model_config.max_token_len,
                                left_pad=self.left_pad,
                                include_decimal_point=self.include_decimal_point,
                            ),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        upstream_transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                    outputs=[
                        DetokenizeReasoning(
                            PaligemmaCoTTokenizer(
                                model_config.max_token_len,
                                left_pad=self.left_pad,
                                include_decimal_point=self.include_decimal_point,
                            )
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    # Determines how the assets will be loaded.
    assets: upstream_config.AssetsConfig = dataclasses.field(default_factory=upstream_config.AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[CoTDataConfig | None] = None

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> CoTDataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> CoTDataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        return dataclasses.replace(
            self.base_config or CoTDataConfig(),
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(
        self, assets_dir: epath.Path, asset_id: str | None
    ) -> dict[str, upstream_transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class RLDSDroidCoTDataConfig(DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    rlds_data_dir: str | None = None
    action_space: cot_rlds_dataset.DroidActionSpace | None = None
    cot: bool = True
    language_action_dir: str = "/n/fs/robot-data/vlm-syn/posed_droid"
    shuffle_buffer_size: int = 250_000
    # Number of future steps to sum over for language actions
    summation_steps: int = 15
    max_samples: int | None = None
    sum_decimal: str = "2f"
    left_pad: bool = True
    include_decimal_point: bool = True

    # If set, validation loader will materialize a fixed subset of this many
    # flattened samples via take(K).cache().repeat(), ensuring consistent val batches.
    val_max_samples: int | None = None
    val_fraction: float | None = None
    validation_mode: str = "easy"
    vis_dataset: bool = False
    use_wrist_image: bool = False
    use_idle_filter: bool = True
    # Train-time dropout (applied in DroidCoTInputs). Set nonzero only for training.
    wrist_image_dropout_prob: float = 0.0
    text_state_dropout_prob: float = 0.0
    # Drop samples where gripper is out of view after projection to resized image
    drop_gripper_oob: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> CoTDataConfig:
        # Load base config first to access norm stats (for state binning in prompt augmentation)
        base_cfg = self.create_base_config(assets_dirs, model_config)

        repack_dict = {
            # lihan: always name base image as "exterior_image_1_left", though it should come from the camera which language action is annotated.
            "observation/exterior_image_1_left": "observation/image",
            "observation/cartesian_position": "observation/cartesian_position",
            "observation/gripper_position": "observation/gripper_position",
            "actions": "actions",
            "prompt": "prompt",
            "language_actions": "language_actions",
        }
        if self.vis_dataset:
            repack_dict["camera_intrinsics"] = "camera_intrinsics"
            repack_dict["camera_extrinsics"] = "camera_extrinsics"
            repack_dict["observation/cartesian_position_window"] = "observation/cartesian_position_window"
        if self.use_wrist_image:
            repack_dict["observation/wrist_image_left"] = "observation/wrist_image"
        repack_transform = upstream_transforms.Group(inputs=[upstream_transforms.RepackTransform(repack_dict)])

        # Extract state norm stats (if available) to pass into the DroidCoTInputs for binning
        state_stats = None
        if base_cfg.norm_stats is not None and "state" in base_cfg.norm_stats:
            state_stats = base_cfg.norm_stats["state"]

        data_transforms = upstream_transforms.Group(
            inputs=[
                droid_cot_policy.DroidCoTInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                    sum_decimal=self.sum_decimal,
                    state_norm_stats=state_stats,
                    wrist_image_dropout_prob=self.wrist_image_dropout_prob,
                    text_state_dropout_prob=self.text_state_dropout_prob,
                )
            ],
            outputs=[droid_cot_policy.DroidCoTOutputs()],
        )

        assert self.action_space == cot_rlds_dataset.DroidActionSpace.CARTESIAN_POSITION
        # Data loader returns absolute joint position actions -- convert to delta actions for training.
        delta_action_mask = upstream_transforms.make_bool_mask(6, -1)
        data_transforms = data_transforms.push(
            inputs=[upstream_transforms.DeltaActions(delta_action_mask)],
            # outputs=[upstream_transforms.AbsoluteActions(delta_action_mask)],
        )

        model_transforms = ModelTransformFactory(
            left_pad=self.left_pad, include_decimal_point=self.include_decimal_point
        )(model_config)

        assert self.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            base_cfg,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            use_quantile_norm=model_config.model_type == ModelType.PI0_FAST,
            rlds_data_dir=self.rlds_data_dir,
            action_space=self.action_space,
            cot=self.cot,
            language_action_dir=self.language_action_dir,
            shuffle_buffer_size=self.shuffle_buffer_size,
            summation_steps=self.summation_steps,
            max_samples=self.max_samples,
            sum_decimal=self.sum_decimal,
            left_pad=self.left_pad,
            include_decimal_point=self.include_decimal_point,
            use_wrist_image=self.use_wrist_image,
            val_max_samples=self.val_max_samples,
            val_fraction=self.val_fraction,
            validation_mode=self.validation_mode,
            vis_dataset=self.vis_dataset,
            use_idle_filter=self.use_idle_filter,
            drop_gripper_oob=self.drop_gripper_oob,
            wrist_image_dropout_prob=self.wrist_image_dropout_prob,
            text_state_dropout_prob=self.text_state_dropout_prob,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    # Uses a CLI-friendly choice wrapper to avoid nested subcommands.
    weight_loader: weight_loaders.WeightLoaderChoice = dataclasses.field(
        default_factory=weight_loaders.WeightLoaderChoice
    )

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=upstream_config.FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 30_000

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 1000
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True
    # If set, will rewind wandb run to this step when resuming (requires wandb SDK >= 0.17.1)
    rewind_to_step: int | None = None

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    # Do validation or not
    do_val: bool = False

    @property
    def assets_dirs(self) -> pathlib.Path | epath.Path:
        """Assets directory (works for local paths and gs://…)."""
        return _to_path(self.assets_base_dir, self.name)

    @property
    def checkpoint_dir(self) -> pathlib.Path | epath.Path:
        """Checkpoint directory (local or Cloud Storage)."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return _to_path(self.checkpoint_base_dir, self.name, self.exp_name)

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    TrainConfig(
        name="pi_droid_cot_v4",
        do_val=True,
        model=pi_cot_config.PiCoTConfig(
            action_horizon=10,
            max_token_len=110,
            number_token_weight=1.0,
            pi05=True,
            discrete_state_input=True,
        ),
        data=RLDSDroidCoTDataConfig(
            repo_id="droid",
            rlds_data_dir="gs://pi0-cot",
            language_action_dir="gs://pi0-cot/droid-lang-actions",
            action_space=cot_rlds_dataset.DroidActionSpace.CARTESIAN_POSITION,
            base_config=CoTDataConfig(
                prompt_from_task=True,
            ),
            shuffle_buffer_size=250_000,
            assets=upstream_config.AssetsConfig(
                assets_dir="gs://pi0-cot/assets/pi0_droid_cot_v4",
                asset_id="droid",
            ),
            summation_steps=15,
            sum_decimal="0f",
            left_pad=True,
            include_decimal_point=False,
            validation_mode="easy",
            vis_dataset=False,
            use_wrist_image=False,
            val_max_samples=60000,
            val_fraction=0.02,
            use_idle_filter=True,
            drop_gripper_oob=False,
        ),
        num_train_steps=100_000,
        fsdp_devices=4,
        batch_size=256,
        log_interval=50,
        save_interval=500,
        weight_loader=weight_loaders.WeightLoaderChoice(kind="paligemma"),
        keep_period=10000,
        # weight_loader=weight_loaders.WeightLoaderChoice(kind="checkpoint", params_path="gs://openpi-assets/checkpoints/pi0_base/params"),
        assets_base_dir="gs://pi0-cot/assets",
        checkpoint_base_dir="gs://pi0-cot/checkpoints",
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=1e-4,
            decay_steps=1_000_000,
            decay_lr=1e-4,
        ),
    ),
    TrainConfig(
        name="pi_droid_cot_v6",
        do_val=True,
        model=pi_cot_config.PiCoTConfig(
            action_horizon=10,
            max_token_len=110,
            number_token_weight=1.0,
            pi05=True,
            discrete_state_input=True,
        ),
        data=RLDSDroidCoTDataConfig(
            repo_id="droid",
            rlds_data_dir="gs://v6_east1d",
            language_action_dir="gs://v6_east1d/droid-lang-actions",
            action_space=cot_rlds_dataset.DroidActionSpace.CARTESIAN_POSITION,
            base_config=CoTDataConfig(
                prompt_from_task=True,
            ),
            shuffle_buffer_size=250_000,
            assets=upstream_config.AssetsConfig(
                assets_dir="gs://v6_east1d/assets/pi0_droid_cot_v4",
                asset_id="droid",
            ),
            summation_steps=15,
            sum_decimal="0f",
            left_pad=True,
            include_decimal_point=False,
            validation_mode="easy",
            vis_dataset=False,
            use_wrist_image=False,
            val_max_samples=60000,
            val_fraction=0.02,
            use_idle_filter=True,
            drop_gripper_oob=False,
        ),
        num_train_steps=100_000,
        fsdp_devices=8,
        batch_size=256,
        save_interval=500,
        log_interval=50,
        weight_loader=weight_loaders.WeightLoaderChoice(kind="paligemma"),
        # weight_loader=weight_loaders.WeightLoaderChoice(kind="checkpoint", params_path="gs://openpi-assets/checkpoints/pi0_base/params"),
        assets_base_dir="gs://v6_east1d/assets",
        checkpoint_base_dir="gs://v6_east1d/checkpoints",
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=1e-4,
            decay_steps=1_000_000,
            decay_lr=1e-4,
        ),
        # ema_decay=None,
        keep_period=10000,
    ),
    TrainConfig(
        name="pi_droid_cot_v5",
        do_val=True,
        model=pi_cot_config.PiCoTConfig(
            action_horizon=10,
            max_token_len=110,
            number_token_weight=1.0,
            pi05=True,
            discrete_state_input=True,
        ),
        data=RLDSDroidCoTDataConfig(
            repo_id="droid",
            rlds_data_dir="gs://v5_central1_a",
            language_action_dir="gs://v5_central1_a/droid-lang-actions",
            action_space=cot_rlds_dataset.DroidActionSpace.CARTESIAN_POSITION,
            base_config=CoTDataConfig(
                prompt_from_task=True,
            ),
            shuffle_buffer_size=250_000,
            assets=upstream_config.AssetsConfig(
                assets_dir="gs://v5_central1_a/assets/pi0_droid_cot_v4",
                asset_id="droid",
            ),
            summation_steps=15,
            sum_decimal="0f",
            left_pad=True,
            include_decimal_point=False,
            validation_mode="easy",
            vis_dataset=False,
            use_wrist_image=False,
            val_max_samples=60000,
            val_fraction=0.02,
            use_idle_filter=True,
            drop_gripper_oob=False,
        ),
        num_train_steps=100_000,
        fsdp_devices=8,
        batch_size=256,
        save_interval=500,
        log_interval=50,
        weight_loader=weight_loaders.WeightLoaderChoice(kind="paligemma"),
        # weight_loader=weight_loaders.WeightLoaderChoice(kind="checkpoint", params_path="gs://openpi-assets/checkpoints/pi0_base/params"),
        assets_base_dir="gs://v5_central1_a/assets",
        checkpoint_base_dir="gs://v5_central1_a/checkpoints",
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=1e-4,
            decay_steps=1_000_000,
            decay_lr=1e-4,
        ),
        # ema_decay=None,
        keep_period=10000,
    ),
    TrainConfig(
        name="pi_droid_cot_local",
        do_val=True,
        model=pi_cot_config.PiCoTConfig(
            action_horizon=10,
            max_token_len=110,
            pi05=False,
            discrete_state_input=False,
        ),
        data=RLDSDroidCoTDataConfig(
            repo_id="droid",
            rlds_data_dir="/n/fs/robot-data/data/",
            language_action_dir="/n/fs/robot-data/vlm-syn/droid-lang-actions",
            action_space=cot_rlds_dataset.DroidActionSpace.CARTESIAN_POSITION,
            base_config=CoTDataConfig(
                prompt_from_task=True,
            ),
            shuffle_buffer_size=250_000,
            assets=upstream_config.AssetsConfig(
                assets_dir="/n/fs/robot-data/pi0-cot/assets/pi0_droid_cot_v4",
                asset_id="droid",
            ),
            summation_steps=15,
            sum_decimal="0f",
            left_pad=True,
            include_decimal_point=False,
            validation_mode="easy",
            vis_dataset=False,
            use_wrist_image=False,
            val_max_samples=60000,
            val_fraction=0.02,
            use_idle_filter=True,
            drop_gripper_oob=False,
        ),
        num_train_steps=100_000,
        fsdp_devices=8,
        batch_size=1,
        save_interval=1000,
        log_interval=50,
        weight_loader=weight_loaders.WeightLoaderChoice(
            kind="checkpoint",
            params_path="/n/fs/robot-data/cache/openpi/openpi-assets/checkpoints/pi0_base/params",
        ),
        assets_base_dir="/n/fs/robot-data/pi0-cot/assets",
        checkpoint_base_dir="/n/fs/robot-data/pi0-cot/checkpoints",
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=1_000,
            peak_lr=1e-4,
            decay_steps=1_000_000,
            decay_lr=1e-4,
        ),
        # keep_period=20_000,
    ),
    *upstream_config._CONFIGS,
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
