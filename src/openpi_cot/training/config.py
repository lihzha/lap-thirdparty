"""See _CONFIGS for the list of available configs."""

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

import openpi_cot.dataloader.cot_rlds_dataset as cot_rlds_dataset
import openpi_cot.models.adapters.model_adapter as _model_adapter
from openpi_cot.models.adapters.tokenizer_adapter import PaligemmaCoTTokenizer
import openpi_cot.models.pi_cot_config as pi_cot_config
import openpi_cot.policies.cot_policy as cot_policy
import openpi_cot.shared.adapters.normalize_adapter as _normalize_adapter
from openpi_cot.shared.download import maybe_download
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


def build_picot_model(
    *,
    action_horizon: int = 10,
    max_token_len: int = 110,
    number_token_weight: float | None = 1.0,
    pi05: bool = True,
    discrete_state_input: bool = True,
) -> _model.BaseModelConfig:
    """Convenience helper for common PiCoT model instantiations."""
    return pi_cot_config.PiCoTConfig(
        action_horizon=action_horizon,
        max_token_len=max_token_len,
        number_token_weight=number_token_weight,
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


def build_droid_cot_data(
    *,
    rlds_data_dir: str,
    language_action_dir: str,
    assets_dir: str,
    asset_id: str = "droid",
    prompt_from_task: bool = True,
    shuffle_buffer_size: int = 250_000,
    summation_steps: int = 15,
    sum_decimal: str = "0f",
    left_pad: bool = True,
    include_decimal_point: bool = False,
    validation_mode: str = "easy",
    vis_dataset: bool = False,
    use_wrist_image: bool = False,
    val_max_samples: int | None = 60000,
    val_fraction: float | None = 0.02,
    use_idle_filter: bool = True,
    drop_gripper_oob: bool = False,
) -> "RLDSCoTDataConfig":
    """Helper to build a standard DROID CoT RLDS data config."""
    return RLDSCoTDataConfig(
        repo_id="droid",
        rlds_data_dir=rlds_data_dir,
        language_action_dir=language_action_dir,
        action_space=cot_rlds_dataset.DroidActionSpace.CARTESIAN_POSITION,
        base_config=CoTDataConfig(
            prompt_from_task=prompt_from_task,
        ),
        shuffle_buffer_size=shuffle_buffer_size,
        assets=upstream_config.AssetsConfig(
            assets_dir=assets_dir,
            asset_id=asset_id,
        ),
        summation_steps=summation_steps,
        sum_decimal=sum_decimal,
        left_pad=left_pad,
        include_decimal_point=include_decimal_point,
        validation_mode=validation_mode,
        vis_dataset=vis_dataset,
        use_wrist_image=use_wrist_image,
        val_max_samples=val_max_samples,
        val_fraction=val_fraction,
        use_idle_filter=use_idle_filter,
        drop_gripper_oob=drop_gripper_oob,
        dataset_type="droid",
    )


def build_oxe_cot_data(
    *,
    rlds_data_dir: str,
    data_mix: str,
    assets_dir: str,
    asset_id: str = "oxe",
    shuffle_buffer_size: int = 250_000,
    sum_decimal: str = "0f",
    left_pad: bool = True,
    include_decimal_point: bool = False,
    val_max_samples: int | None = 60000,
    val_fraction: float | None = 0.02,
    validation_mode: str = "easy",
    vis_dataset: bool = False,
    use_wrist_image: bool = False,
) -> "RLDSCoTDataConfig":
    """Helper to build an OXE CoT RLDS data config."""
    return RLDSCoTDataConfig(
        repo_id="oxe",
        rlds_data_dir=rlds_data_dir,
        data_mix=data_mix,
        shuffle_buffer_size=shuffle_buffer_size,
        max_samples=None,
        left_pad=left_pad,
        include_decimal_point=include_decimal_point,
        sum_decimal=sum_decimal,
        val_max_samples=val_max_samples,
        val_fraction=val_fraction,
        validation_mode=validation_mode,
        vis_dataset=vis_dataset,
        use_wrist_image=use_wrist_image,
        assets=upstream_config.AssetsConfig(
            assets_dir=assets_dir,
            asset_id=asset_id,
        ),
        dataset_type="oxe",
    )


@dataclasses.dataclass(frozen=True)
class CoTDataConfig(upstream_config.DataConfig):
    # TODO: remove the cot argument
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
    # If true, will drop samples where projected gripper is outside the resized image bounds.
    drop_gripper_oob: bool = False

    # Dataset selection and OXE/Combined-specific knobs
    # One of {"droid", "oxe", "combined"}; used by the RLDS loader switch.
    dataset_type: Literal["droid", "oxe", "combined"] = "droid"
    # OXE fields (used when dataset_type == "oxe" or "combined")
    data_mix: str | None = "oxe_pi_magic_soup"
    # Combined-only: weight for DROID when interleaving with OXE
    droid_weight: float = 2.0


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(upstream_config.ModelTransformFactory):
    """Creates model transforms for standard pi0 models."""

    left_pad: bool = True
    include_decimal_point: bool = True

    def __call__(self, model_config: _model.BaseModelConfig) -> upstream_transforms.Group:
        if model_config.model_type == ModelType.PI_COT:
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
        return super().__call__(model_config)


@dataclasses.dataclass(frozen=True)
class RLDSCoTDataConfig(CoTDataConfig, upstream_config.DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    dataset_type: Literal["droid", "oxe", "combined"] = "droid"

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> CoTDataConfig:
        cot_fields = CoTDataConfig.__dataclass_fields__.keys()
        data = {k: getattr(self, k) for k in cot_fields}
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        data.update(
            repo_id=repo_id,
            asset_id=asset_id,
            # norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            norm_stats=None,
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )
        return CoTDataConfig(**data)

    def _load_norm_stats(
        self, assets_dir: epath.Path, asset_id: str | None
    ) -> dict[str, upstream_transforms.NormStats] | None:
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

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> CoTDataConfig:
        base_cfg = self.create_base_config(assets_dirs, model_config)
        repack_dict = {
            # always name base image as "exterior_image_1_left", though it should come from the camera which language action is annotated.
            "observation/exterior_image_1_left": "observation/image",
            "observation/state": "observation/state",
            "actions": "actions",
            "prompt": "prompt",
            "language_actions": "language_actions",
        }
        if base_cfg.vis_dataset:
            repack_dict["camera_intrinsics"] = "camera_intrinsics"
            repack_dict["camera_extrinsics"] = "camera_extrinsics"
            repack_dict["observation/cartesian_position_window"] = "observation/cartesian_position_window"
        if base_cfg.use_wrist_image:
            repack_dict["observation/wrist_image_left"] = "observation/wrist_image"
        repack_transform = upstream_transforms.Group(inputs=[upstream_transforms.RepackTransform(repack_dict)])

        data_transforms = upstream_transforms.Group(
            inputs=[
                cot_policy.CoTInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                    sum_decimal=base_cfg.sum_decimal,
                    wrist_image_dropout_prob=base_cfg.wrist_image_dropout_prob,
                )
            ],
            outputs=[cot_policy.CoTOutputs()],
        )

        # assert base_cfg.action_space == cot_rlds_dataset.DroidActionSpace.CARTESIAN_POSITION
        # TODO: Data loader returns absolute joint position actions -- convert to delta actions for training. confirm with oxe
        # delta_action_mask = upstream_transforms.make_bool_mask(6, -1)
        # data_transforms = data_transforms.push(
        #     inputs=[upstream_transforms.DeltaActions(delta_action_mask)],
        #     # outputs=[upstream_transforms.AbsoluteActions(delta_action_mask)],
        # )

        model_transforms = ModelTransformFactory(
            left_pad=base_cfg.left_pad, include_decimal_point=base_cfg.include_decimal_point
        )(model_config)

        assert base_cfg.rlds_data_dir is not None, "Need to set rlds data dir for RLDS data loader."

        return dataclasses.replace(
            base_cfg,
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            use_quantile_norm=model_config.model_type == ModelType.PI0_FAST,
        )


@dataclasses.dataclass(frozen=True)
class TrainConfig(upstream_config.TrainConfig):
    # Overide
    project_name: str = "openpi-cot"
    weight_loader: weight_loaders.WeightLoaderChoice = dataclasses.field(
        default_factory=weight_loaders.WeightLoaderChoice
    )
    # New field
    do_val: bool = False

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
        name="pi_droid_cot_v4",
        do_val=True,
        model=build_picot_model(pi05=True, discrete_state_input=True),
        data=build_droid_cot_data(
            rlds_data_dir="gs://pi0-cot",
            language_action_dir="gs://pi0-cot/droid-lang-actions",
            assets_dir="gs://pi0-cot/assets/pi0_droid_cot_v4",
        ),
        num_train_steps=100_000,
        fsdp_devices=4,
        batch_size=256,
        log_interval=50,
        save_interval=500,
        weight_loader=weight_loaders.WeightLoaderChoice(kind="paligemma"),
        keep_period=10000,
        assets_base_dir="gs://pi0-cot/assets",
        checkpoint_base_dir="gs://pi0-cot/checkpoints",
        lr_schedule=build_cosine_lr(),
    ),
    TrainConfig(
        name="pi_droid_cot_v6",
        do_val=True,
        model=build_picot_model(pi05=True, discrete_state_input=True),
        data=build_droid_cot_data(
            rlds_data_dir="gs://v6_east1d",
            language_action_dir="gs://v6_east1d/droid-lang-actions",
            assets_dir="gs://v6_east1d/assets/pi0_droid_cot_v4",
        ),
        num_train_steps=100_000,
        fsdp_devices=8,
        batch_size=256,
        save_interval=500,
        log_interval=50,
        weight_loader=weight_loaders.WeightLoaderChoice(kind="paligemma"),
        assets_base_dir="gs://v6_east1d/assets",
        checkpoint_base_dir="gs://v6_east1d/checkpoints",
        lr_schedule=build_cosine_lr(),
        keep_period=10000,
    ),
    TrainConfig(
        name="pi_droid_cot_v5",
        do_val=True,
        model=build_picot_model(pi05=True, discrete_state_input=True),
        data=build_droid_cot_data(
            rlds_data_dir="gs://v5_central1_a",
            language_action_dir="gs://v5_central1_a/droid-lang-actions",
            assets_dir="gs://v5_central1_a/assets/pi0_droid_cot_v4",
        ),
        num_train_steps=100_000,
        fsdp_devices=8,
        batch_size=256,
        save_interval=500,
        log_interval=50,
        weight_loader=weight_loaders.WeightLoaderChoice(kind="paligemma"),
        assets_base_dir="gs://v5_central1_a/assets",
        checkpoint_base_dir="gs://v5_central1_a/checkpoints",
        lr_schedule=build_cosine_lr(),
        keep_period=10000,
    ),
    TrainConfig(
        name="pi_droid_cot_local",
        do_val=True,
        model=build_picot_model(pi05=False, discrete_state_input=False),
        data=build_droid_cot_data(
            rlds_data_dir="/n/fs/robot-data/data/",
            language_action_dir="/n/fs/robot-data/vlm-syn/droid-lang-actions",
            assets_dir="/n/fs/robot-data/pi0-cot/assets/pi0_droid_cot_v4",
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
        lr_schedule=build_cosine_lr(),
        # keep_period=20_000,
    ),
    TrainConfig(
        name="pi_oxe_cot_v4",
        do_val=True,
        model=build_picot_model(pi05=True, discrete_state_input=True),
        data=build_oxe_cot_data(
            rlds_data_dir="gs://pi0-cot/OXE",
            data_mix="oxe_pi_magic_soup",
            assets_dir="gs://pi0-cot/assets/pi0_oxe_cot_v4",
        ),
        num_train_steps=100_000,
        fsdp_devices=4,
        batch_size=256,
        save_interval=500,
        log_interval=50,
        weight_loader=weight_loaders.WeightLoaderChoice(kind="paligemma"),
        assets_base_dir="gs://pi0-cot/assets",
        checkpoint_base_dir="gs://pi0-cot/checkpoints",
        lr_schedule=build_cosine_lr(),
        keep_period=10000,
    ),
    *upstream_config._CONFIGS,  # noqa: SLF001
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
