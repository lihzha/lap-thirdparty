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

from openpi_cot.dataloader.helpers import ActionEncoding
from openpi_cot.dataloader.helpers import StateEncoding
import openpi_cot.models.adapters.model_adapter as _model_adapter
from openpi_cot.models.adapters.tokenizer_adapter import PaligemmaCoTTokenizer
import openpi_cot.models.pi_cot_config as pi_cot_config
import openpi_cot.policies.cot_policy as cot_policy
import openpi_cot.policies.libero_policy as libero_policy
import openpi_cot.policies.tiger_policy as tiger_policy
import openpi_cot.policies.vqa_policy as vqa_policy
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


def build_droid_cfg(
    tpu_version: str,
    fsdp_devices: int,
    batch_size: int,
    droid_dataset_name: Literal["droid", "droid_subset"] = "droid",
    droid_rlds_data_dir: str | None = None,
):
    language_action_dir = None
    ckpt_base_dir = None
    match tpu_version:
        case "v4":
            rlds_data_dir = "gs://pi0-cot/OXE"
        case "v5":
            rlds_data_dir = "gs://v5_central1_a/OXE"
        case "v6":
            rlds_data_dir = "gs://v6_east1d/OXE"
        case "local":
            rlds_data_dir = "/n/fs/robot-data/data/"
            language_action_dir = "/n/fs/robot-data/vlm-syn/droid-lang-actions"
            ckpt_base_dir = "/n/fs/robot-data/pi0-cot/checkpoints"
        case _:
            raise ValueError(f"Invalid TPU version: {tpu_version}")

    if language_action_dir is None:
        language_action_dir = rlds_data_dir.replace("OXE", "droid-base-lang-actions")
    if ckpt_base_dir is None:
        ckpt_base_dir = rlds_data_dir.replace("OXE", "checkpoints")

    return TrainConfig(
        name=f"pi_droid_cot_{tpu_version}",
        data=RLDSCoTDataConfig(
            repo_id="droid",
            asset_id="droid",
            dataset_type="droid",
            rlds_data_dir=rlds_data_dir,
            language_action_dir=language_action_dir,
            droid_dataset_name=droid_dataset_name,
            droid_rlds_data_dir=droid_rlds_data_dir,
        ),
        fsdp_devices=fsdp_devices,
        batch_size=batch_size,
        checkpoint_base_dir=ckpt_base_dir,
        weight_loader=weight_loaders.WeightLoaderChoice(kind="paligemma"),
    )


@dataclasses.dataclass(frozen=True)
class CoTDataConfig(upstream_config.DataConfig):
    # TODO: remove the cot argument
    cot: bool = False
    shuffle_buffer_size: int = 250_000
    # Optional cap on number of unique flattened samples for overfitting tests
    max_samples: int | None = None
    # Validation controls for RLDS-CoT dataset splitting/visualization
    val_max_samples: int | None = 60000
    val_fraction: float | None = 0.02
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

    def __call__(self, model_config: _model.BaseModelConfig) -> upstream_transforms.Group:
        if model_config.model_type == ModelType.PI_COT:
            assert isinstance(model_config, pi_cot_config.PiCoTConfig)
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
                outputs=[
                    DetokenizeReasoning(
                        PaligemmaCoTTokenizer(
                            model_config.max_token_len,
                            prompt_format=self.prompt_format,
                            tokenizer_type=self.tokenizer_type,
                        )
                    )
                ],
            )
        return super().__call__(model_config)


@dataclasses.dataclass(frozen=True)
class TigerModelTransformFactory(upstream_config.ModelTransformFactory):
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

    def __call__(self, model_config: _model.BaseModelConfig) -> upstream_transforms.Group:
        if model_config.model_type == ModelType.PI_COT:
            assert isinstance(model_config, pi_cot_config.PiCoTConfig)
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
                outputs=[],
            )
        return super().__call__(model_config)


@dataclasses.dataclass(frozen=True)
class RLDSCoTDataConfig(CoTDataConfig, upstream_config.DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> CoTDataConfig:
        cot_fields = CoTDataConfig.__dataclass_fields__.keys()
        data = {k: getattr(self, k) for k in cot_fields}
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        data.update(
            repo_id=repo_id,
            asset_id=asset_id,
            # norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            norm_stats=None,  # Note: Normalization is handled on dataset level
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
        ## Note: Repack is handled on dataset level
        # repack_dict = {
        #     # always name base image as "exterior_image_1_left", though it should come from the camera which language action is annotated.
        #     "observation/exterior_image_1_left": "observation/image",
        #     "observation/state": "observation/state",
        #     "actions": "actions",
        #     "prompt": "prompt",
        #     "language_actions": "language_actions",
        # }
        # if base_cfg.vis_dataset:
        #     repack_dict["camera_intrinsics"] = "camera_intrinsics"
        #     repack_dict["camera_extrinsics"] = "camera_extrinsics"
        #     repack_dict["observation/cartesian_position_window"] = "observation/cartesian_position_window"
        # if base_cfg.use_wrist_image:
        #     repack_dict["observation/wrist_image_left"] = "observation/wrist_image"
        # repack_transform = upstream_transforms.Group(inputs=[upstream_transforms.RepackTransform(repack_dict)])

        data_transforms = upstream_transforms.Group(
            inputs=[
                cot_policy.CoTInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                    wrist_image_dropout_prob=base_cfg.wrist_image_dropout_prob,
                    action_encoding=base_cfg.action_encoding,
                    language_action_config=cot_policy.get_language_action_config(base_cfg.language_action_config_name),
                    # Add prediction fields
                    enable_prediction_training=model_config.enable_prediction_training,
                    prediction_prompt=base_cfg.prediction_prompt,
                )
            ],
            outputs=[cot_policy.CoTOutputs(decoding_schema=base_cfg.decoding_schema)],
        )

        # assert base_cfg.action_space == cot_rlds_dataset.DroidActionSpace.CARTESIAN_POSITION
        # TODO: Data loader returns absolute joint position actions -- convert to delta actions for training. confirm with oxe
        # delta_action_mask = upstream_transforms.make_bool_mask(6, -1)
        # data_transforms = data_transforms.push(
        #     inputs=[upstream_transforms.DeltaActions(delta_action_mask)],
        #     # outputs=[upstream_transforms.AbsoluteActions(delta_action_mask)],
        # )

        model_transforms = ModelTransformFactory(
            prompt_format=model_config.prompt_format,
            prediction_prompt=base_cfg.prediction_prompt,
            tokenizer_type="gemma3" if "gemma3" in model_config.paligemma_variant else "paligemma",
        )(model_config)

        return dataclasses.replace(
            base_cfg,
            # repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            use_quantile_norm=model_config.model_type == ModelType.PI0_FAST,
        )


@dataclasses.dataclass(frozen=True)
class VQADataConfig(RLDSCoTDataConfig):
    """
    Config for VQA evaluation.
    """

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> CoTDataConfig:
        base_cfg = self.create_base_config(assets_dirs, model_config)

        data_transforms = upstream_transforms.Group(
            inputs=[
                vqa_policy.VQAInputs(
                    model_type=model_config.model_type,
                )
            ],
            outputs=[vqa_policy.VQAOutputs()],
        )

        model_transforms = ModelTransformFactory(
            prompt_format=model_config.prompt_format,
            prediction_prompt=base_cfg.prediction_prompt,
            tokenizer_type="gemma3" if "gemma3" in model_config.paligemma_variant else "paligemma",
        )(model_config)

        return dataclasses.replace(
            base_cfg,
            # repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            use_quantile_norm=model_config.model_type == ModelType.PI0_FAST,
        )


@dataclasses.dataclass(frozen=True)
class LiberoDataConfig(CoTDataConfig, upstream_config.DataConfigFactory):
    """
    Config for training on DROID, using RLDS data format (for efficient training on larger datasets).
    """

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> CoTDataConfig:
        cot_fields = CoTDataConfig.__dataclass_fields__.keys()
        data = {k: getattr(self, k) for k in cot_fields}
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        data.update(
            repo_id=repo_id,
            asset_id=asset_id,
            # norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            norm_stats=None,  # Note: Normalization is handled on dataset level
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

        data_transforms = upstream_transforms.Group(
            inputs=[
                libero_policy.LiberoInputs(
                    model_type=model_config.model_type,
                )
            ],
            outputs=[libero_policy.LiberoOutputs()],
        )

        # assert base_cfg.action_space == cot_rlds_dataset.DroidActionSpace.CARTESIAN_POSITION
        # TODO: Data loader returns absolute joint position actions -- convert to delta actions for training. confirm with oxe
        # delta_action_mask = upstream_transforms.make_bool_mask(6, -1)
        # data_transforms = data_transforms.push(
        #     inputs=[upstream_transforms.DeltaActions(delta_action_mask)],
        #     # outputs=[upstream_transforms.AbsoluteActions(delta_action_mask)],
        # )

        model_transforms = ModelTransformFactory(
            prediction_prompt=base_cfg.prediction_prompt,
            tokenizer_type="gemma3" if "gemma3" in model_config.paligemma_variant else "paligemma",
        )(model_config)

        return dataclasses.replace(
            base_cfg,
            # repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            use_quantile_norm=model_config.model_type == ModelType.PI0_FAST,
        )


@dataclasses.dataclass(frozen=True)
class TigerDataConfig(CoTDataConfig, upstream_config.DataConfigFactory):
    """
    Config for training on Tiger demos, using LeRobot format.
    """

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> CoTDataConfig:
        cot_fields = CoTDataConfig.__dataclass_fields__.keys()
        data = {k: getattr(self, k) for k in cot_fields}
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        asset_id = self.assets.asset_id or repo_id
        data.update(
            repo_id=repo_id,
            asset_id=asset_id,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )
        return CoTDataConfig(**data)

    def _load_norm_stats(
        self, assets_dir: epath.Path, asset_id: str | None
    ) -> dict[str, upstream_transforms.NormStats] | None:
        """Load normalization statistics if available."""
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize_adapter.load(maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.warning(
                f"Norm stats not found in {data_assets_dir}. "
                f"Run 'python scripts/compute_norm_stats.py --config-name <config_name>' to compute them."
            )
        return None

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> CoTDataConfig:
        base_cfg = self.create_base_config(assets_dirs, model_config)

        data_transforms = upstream_transforms.Group(
            inputs=[
                tiger_policy.TigerInputs(
                    model_type=model_config.model_type,
                )
            ],
            outputs=[tiger_policy.TigerOutputs()],
        )

        model_transforms = TigerModelTransformFactory(
            prediction_prompt=base_cfg.prediction_prompt,
            prompt_format=model_config.prompt_format,
            tokenizer_type="gemma3" if "gemma3" in model_config.paligemma_variant else "paligemma",
        )(model_config)

        return dataclasses.replace(
            base_cfg,
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
    build_droid_cfg("v4", fsdp_devices=4, batch_size=256),
    build_droid_cfg("v5", fsdp_devices=8, batch_size=256),
    build_droid_cfg("v6", fsdp_devices=8, batch_size=256),
    build_droid_cfg("local", fsdp_devices=1, batch_size=4),
    TrainConfig(
        name="pi_combined_cot_v4",
        data=RLDSCoTDataConfig(
            repo_id="combined",
            asset_id="combined",
            dataset_type="combined",
            droid_dataset_name="droid",
            rlds_data_dir="gs://pi0-cot/OXE",
            language_action_dir="gs://pi0-cot/droid-base-lang-actions",
            data_mix="oxe_pi_magic_soup",
            shuffle_buffer_size=300_000,
        ),
        fsdp_devices=4,
        batch_size=256,
        weight_loader=weight_loaders.WeightLoaderChoice(
            kind="checkpoint", params_path="gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        checkpoint_base_dir="gs://pi0-cot/checkpoints",
        save_interval=500,
        keep_period=10000,
        resume=True,
    ),
    TrainConfig(
        name="pi_combined_cot_v6",
        data=RLDSCoTDataConfig(
            repo_id="combined",
            asset_id="combined",
            dataset_type="combined",
            droid_dataset_name="droid",
            rlds_data_dir="gs://v6_east1d/OXE",
            language_action_dir="gs://v6_east1d/droid-base-lang-actions",
            data_mix="oxe_pi_magic_soup_with_other_states_with_bimanual",
            shuffle_buffer_size=400_000,
        ),
        fsdp_devices=4,
        batch_size=256,
        weight_loader=weight_loaders.WeightLoaderChoice(
            kind="checkpoint", params_path="gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        checkpoint_base_dir="gs://v6_east1d/checkpoints",
        save_interval=500,
        keep_period=5000,
        resume=True,
    ),
    TrainConfig(
        name="pi_combined_cot_v6europe",
        data=RLDSCoTDataConfig(
            repo_id="combined",
            asset_id="combined",
            dataset_type="combined",
            droid_dataset_name="droid",
            rlds_data_dir="gs://v6_europe_west4a/OXE",
            language_action_dir="gs://v6_europe_west4a/droid-base-lang-actions",
            data_mix="oxe_pi_magic_soup_with_other_states_with_bimanual",
            shuffle_buffer_size=400_000,
        ),
        fsdp_devices=4,
        batch_size=256,
        weight_loader=weight_loaders.WeightLoaderChoice(
            kind="checkpoint", params_path="gs://openpi-assets/checkpoints/pi05_base/params"
        ),
        checkpoint_base_dir="gs://v6_europe_west4a/checkpoints",
        save_interval=500,
        keep_period=5000,
        resume=True,
    ),
    TrainConfig(
        name="pi_combined_cot_local",
        data=RLDSCoTDataConfig(
            repo_id="combined",
            asset_id="combined",
            dataset_type="combined",
            rlds_data_dir="/n/fs/vla-mi/datasets/OXE",
            language_action_dir="/n/fs/robot-data/vlm-syn/droid-lang-actions",
            data_mix="oxe_pi_magic_soup",
            shuffle_buffer_size=400_000,
        ),
        fsdp_devices=4,
        batch_size=256,
        weight_loader=weight_loaders.WeightLoaderChoice(kind="paligemma"),
        checkpoint_base_dir="/n/fs/robot-data/pi0-cot/checkpoints",
    ),
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
        name="pi05_libero_eval",
        model=pi_cot_config.PiCoTConfig(
            action_horizon=10,
            max_token_len=180,
            pi05=True,
            discrete_state_input=True,
        ),
        data=LiberoDataConfig(
            repo_id="droid",
            asset_id="droid",
            dataset_type="droid",
        ),
    ),
    TrainConfig(
        name="pi05_tiger_finetune_local",
        model=pi_cot_config.PiCoTConfig(
            action_horizon=10,
            max_token_len=180,
            pi05=True,
            discrete_state_input=True,
            enable_action_training=True,
            enable_langact_training=False,
            prompt_format="pi05",
        ),
        data=TigerDataConfig(
            repo_id="your_hf_username/tiger_demos",
            asset_id="tiger",
            dataset_type="droid",  # Use droid type for compatibility
            rlds_data_dir=None,  # LeRobot dataset will be loaded from HF_LEROBOT_HOME
        ),
        fsdp_devices=1,
        batch_size=4,
        num_train_steps=10000,
        save_interval=500,
        log_interval=50,
        checkpoint_base_dir="/n/fs/robot-data/pi0-cot/checkpoints",
        weight_loader=weight_loaders.WeightLoaderChoice(
            kind="checkpoint",
            params_path="gs://openpi-assets/checkpoints/pi05_base/params",
        ),
    ),
    TrainConfig(
        name="pi05_tiger_finetune_local_low_mem",
        # Here is an example of loading a pi0 model for LoRA fine-tuning.
        model=pi_cot_config.PiCoTConfig(
            paligemma_variant="gemma_2b_lora",
            action_expert_variant="gemma_300m_lora",
            action_horizon=10,
            max_token_len=180,
            pi05=True,
            discrete_state_input=True,
            enable_action_training=True,
            enable_langact_training=False,
            prompt_format="pi05",
        ),
        data=TigerDataConfig(
            repo_id="your_hf_username/tiger_demos",
            asset_id="tiger",
            dataset_type="droid",  # Use droid type for compatibility
            rlds_data_dir=None,  # LeRobot dataset will be loaded from HF_LEROBOT_HOME
            assets=upstream_config.AssetsConfig(
                # Important: reuse the original DROID norm stats during fine-tuning!
                assets_dir="/n/fs/robot-data/openpi-cot/assets/pi05_tiger_finetune_local/your_hf_username",
                asset_id="tiger_demos",
            ),
        ),
        weight_loader=weight_loaders.WeightLoaderChoice(
            kind="checkpoint",
            params_path="gs://openpi-assets/checkpoints/pi05_base/params",
        ),
        num_train_steps=30_000,
        # The freeze filter defines which parameters should be frozen during training.
        # We have a convenience function in the model config that returns the default freeze filter
        # for the given model config for LoRA finetuning. Just make sure it matches the model config
        # you chose above.
        freeze_filter=pi_cot_config.PiCoTConfig(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        # Turn off EMA for LoRA finetuning.
        ema_decay=None,
        fsdp_devices=4,
        batch_size=32,
        save_interval=1000,
        log_interval=100,
        checkpoint_base_dir="/n/fs/robot-data/pi0-cot/checkpoints",
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
            max_token_len=100,
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
        weight_loader=weight_loaders.WeightLoaderChoice(kind="gemma3", params_path="gs://pi0-cot/cache/gemma3-4b"),
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
        weight_loader=weight_loaders.WeightLoaderChoice(kind="gemma3", params_path="/home/ajhancock/Desktop/openpi-cot/src/openpi_cot/ckpts/gemma3-4b-it"),
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


def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
