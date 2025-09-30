"""
materialize.py

Factory class for initializing Open-X Embodiment dataset kwargs and other parameters; provides and exports functions for
clear control flow.
"""

from copy import deepcopy
import logging
from pathlib import Path
from typing import Any

from openpi_cot.dataloader.oxe_utils.configs import OXE_DATASET_CONFIGS
from openpi_cot.dataloader.oxe_utils.configs import ActionEncoding
from openpi_cot.dataloader.oxe_utils.data_utils import NormalizationType
from openpi_cot.dataloader.transforms import UNIFIED_STANDARDIZATION_TRANSFORMS


def make_oxe_dataset_kwargs(
    dataset_name: str,
    rlds_data_dir: Path,
    load_camera_views: tuple[str] = ("primary",),
    load_depth: bool = False,
    load_proprio: bool = True,
    load_language: bool = True,
    action_proprio_normalization_type: NormalizationType = NormalizationType.NORMAL,
) -> dict[str, Any]:
    """Generates config (kwargs) for given dataset from Open-X Embodiment."""
    dataset_kwargs = deepcopy(OXE_DATASET_CONFIGS[dataset_name])
    if dataset_kwargs["action_encoding"] not in [ActionEncoding.EEF_POS, ActionEncoding.EEF_R6]:
        raise ValueError(f"Cannot load `{dataset_name}`; only EEF_POS & EEF_R6 actions supported!")

    language_annotations = dataset_kwargs.get("language_annotations")
    if not language_annotations or language_annotations.lower() == "none":
        raise ValueError(f"Cannot load `{dataset_name}`; language annotations required!")

    robot_morphology = dataset_kwargs.get("robot_morphology", "")
    if robot_morphology.lower() == "bi-manual":
        raise ValueError(f"Cannot load `{dataset_name}`; bi-manual datasets are not supported!")

    has_suboptimal = dataset_kwargs.get("has_suboptimal")
    if isinstance(has_suboptimal, str):
        has_suboptimal = has_suboptimal.lower() == "yes"
    if has_suboptimal:
        logging.warning(f"Cannot load `{dataset_name}`; suboptimal datasets are not supported!")

    # [Contract] For EEF_POS & EEF_R6 actions, only the last action dimension (gripper) is absolute!
    # Normalize all action dimensions *except* the gripper
    if dataset_kwargs["action_encoding"] is ActionEncoding.EEF_POS:
        # dataset_kwargs["absolute_action_mask"] = [False] * 6 + [True]
        # dataset_kwargs["action_normalization_mask"] = [True] * 6 + [False]
        pass
    elif dataset_kwargs["action_encoding"] is ActionEncoding.EEF_R6:
        # dataset_kwargs["absolute_action_mask"] = [False] * 9 + [True]
        # dataset_kwargs["action_normalization_mask"] = [True] * 9 + [False]
        pass
    else:
        raise ValueError(f"Cannot load `{dataset_name}`; only EEF_POS & EEF_R6 actions supported!")

    # if dataset_kwargs["state_encoding"] not in [StateEncoding.POS_EULER, StateEncoding.POS_QUAT, StateEncoding.EEF_R6]:
    #     raise ValueError(f"Cannot load `{dataset_name}`; only POS_EULER, POS_QUAT & EEF_R6 state encodings supported!")

    dataset_kwargs["action_proprio_normalization_type"] = action_proprio_normalization_type

    # Adjust Loaded Camera Views
    if len(missing_keys := (set(load_camera_views) - set(dataset_kwargs["image_obs_keys"]))) > 0:
        raise ValueError(f"Cannot load `{dataset_name}`; missing camera views `{missing_keys}`")

    # Filter
    dataset_kwargs["image_obs_keys"] = {
        k: v for k, v in dataset_kwargs["image_obs_keys"].items() if k in load_camera_views
    }
    for k, v in dataset_kwargs["image_obs_keys"].items():
        if k == "primary":
            assert v is not None, f"primary image is required for {dataset_name}"
    dataset_kwargs["depth_obs_keys"] = {
        k: v for k, v in dataset_kwargs["depth_obs_keys"].items() if k in load_camera_views
    }

    # Eliminate Unnecessary Keys
    # dataset_kwargs.pop("state_encoding")
    # dataset_kwargs.pop("action_encoding")
    if not load_depth:
        dataset_kwargs.pop("depth_obs_keys")
    if not load_proprio:
        dataset_kwargs.pop("state_obs_keys")

    # Load Language
    if load_language:
        dataset_kwargs["language_key"] = "language_instruction"

    # Specify Standardization Transform
    # Use unified registry (superset), still supports all OXE datasets
    dataset_kwargs["standardize_fn"] = UNIFIED_STANDARDIZATION_TRANSFORMS[dataset_name]

    # Add any aux arguments
    if "aux_kwargs" in dataset_kwargs:
        dataset_kwargs.update(dataset_kwargs.pop("aux_kwargs"))

    return {"name": dataset_name, "data_dir": str(rlds_data_dir), **dataset_kwargs}


def get_oxe_dataset_kwargs_and_weights(
    rlds_data_dir: Path,
    mixture_spec: list[tuple[str, float]],
    load_camera_views: tuple[str] = ("primary",),
    load_depth: bool = False,
    load_proprio: bool = True,
    load_language: bool = True,
    action_proprio_normalization_type: NormalizationType = NormalizationType.NORMAL,
) -> tuple[dict[str, Any], list[float]]:
    """
    Generates dataset kwargs for a given dataset mix from the Open X-Embodiment dataset. The returned kwargs
    (per-dataset configs) and weights can be passed directly to `make_interleaved_dataset`.

    :param rlds_data_dir: Base directory containing RLDS/TFDS-formatted datasets (from Open-X)
    :param mixture_spec: List of (dataset_name, sampling_weight) from `oxe.mixtures.OXE_NAMED_MIXTURES`
    :param load_camera_views: Camera views to load; see `oxe.dataset_configs.py` for available views.
    :param load_depth: Load depth information in addition to camera RGB.
    :param load_proprio: Load proprioceptive state.
    :param load_language: Load language instructions.
    :param action_proprio_normalization_type: Normalization scheme to use for proprioceptive actions.

    return: Tuple of (per_dataset_kwargs, sampling_weights)
    """
    included_datasets, filtered_mixture_spec = set(), []
    for d_name, d_weight in mixture_spec:
        if d_name in included_datasets:
            logging.warning(f"Skipping Duplicate Dataset: `{(d_name, d_weight)}`")
            continue

        included_datasets.add(d_name)
        filtered_mixture_spec.append((d_name, d_weight))

    # Assemble Dataset Config (kwargs) and Weights
    per_dataset_kwargs, sampling_weights = [], []
    for d_name, d_weight in filtered_mixture_spec:
        per_dataset_kwargs.append(
            make_oxe_dataset_kwargs(
                d_name,
                rlds_data_dir,
                load_camera_views,
                load_depth,
                load_proprio,
                load_language,
                action_proprio_normalization_type,
            )
        )
        sampling_weights.append(d_weight)

    return per_dataset_kwargs, sampling_weights


def make_droid_dataset_kwargs(
    rlds_data_dir: Path,
    *,
    load_camera_views: tuple[str] = ("primary", "wrist"),
    action_proprio_normalization_type: NormalizationType = NormalizationType.NORMAL,
) -> dict[str, Any]:
    """Minimal DROID config to align with the unified pipeline.

    Note: DROID is not part of OXE_DATASET_CONFIGS. We construct the keys expected by
    the unified SingleOXECoTRldsDatasetRaw pipeline.
    """
    dataset_name = "droid"
    image_obs_keys = {"primary": "exterior_image_1_left", "wrist": "wrist_image_left"}
    # unify contract: state_obs_keys present for proprio path
    state_obs_keys = ["proprio"]
    standardize_fn = UNIFIED_STANDARDIZATION_TRANSFORMS.get(dataset_name)
    if standardize_fn is None:
        # Defer to dataset-specific restructure; unified OXE loader won't use this path
        standardize_fn = lambda x: x
    return {
        "name": dataset_name,
        "data_dir": str(rlds_data_dir),
        "image_obs_keys": image_obs_keys,
        "state_obs_keys": state_obs_keys,
        "language_key": "language_instruction",
        "standardize_fn": standardize_fn,
        "action_proprio_normalization_type": action_proprio_normalization_type,
        "state_encoding": ActionEncoding.EEF_POS,  # placeholder; per-dataset restructure handles precise encodings
        "action_encoding": ActionEncoding.EEF_POS,
        "control_frequency": 15,
        "is_absolute_action": True,
    }


def get_unified_dataset_kwargs_and_weights(
    rlds_data_dir: Path,
    mixture_spec: list[tuple[str, float]],
    *,
    include_droid: bool,
    load_camera_views: tuple[str] = ("primary", "wrist"),
    load_depth: bool = False,
    load_proprio: bool = True,
    load_language: bool = True,
    action_proprio_normalization_type: NormalizationType = NormalizationType.NORMAL,
) -> tuple[list[dict[str, Any]], list[float]]:
    """Unified factory that can include DROID alongside OXE datasets.

    Returns kwargs compatible with the unified loader.
    """
    if include_droid:
        # separate list to avoid name collisions and keep order
        oxe_kwargs, oxe_weights = get_oxe_dataset_kwargs_and_weights(
            rlds_data_dir,
            mixture_spec,
            load_camera_views,
            load_depth,
            load_proprio,
            load_language,
            action_proprio_normalization_type,
        )
        droid_kwargs = [
            make_droid_dataset_kwargs(
                rlds_data_dir,
                load_camera_views=load_camera_views,
                action_proprio_normalization_type=action_proprio_normalization_type,
            )
        ]
        droid_weights = [1.0]
        return [*oxe_kwargs, *droid_kwargs], [*oxe_weights, *droid_weights]
    return get_oxe_dataset_kwargs_and_weights(
        rlds_data_dir,
        mixture_spec,
        load_camera_views,
        load_depth,
        load_proprio,
        load_language,
        action_proprio_normalization_type,
    )
