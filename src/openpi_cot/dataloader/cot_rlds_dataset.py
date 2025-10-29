"""CoT RLDS Dataset - Main module with backward-compatible re-exports.

This module has been refactored into smaller, more maintainable components:
- specs.py: Data specifications and constants
- image_utils.py: Image processing utilities
- dataset_utils.py: Generic dataset utilities
- base_dataset.py: Base dataset class
- droid_dataset.py: DROID-specific implementation
- oxe_datasets.py: OXE dataset implementations
- dataset_mixer.py: Multi-dataset orchestration

All classes and functions are re-exported here for backward compatibility.
"""

# Re-export specifications
from openpi_cot.dataloader.specs import CoTRldsDatasetSpec

# Re-export image utilities
from openpi_cot.dataloader.image_utils import make_decode_images_fn

# Re-export dataset utilities
from openpi_cot.dataloader.dataset_utils import (
    dataset_size,
    gather_with_padding,
    prepare_batched_dataset,
    print_memory_usage,
)

# Re-export base dataset class
from openpi_cot.dataloader.base_dataset import _SingleCoTDataset

# Re-export DROID dataset
from openpi_cot.dataloader.droid_dataset import DroidCoTDataset

# Re-export OXE datasets
from openpi_cot.dataloader.oxe_datasets import (
    _DobbeCoTDataset,
    _LiberoCoTDataset,
    _SampleR1LiteCoTDataset,
    _SingleOXECoTDataset,
    PlanningDataset,
)

# Re-export dataset mixer
from openpi_cot.dataloader.dataset_mixer import OXECoTDatasets

# Define public API
__all__ = [
    # Specifications
    "CoTRldsDatasetSpec",
    # Image utilities
    "make_decode_images_fn",
    # Dataset utilities
    "dataset_size",
    "gather_with_padding",
    "prepare_batched_dataset",
    "print_memory_usage",
    # Base dataset
    "_SingleCoTDataset",
    # DROID dataset
    "DroidCoTDataset",
    # OXE datasets
    "_SingleOXECoTDataset",
    "_DobbeCoTDataset",
    "_LiberoCoTDataset",
    "PlanningDataset",
    "_SampleR1LiteCoTDataset",
    # Dataset mixer
    "OXECoTDatasets",
]
