# DataLoader State Checkpointing

This document explains how to save and load the state of `CoTRLDSDataLoader` to enable resuming training from a specific point in the dataset.

## Overview

The dataloader checkpointing feature allows you to:
- Save the exact position in the dataset during training
- Resume training from that exact position without repeating or skipping data
- Track the number of batches processed

This is implemented using TensorFlow's `tf.train.Checkpoint` API to save the underlying dataset iterator state.

## Requirements

To use checkpointing, you **must** create the dataloader with `persistent_iterator=True`:

```python
from openpi_cot.dataloader.cot_data_loader import create_data_loader

# Create dataloader with persistent iterator support
dataloader = create_data_loader(
    config=your_config,
    persistent_iterator=True,  # Required for checkpointing!
    shuffle=True,
    split="train",
)
```

## Basic Usage

### Saving Dataloader State

```python
# During training, save the dataloader state periodically
for batch_idx, (observations, actions) in enumerate(dataloader):
    # ... training code ...

    # Save checkpoint every N batches
    if batch_idx % 1000 == 0:
        checkpoint_path = dataloader.save_dataloader_state(
            checkpoint_dir="./checkpoints/dataloader"
        )
        print(f"Saved dataloader checkpoint to {checkpoint_path}")
```

### Loading Dataloader State

```python
# When resuming training, load the checkpoint before starting iteration
dataloader = create_data_loader(
    config=your_config,
    persistent_iterator=True,  # Required!
    shuffle=True,
    split="train",
)

# Load the saved state
batches_seen = dataloader.load_dataloader_state(
    checkpoint_dir="./checkpoints/dataloader"
)
print(f"Resumed from batch {batches_seen}")

# Continue training from where you left off
for batch_idx, (observations, actions) in enumerate(dataloader):
    # ... training code ...
    pass
```

## Complete Example

```python
import os
from openpi_cot.dataloader.cot_data_loader import create_data_loader

def train_with_checkpoint_resume(config, checkpoint_dir="./checkpoints"):
    # Create dataloader with persistent iterator
    dataloader = create_data_loader(
        config=config,
        persistent_iterator=True,
        shuffle=True,
        split="train",
    )

    # Try to resume from checkpoint
    dataloader_ckpt_dir = os.path.join(checkpoint_dir, "dataloader")
    start_batch = 0

    if os.path.exists(dataloader_ckpt_dir):
        try:
            start_batch = dataloader.load_dataloader_state(dataloader_ckpt_dir)
            print(f"Resumed from batch {start_batch}")
        except ValueError as e:
            print(f"No checkpoint found, starting from scratch: {e}")

    # Training loop
    for batch_idx, (observations, actions) in enumerate(dataloader):
        # Your training code here
        loss = train_step(observations, actions)

        # Save checkpoint every 1000 batches
        if batch_idx % 1000 == 0:
            dataloader.save_dataloader_state(dataloader_ckpt_dir)
            print(f"Batch {batch_idx}: loss={loss:.4f}, checkpoint saved")
```

## API Reference

### `CoTRLDSDataLoader.save_dataloader_state(checkpoint_dir: str) -> str`

Save the current state of the dataloader iterator.

**Parameters:**
- `checkpoint_dir` (str): Directory where checkpoint files will be saved

**Returns:**
- str: Path to the saved checkpoint file

**Raises:**
- `ValueError`: If `persistent_iterator=False` when creating the dataloader

**Example:**
```python
save_path = dataloader.save_dataloader_state("./checkpoints/dataloader")
```

### `CoTRLDSDataLoader.load_dataloader_state(checkpoint_dir: str) -> int`

Load the dataloader state from a checkpoint.

**Parameters:**
- `checkpoint_dir` (str): Directory containing checkpoint files

**Returns:**
- int: Number of batches that were processed when the checkpoint was saved

**Raises:**
- `ValueError`: If no checkpoint found or if `persistent_iterator=False`

**Example:**
```python
batches_seen = dataloader.load_dataloader_state("./checkpoints/dataloader")
```

### `CoTRLDSDataLoader.get_batches_seen() -> int`

Get the total number of batches seen by this dataloader.

**Returns:**
- int: Count of batches processed

**Example:**
```python
total_batches = dataloader.get_batches_seen()
```

## Implementation Details

### Architecture

The checkpointing system works at multiple levels:

1. **_SingleCoTDataset**: Provides `create_checkpointable_iterator()` method that returns a TensorFlow iterator
2. **IterableTransformedDataset**: Maintains a persistent TensorFlow iterator when `persistent_iterator=True`
3. **CoTRLDSDataLoader**: Exposes high-level `save_dataloader_state()` and `load_dataloader_state()` methods

### What Gets Saved

The checkpoint includes:
- **Iterator position**: Exact position in the dataset
- **Batch counter**: Number of batches processed
- **Shuffle state**: Random state for shuffled datasets (if applicable)
- **Buffer state**: Prefetch and shuffle buffers

### Limitations

1. **External state**: Cannot checkpoint iterators that use `tf.py_function` or other external state
2. **File size**: Checkpoints may be large (hundreds of MB) due to buffered data from shuffle/prefetch operations
3. **Persistent iterator required**: Must set `persistent_iterator=True` when creating the dataloader
4. **Cross-version compatibility**: Checkpoint format may change between TensorFlow versions

### Performance Considerations

- Saving checkpoints involves writing large amounts of buffered data to disk
- Loading checkpoints reads this data back into memory
- Consider checkpoint frequency vs. disk I/O overhead
- Recommended: Save every 500-1000 batches for typical datasets

## Troubleshooting

### Error: "Dataloader must be created with persistent_iterator=True"

**Solution:** Recreate the dataloader with the flag enabled:
```python
dataloader = create_data_loader(config, persistent_iterator=True, ...)
```

### Error: "No checkpoint found in {directory}"

**Solution:** Verify the checkpoint directory exists and contains checkpoint files:
```python
import os
checkpoint_dir = "./checkpoints/dataloader"
if os.path.exists(checkpoint_dir):
    files = os.listdir(checkpoint_dir)
    print(f"Files in checkpoint dir: {files}")
```

### Iterator doesn't resume from correct position

**Possible causes:**
- Checkpoint was saved at a different point than expected
- Dataset configuration changed between save and load
- Using different shuffle seeds

**Solution:** Ensure consistent dataset configuration and verify the batch counter:
```python
batches_seen = dataloader.load_dataloader_state(checkpoint_dir)
print(f"Resuming from batch {batches_seen}")
assert batches_seen == expected_batch_count
```

## Related Files

- `src/openpi_cot/dataloader/cot_data_loader.py` - Main dataloader implementation
- `src/openpi_cot/dataloader/cot_rlds_dataset.py` - Dataset implementation with `create_checkpointable_iterator()`
