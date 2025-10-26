# DataLoader Checkpoint Tests

This directory contains tests for the dataloader state checkpointing functionality.

## Overview

The checkpoint functionality allows you to save and restore the exact position of the dataloader iterator during training. This enables:
- Resuming training from any point in the dataset
- No data duplication or skipping when resuming
- Tracking the number of batches processed

## Test Files

### 1. `test_dataloader_checkpoint_simple.py` (Recommended)

A simple, easy-to-run test that validates basic checkpoint functionality.

**Usage:**
```bash
# Run with your config
python scripts/tests/test_dataloader_checkpoint_simple.py --config-name=your_config

# Example with a specific config
python scripts/tests/test_dataloader_checkpoint_simple.py --config-name=gemma3
```

**What it tests:**
- ✓ Creating dataloader with persistent_iterator=True
- ✓ Iterating through batches
- ✓ Saving checkpoint
- ✓ Loading checkpoint into new dataloader
- ✓ Continuing iteration from restored state
- ✓ Error handling for non-persistent iterators

**Expected output:**
```
================================================================================
DataLoader Checkpoint Test
================================================================================

Loading configuration...
✓ Config loaded: your_config
✓ Checkpoint directory: /tmp/.../test_checkpoint

Step 1: Creating dataloader with persistent_iterator=True...
✓ Dataloader created successfully

Step 2: Iterating through 5 batches...
  Batch 1: obs.state shape=(32, 1, 7), actions shape=(32, 16, 7)
  Batch 2: obs.state shape=(32, 1, 7), actions shape=(32, 16, 7)
  ...
✓ Successfully processed 5 batches

...

================================================================================
ALL TESTS PASSED! ✓
================================================================================
```

### 2. `test_dataloader_checkpoint.py`

A comprehensive test suite with multiple test scenarios.

**Usage:**
```bash
python scripts/tests/test_dataloader_checkpoint.py --config-name=your_config
```

**What it tests:**
- All tests from the simple version
- Multiple save/load cycles
- Batch identifier tracking
- Multi-host scenario handling (if applicable)

## Quick Start

### Prerequisites

1. Make sure you have a valid training config
2. Ensure your data is accessible (RLDS data directory configured)

### Running the Tests

**Option 1: Simple test (fastest)**
```bash
cd /home/irom-lab/projects/openpi-cot
python scripts/tests/test_dataloader_checkpoint_simple.py --config-name=gemma3
```

**Option 2: Comprehensive test**
```bash
cd /home/irom-lab/projects/openpi-cot
python scripts/tests/test_dataloader_checkpoint.py --config-name=gemma3
```

## Integration with Training

To use checkpointing in your training script, see the updated `scripts/train.py`:

```python
# Create dataloader with persistent_iterator=True
data_loader = _data_loader.create_data_loader(
    config,
    sharding=data_sharding,
    shuffle=True,
    seed=config.seed,
    persistent_iterator=True,  # Enable checkpointing
)

# During training, save the dataloader state along with model checkpoint
if step % config.save_interval == 0:
    # Save model checkpoint (existing code)
    checkpoint_manager = _checkpoints.save_state(
        checkpoint_manager,
        train_state,
        data_loader,  # Pass dataloader to save its state
        step,
        ...
    )

# When resuming, load the dataloader state
if resuming:
    train_state = _checkpoints.restore_state(
        checkpoint_manager,
        train_state,
        data_loader,  # Pass dataloader to restore its state
    )
```

## Troubleshooting

### Error: "Dataloader must be created with persistent_iterator=True"

**Solution:** Always create the dataloader with `persistent_iterator=True`:
```python
data_loader = create_data_loader(..., persistent_iterator=True)
```

### Error: "No checkpoint found in {directory}"

**Solution:** Ensure you've saved a checkpoint first:
```python
# Save before trying to load
data_loader.save_dataloader_state("./checkpoints/dataloader")

# Then load
data_loader.load_dataloader_state("./checkpoints/dataloader")
```

### Test fails with config loading error

**Solution:** Make sure you're running from the project root and have a valid config:
```bash
cd /home/irom-lab/projects/openpi-cot
python scripts/tests/test_dataloader_checkpoint_simple.py --config-name=<your_config>
```

### Memory issues with checkpoints

**Note:** Checkpoint files can be large (hundreds of MB) due to buffered data. This is expected behavior. The checkpoints include:
- Iterator position
- Shuffle buffers
- Prefetch buffers
- Batch counter

## Additional Resources

- See `docs/dataloader_checkpoint_example.md` for detailed API documentation
- See implementation in `src/openpi_cot/dataloader/cot_data_loader.py`
- See dataset implementation in `src/openpi_cot/dataloader/cot_rlds_dataset.py`

## Testing on Different Configs

The tests work with any valid training config. Try different configs to ensure compatibility:

```bash
# Test with different configs
python scripts/tests/test_dataloader_checkpoint_simple.py --config-name=config1
python scripts/tests/test_dataloader_checkpoint_simple.py --config-name=config2
python scripts/tests/test_dataloader_checkpoint_simple.py --config-name=config3
```

## Expected Test Duration

- **Simple test**: ~30-60 seconds (depends on dataset loading time)
- **Comprehensive test**: ~2-5 minutes

Most of the time is spent on initial dataset loading and iterator creation.
