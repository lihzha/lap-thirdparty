# Training Test Scripts

This directory contains test scripts for validating the training infrastructure without running expensive real training.

## Overview

### 1. `test_training_mock.py` - Full Training Loop Test

A comprehensive mock training script that simulates the entire training pipeline with fake components:

- **Fake Dataset**: Generates random batches with correct structure
- **Fake Model**: Minimal parameters (just a few arrays)
- **Fake Loss**: Deterministic loss calculation with controllable high-loss samples
- **Real Infrastructure**: Actual optimizer, checkpointing, logging, hard example tracking

**Use this to test:**
- Training loop logic
- Checkpoint saving/loading
- Hard example tracking integration
- Multi-host coordination (if on TPU/multi-GPU)
- Memory management

**Usage:**

```bash
# Quick test (100 steps)
python scripts/test_training_mock.py

# Custom configuration
python scripts/test_training_mock.py \
    --num_train_steps 200 \
    --batch_size 32 \
    --log_interval 20 \
    --hard_example_log_interval 100 \
    --max_hard_examples_buffer 50

# Disable hard example tracking
python scripts/test_training_mock.py --no_hard_examples

# Custom checkpoint directory
python scripts/test_training_mock.py --checkpoint_dir /tmp/my_test_ckpts
```

### 2. `test_hard_examples_integration.py` - Focused Hard Examples Test

A simpler, focused test specifically for hard example tracking:

- **Purpose**: Validate hard example detection, storage, and retrieval
- **Fast**: Small images (64x64), minimal overhead
- **Focused**: Tests only the hard example tracking feature

**Use this to test:**
- Buffer management (top-K selection)
- Image extraction and resizing
- Deduplication logic
- Memory usage with many samples

**Usage:**

```bash
# Quick test (50 steps)
python scripts/test_hard_examples_integration.py

# Longer stress test
python scripts/test_hard_examples_integration.py --steps 500 --buffer_size 100

# Large batches
python scripts/test_hard_examples_integration.py --batch_size 64 --steps 100
```

## Key Features

### Mock Components

Both scripts use realistic mock components:

**MockTokenizer**:
- `encode()`: Deterministic token generation
- `decode()`: Generates readable action strings like "move right 10cm and move forward 5cm"

**MockDataset**:
- Generates batches with same structure as real data
- Includes: images, tokenized prompts, masks, robot state
- Deterministic (uses seed for reproducibility)

**MockModel**:
- Minimal parameter arrays (128-dim hidden state)
- `compute_loss()`: Generates realistic loss distribution
  - Base losses: 0.5-2.0
  - High losses: 3.0-10.0 (for ~10-15% of samples)
  - Deterministic based on batch content

### Hard Example Tracking

Both scripts test the `HardExampleTracker`:

1. **Update losses**: Track per-sample losses each step
2. **Add examples**: Extract high-loss samples with images
3. **Buffer management**: Maintain top-K samples (default: 50)
4. **Logging**: Periodically generate payloads for wandb

**Expected behavior:**
- Buffer size stays <= max_hard_examples
- Buffer is always sorted by loss (descending)
- High-loss samples are captured correctly
- Images are resized and stored efficiently

## Expected Output

### test_training_mock.py

```
================================================================================
MOCK TRAINING TEST
================================================================================
Configuration:
  Steps: 100
  Batch size: 16
  Log interval: 10
  Track hard examples: True
  Checkpoint dir: /tmp/mock_training_checkpoints
================================================================================
Initializing mock model and training state...
Initializing hard example tracker...
Starting training loop...
Step    0: loss=2.1234, grad_norm=0.4567, param_norm=12.3456, token_acc=0.7890, max_per_sample_loss=8.2341
Step   10: loss=1.9876, grad_norm=0.4123, param_norm=12.3567, token_acc=0.8012, max_per_sample_loss=7.5432
...
Step   50: loss=1.7654, grad_norm=0.3890, param_norm=12.4012, token_acc=0.8234, max_per_sample_loss=6.8901
  Hard examples: 10 entries, threshold=5.2341, total_samples=816
    #1: loss=8.2341, step=0, lang_action=move right 15cm and move forward 10cm (len=32)...
    #2: loss=7.8901, step=10, lang_action=move left 20cm and move up 5cm (len=28)...
    #3: loss=7.5432, step=20, lang_action=move forward 12cm and move down 8cm (len=35)...
...
================================================================================
Training complete!
================================================================================
```

### test_hard_examples_integration.py

```
================================================================================
HARD EXAMPLE TRACKING INTEGRATION TEST
================================================================================
Config:
  Steps: 50
  Batch size: 16
  Buffer size: 50
  Log interval: 10
  Process index: 0
  Process count: 1
================================================================================
Step   0: loss_range=[0.51, 12.34], high_loss_count=2, buffer_size=2
Step  10: loss_range=[0.63, 9.87], high_loss_count=1, buffer_size=3
...
--------------------------------------------------------------------------------
HARD EXAMPLES AT STEP 50:
  Total entries: 10
  Quantile threshold: 5.4567
  Interval samples: 800
    #1: loss=12.3456, process=0, global_idx=8, image_shape=(64, 64, 3)
    #2: loss=11.2345, process=0, global_idx=24, image_shape=(64, 64, 3)
    ...
--------------------------------------------------------------------------------
================================================================================
TEST COMPLETE
================================================================================
Statistics:
  Total samples seen: 800
  Total high-loss samples (>4.0): 112
  High-loss percentage: 14.00%
  Max loss seen: 12.3456
  Final buffer size: 50
  Final keys tracked: 50
✓ Buffer size constraint satisfied
✓ Buffer correctly sorted by loss
✓ Keys match buffer entries
✓ All buffer entries have valid images
================================================================================
✅ TEST PASSED
```

## Common Issues

### Issue: Import errors

**Solution:** Make sure you're running from the repository root or that the path is set correctly:

```bash
cd /path/to/openpi-cot
python scripts/test_training_mock.py
```

### Issue: JAX device errors

**Solution:** The scripts work with CPU, GPU, or TPU. If you get device errors:

```bash
# Force CPU
JAX_PLATFORMS=cpu python scripts/test_training_mock.py
```

### Issue: Out of memory

**Solution:** Reduce batch size or buffer size:

```bash
python scripts/test_training_mock.py --batch_size 8 --max_hard_examples_buffer 25
```

### Issue: Checkpoint errors

**Solution:** Make sure the checkpoint directory is writable:

```bash
python scripts/test_training_mock.py --checkpoint_dir /tmp/test_ckpts
```

## Multi-Host Testing

To test multi-host gathering on TPU/multi-GPU:

```bash
# Will automatically use multiple hosts if available
python scripts/test_training_mock.py --fsdp_devices 8

# Or use JAX's distributed initialization
python -m jax.distributed scripts/test_training_mock.py
```

The hard example tracker will:
1. Each host collects its own high-loss samples independently
2. At log interval, all hosts serialize their payloads
3. Process 0 gathers all payloads using `process_allgather`
4. Process 0 merges with round-robin fairness
5. Process 0 logs the top-K examples to wandb

## Performance Benchmarks

On a typical workstation (CPU only):

| Script | Steps | Batch Size | Time | Memory |
|--------|-------|------------|------|--------|
| test_training_mock.py | 100 | 16 | ~30s | ~500MB |
| test_training_mock.py | 500 | 32 | ~3min | ~800MB |
| test_hard_examples_integration.py | 50 | 16 | ~5s | ~200MB |
| test_hard_examples_integration.py | 500 | 64 | ~45s | ~600MB |

These tests are **much faster** than real training because:
- No heavy model forward/backward passes
- No real data loading from disk/GCS
- Minimal computation in loss function
- Small images (128x128 or 64x64)

## Next Steps

After these tests pass:

1. **Fix any identified issues** in `HardExampleTracker`
2. **Integrate into real training** (`scripts/train.py`)
3. **Test on small real dataset** (few hundred steps)
4. **Test on multi-host setup** (TPU pod)
5. **Monitor wandb** for logged hard examples

## Debugging

Add verbose logging:

```bash
# Set logging level
export JAX_LOG_LEVEL=DEBUG
python scripts/test_training_mock.py
```

Add breakpoints in the code:

```python
# In test_training_mock.py or test_hard_examples_integration.py
import pdb; pdb.set_trace()
```

Check buffer state:

```python
# After tracker.add_local_examples()
print(f"Buffer: {len(tracker._hard_example_buffer)}")
print(f"Top losses: {[e['loss'] for e in tracker._hard_example_buffer[:5]]}")
print(f"Keys: {len(tracker._hard_example_keys)}")
```
