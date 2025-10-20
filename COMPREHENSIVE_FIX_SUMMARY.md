# Comprehensive Dataloader Fixes Summary

This document summarizes all the fixes applied to the `cot_rlds_dataset.py` dataloader.

---

## Overview of Issues Fixed

1. **Zero-padding vs Repetition Bug** (in `chunk_actions` and `group_language_actions`)
2. **Inefficient 1D Gathering** (in `add_prediction_pairs`)
3. **Invalid Action Gathering** (in `add_prediction_pairs` - realized_deltas issue)

---

## Issue 1: Zero-Padding vs Repetition Bug

### The Problem

The original `compute_window_indices` function would **repeat the last element** instead of **zero-padding** when gathering windows near the trajectory end.

**Affected functions:**
- `chunk_actions`: Chunking actions into horizon-length windows
- `group_language_actions`: Gathering future actions for language action generation

**Root cause:**
```python
def compute_window_indices(sequence_length, window_size):
    indices = base + offsets  # [T, window]
    return tf.minimum(indices, sequence_length - 1)  # ❌ Clamps to last index
```

When indices exceed the sequence length, they get clamped to `sequence_length - 1`, causing the last element to be gathered multiple times.

**Example:**
```python
# Trajectory length: 8, window size: 4
# At timestep t=7, want indices [7, 8, 9, 10]
# After clamping: [7, 7, 7, 7]
# Result: action[7] repeated 4 times ❌
# Expected: [action[7], 0, 0, 0] ✓
```

### The Solution

Created a unified `gather_with_padding` function that:
1. Creates indices without clamping
2. Creates a validity mask to identify out-of-bounds indices
3. Clamps indices **only for the gather operation** (to avoid TensorFlow errors)
4. Multiplies gathered data by the validity mask to **zero out** invalid positions

**Benefits:**
- ✓ Correct zero-padding behavior
- ✓ Supports both fixed and variable-length windows
- ✓ More efficient (batched 2D operations)
- ✓ Unified implementation reduces code duplication

---

## Issue 2: Inefficient 1D Gathering

### The Problem

The original `add_prediction_pairs` used `tf.map_fn` to iterate over each timestep individually, which is inefficient.

**Old approach:**
```python
def gather_and_pad_numeric(t_idx, delta):
    # Gather for single timestep
    indices = t_idx + tf.range(delta)
    actions = tf.gather(traj["raw_action"], indices)
    # Serialize and pad...
    return padded

prediction_lang_actions = tf.map_fn(
    lambda x: gather_and_pad_numeric(x[0], x[1]),
    (tf.range(traj_len), deltas),
    ...
)
```

This creates T separate gather operations (one per timestep).

### The Solution

Use **2D batched gathering** with `gather_with_padding`:
```python
actions_window = gather_with_padding(
    data=traj["raw_action"],
    sequence_length=traj_len,
    window_size=summation_steps,
    per_timestep_windows=deltas_clamped,  # Variable window per timestep
)  # Single batched operation!
```

**Benefits:**
- ✓ **Much faster**: Single batched gather vs T serial operations
- ✓ Better TensorFlow graph optimization
- ✓ Cleaner code

---

## Issue 3: Invalid Action Gathering (Realized Deltas)

### The Problem

When sampling future frame deltas, boundary clamping would cause a mismatch between:
- The **actual visual gap** between images
- The **number of actions gathered**

**Why this matters:**
In robotics datasets:
- `action[t]` transitions from `obs[t]` → `obs[t+1]`
- For T observations, valid actions are `[action[0], ..., action[T-2]]`
- **`action[T-1]` is invalid** (would transition to non-existent `obs[T]`)

**Example of the bug:**
```python
# t=7, sampled delta=5, traj_len=10
deltas[7] = 5

# Image gathering
future_indices[7] = min(7 + 5, 9) = 9  # Clamped!
# Shows images: [obs[7], obs[9]] → actual gap = 2 steps

# Action gathering (BUGGY)
deltas_clamped[7] = min(5, summation_steps) = 5  # Still uses original delta!
# Gathers: [action[7], action[8], action[9], 0, 0]
#                                  ^^^^^^^^^ INVALID! Should not be gathered

# Problems:
# 1. action[9] represents obs[9] → obs[10], but obs[10] doesn't exist
# 2. Visual gap is 2 steps, but we gather 3 valid actions
# 3. prediction_delta would store 5, but actual gap is 2 (misleading metadata)
```

### The Solution

Compute the **realized delta** (actual visual gap after clamping) and use it for action gathering:

```python
future_indices = tf.minimum(tf.range(traj_len, dtype=tf.int32) + deltas, traj_len - 1)

# Compute REALIZED delta (actual gap)
realized_deltas = future_indices - tf.range(traj_len, dtype=tf.int32)

# Use realized_deltas for action gathering
deltas_clamped = tf.minimum(realized_deltas, summation_steps)

actions_window = gather_with_padding(
    data=traj["raw_action"],
    sequence_length=traj_len,
    window_size=summation_steps,
    per_timestep_windows=deltas_clamped,  # Uses realized deltas ✓
)

# Store accurate metadata
traj["prediction_delta"] = realized_deltas  # Not original sampled deltas
```

**With the fix:**
```python
# t=7, sampled delta=5, traj_len=10
future_indices[7] = 9
realized_deltas[7] = 9 - 7 = 2  # Actual gap ✓
deltas_clamped[7] = min(2, summation_steps) = 2

# Gathers: [action[7], action[8], 0, 0, 0]
# ✓ Only 2 valid actions (matching the 2-step visual gap)
# ✓ Does NOT include invalid action[9]
# ✓ prediction_delta stores 2 (accurate metadata)
```

**Benefits:**
- ✓ Never gathers invalid `action[T-1]`
- ✓ Number of actions matches actual visual gap
- ✓ Accurate metadata (`prediction_delta` = realized gap)
- ✓ Clearer training signal (model learns correct action sequence length)

---

## Implementation Details

### New Function: `gather_with_padding`

**Location:** `cot_rlds_dataset.py:184-247`

**Signature:**
```python
def gather_with_padding(
    data: tf.Tensor,              # Source tensor [T, ...]
    sequence_length: tf.Tensor,    # Scalar, length of sequence
    window_size: int | tf.Tensor,  # Window size to gather
    per_timestep_windows: tf.Tensor | None = None,  # Optional variable windows
) -> tf.Tensor:                    # Returns [T, window_size, ...]
```

**Key features:**
- Supports fixed window sizes (for `chunk_actions`, `group_language_actions`)
- Supports variable windows per timestep (for `add_prediction_pairs`)
- Proper zero-padding instead of repetition
- Efficient batched 2D operations

### Changes to Existing Functions

1. **`chunk_actions`** (line 527-539):
   - Now uses `gather_with_padding` with fixed window
   - Simplified from ~9 lines to ~6 lines

2. **`group_language_actions`** (line 543-590):
   - Now uses `gather_with_padding` with fixed window
   - Removed manual index creation and masking logic
   - Simplified from ~30 lines to ~18 lines

3. **`add_prediction_pairs`** (line 594-703):
   - Computes `realized_deltas` after clamping future_indices
   - Uses `gather_with_padding` with variable windows
   - Stores `realized_deltas` in metadata (not original deltas)
   - Converted from serial `tf.map_fn` to batched 2D gather
   - More efficient and semantically correct

---

## Testing

### Test Files Created

1. **`test_language_actions_padding.py`**
   - Demonstrates Issue 1 (zero-padding vs repetition)
   - Shows buggy vs fixed versions side-by-side
   - Includes edge cases (very short trajectories)

2. **`test_unified_gather_fix.py`**
   - Comprehensive test for all three use cases
   - Tests fixed windows (`chunk_actions`, `group_language_actions`)
   - Tests variable windows (`add_prediction_pairs`)
   - Tests edge cases

3. **`test_realized_deltas_fix.py`**
   - Demonstrates Issue 3 (realized deltas)
   - Shows how fix prevents gathering invalid `action[T-1]`
   - Verifies metadata correctness
   - Includes boundary condition scenarios

### Running Tests

```bash
# Test zero-padding fix
python test_language_actions_padding.py

# Test unified gather function
python test_unified_gather_fix.py

# Test realized deltas fix
python test_realized_deltas_fix.py
```

---

## Impact Analysis

### Who is Affected?

**All datasets** when using:
- Action chunking (all training)
- Language action generation (all training with `use_json_actions=False`)
- Prediction training (when `enable_prediction_training=True`)

### Severity by Issue

1. **Zero-padding bug**:
   - **High severity** for short trajectories or large action_horizon
   - Affects last few timesteps of every trajectory
   - Incorrect training signal for end-of-episode behavior

2. **Inefficient gathering**:
   - **Medium severity** (performance only, not correctness)
   - Slower training data pipeline
   - More important for large `summation_steps`

3. **Invalid action gathering**:
   - **High severity** for prediction training
   - Gathers semantically invalid actions near trajectory end
   - Misleading metadata affects model conditioning
   - Training data doesn't match intended task (predicting N actions for M-step gap where N ≠ M)

### Benefits of Fixes

1. **Correctness**: All gathered actions are semantically valid
2. **Performance**: Faster data pipeline with batched operations
3. **Training quality**: Better training signal with accurate action sequences
4. **Metadata accuracy**: `prediction_delta` truthfully represents visual gap
5. **Code quality**: Unified implementation, less duplication, easier to maintain

---

## Files Modified

1. **`src/openpi_cot/dataloader/cot_rlds_dataset.py`**
   - Added `gather_with_padding` function (lines 184-247)
   - Updated `chunk_actions` (lines 527-539)
   - Updated `group_language_actions` (lines 543-590)
   - Updated `add_prediction_pairs` (lines 594-703)
   - Removed old `compute_window_indices` function

## Files Created

1. **`test_language_actions_padding.py`** - Test for Issue 1
2. **`test_unified_gather_fix.py`** - Comprehensive test
3. **`test_realized_deltas_fix.py`** - Test for Issue 3
4. **`LANGUAGE_ACTIONS_FIX_SUMMARY.md`** - Original summary (Issue 1)
5. **`COMPREHENSIVE_FIX_SUMMARY.md`** - This document

---

## Migration Notes

### Breaking Changes

1. **`prediction_delta` values will change** for boundary cases
   - Old: Stores sampled delta (may exceed actual gap)
   - New: Stores realized delta (actual visual gap)
   - **Impact**: If model uses `prediction_delta` as input, retraining may be needed

2. **Action sequences will be different** near trajectory end
   - Old: May include repeated last action or invalid `action[T-1]`
   - New: Proper zero-padding
   - **Impact**: Models trained on old data may need retraining

### Compatibility

- ✓ No changes to dataset output shapes
- ✓ No changes to public API
- ✓ All existing configs should work
- ⚠️  Checkpoints trained with old dataloader may have slightly different behavior

---

## Summary

These fixes ensure the dataloader provides:
- ✓ **Semantically correct** action sequences
- ✓ **Efficient** data pipeline with batched operations
- ✓ **Accurate** metadata for model conditioning
- ✓ **Consistent** behavior across all datasets
- ✓ **Maintainable** unified implementation

All changes are backward-compatible in terms of API, but may affect training data distribution. Retraining is recommended for production models.
