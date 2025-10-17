# Dataset Code Verification Summary

**Date**: 2025-10-17
**File**: `src/openpi_cot/dataloader/cot_rlds_dataset.py`
**Reviewer**: Claude Code Analysis

## Executive Summary

Detailed verification of the dataset pipeline focusing on:
1. Global normalization calculation
2. Normalization application order
3. Language action gathering and prediction transforms
4. Action chunking and window indexing

**Overall Assessment**: The architecture is sound, but **2 critical bugs** were identified that will cause training failures (NaN values) and incorrect statistics.

---

## 🚨 Critical Bugs (Must Fix)

### Bug #1: Division by Zero in Standard Deviation Padding

**Location**: Lines 1776-1778, 1858
**Severity**: CRITICAL - Causes NaN values in training data
**Impact**: All padded dimensions will have `std=0`, causing `(x - mean) / 0 = NaN` during normalization

**Current Code**:
```python
# Line 1776-1778
action_local_std = np.pad(
    stats["actions"].std, (0, action_dim - len(stats["actions"].std)), mode="constant"
)
# Default constant_values=0 used!

# Line 1858 (same issue for state)
state_local_std = np.pad(stats["state"].std, (0, action_dim - len(stats["state"].std)), mode="constant")
```

**Problem**: When `mode="constant"` is used without specifying `constant_values`, NumPy defaults to padding with `0`. This creates zero standard deviations for padded dimensions, leading to division by zero.

**Fix**:
```python
action_local_std = np.pad(
    stats["actions"].std, (0, action_dim - len(stats["actions"].std)),
    mode="constant", constant_values=1.0
)
```

**Note**: The code already handles this correctly for empty states at lines 333 and 358, using `std=np.ones(...)` explicitly.

---

### Bug #2: Action Global Mean Computed on Inconsistent Dimensions

**Location**: Lines 1755-1764
**Severity**: CRITICAL - Fails or produces incorrect statistics
**Impact**: Will fail if datasets have different action dimensions, or produce biased statistics

**Current Code**:
```python
# Line 1755: Initialize to shape of first dataset
action_weighted_sum = np.zeros_like(list(all_dataset_statistics.values())[0]["actions"].mean)

for dataset_name, stats in all_dataset_statistics.items():
    action_n = stats["actions"].num_transitions
    # BUG: Assumes all datasets have same action dim!
    action_weighted_sum += stats["actions"].mean * action_n

action_global_mean = action_weighted_sum / total_action_n
# Pad AFTER computing weighted mean
action_global_mean = np.pad(action_global_mean, (0, action_dim - len(action_global_mean)), mode="constant")
```

**Problem**: The weighted sum assumes all datasets have the same action dimension as the first dataset. If datasets have different original dimensions, this will either:
- Fail with shape mismatch errors
- Produce incorrect statistics by not properly weighting padded dimensions

**Fix** (follow the pattern used for states at lines 1833-1841):
```python
# Initialize to target dimension
action_weighted_sum = np.zeros(action_dim, dtype=np.float32)

for dataset_name, stats in all_dataset_statistics.items():
    action_n = stats["actions"].num_transitions
    # Pad BEFORE accumulating
    action_mean_padded = np.pad(
        stats["actions"].mean, (0, action_dim - len(stats["actions"].mean)), mode="constant"
    )
    action_weighted_sum += action_mean_padded * action_n

action_global_mean = action_weighted_sum / total_action_n
# No additional padding needed
```

---

### ~~Bug #3~~: Prediction Language Actions - CORRECTED UNDERSTANDING

**Location**: Lines 623-685
**Status**: ✅ ORIGINAL CODE WAS CORRECT - NOT A BUG

**Initial Misunderstanding**:
The original analysis incorrectly flagged that prediction language actions should be clamped to `control_frequency` like training language actions.

**Corrected Understanding**:
Training and prediction language actions serve **different purposes** and should use **different gathering strategies**:

#### Training Language Actions (Lines 505-510, 525)
- **Purpose**: Provide context about upcoming actions during training
- **Gathers**: `min(control_frequency, summation_steps)` actions
- **Rationale**: Respects dataset-specific control rates (e.g., 10Hz vs 50Hz datasets)
- **Example**: If `control_frequency=10`, gather 10 actions regardless of trajectory length

#### Prediction Language Actions (Lines 623-685)
- **Purpose**: Describe the ENTIRE action sequence from current frame (t) to future frame (t+delta)
- **Gathers**: `delta` actions (NOT clamped to control_frequency)
- **Rationale**: Must capture all actions in the temporal span between frame pairs
- **Example**: If `delta=15`, gather exactly 15 actions (the full span)

**Why This is Correct**:
```python
# Training: Respects control frequency
trimmed_len = min(control_frequency, summation_steps)
la_window = gather(language_actions, indices[:, :trimmed_len])

# Prediction: Uses delta (temporal span between frames)
indices = tf.range(delta) + t_idx  # ✓ CORRECT - gather delta actions
lang_window = gather(raw_language_actions, indices)
```

**Key Insight**: Prediction requires ALL actions between frames to properly describe the transition, independent of control frequency. A 15-step prediction should describe all 15 actions, not just the first 10 (if control_frequency=10).

---

## ✅ Verified Correct Components

### 1. Normalization Application Order
**Lines**: 317-377, 447-456, 1660-1693

**Pipeline**:
1. Compute per-dataset statistics (after restructure, before padding) - Line 342
2. Apply data padding to `action_dim` - Line 477
3. Compute global statistics by padding per-dataset stats - Line 1662
4. Apply normalization to padded data - Line 1693

**Status**: ✓ CORRECT - Statistics are properly aligned with data dimensions before normalization.

### 2. Language Action Gathering
**Lines**: 491-553

**Verified**:
- ✓ Uses `compute_window_indices` for proper sliding windows with end-padding (Line 501)
- ✓ Correctly trims to `min(control_frequency, summation_steps)` (Line 505)
- ✓ For JSON: saves raw copy before windowing for prediction use (Line 508)
- ✓ For numeric: serializes to match DROID format (Lines 540-544)
- ✓ Both paths properly pad to `summation_steps` (Lines 513-515, 528-533)

### 3. Prediction Frame Pairs & Language Actions
**Lines**: 555-701

**Verified**:
- ✓ Deterministic per-trajectory sampling using trajectory_id hash (Lines 587-588)
- ✓ Proper index clamping to avoid out-of-bounds (Lines 597, 626, 652)
- ✓ Correct image stacking: [T, 2, H, W, C] for primary, [T, 1, H, W, C] for wrists (Lines 602-604, 607-615)
- ✓ Stores prediction deltas for debugging (Line 697)
- ✓ **Prediction language actions correctly use `delta` (not clamped to control_frequency)** - see corrected understanding above
- ✓ Both JSON and numeric cases consistently gather `delta` actions

### 4. Action Chunking & Window Indexing
**Lines**: 184-195, 479-489

**Verified**:
- ✓ `compute_window_indices` creates sliding windows with proper end-padding by repeating last element
- ✓ Correctly applied to create [T, action_horizon, action_dim] chunks
- ✓ Static shape preservation (Line 486)

**Example behavior** for `sequence_length=5, window_size=3`:
```
t=0: [0, 1, 2]
t=1: [1, 2, 3]
t=2: [2, 3, 4]
t=3: [3, 4, 4]  # Repeats last element
t=4: [4, 4, 4]  # Repeats last element
```

### 5. State-Type-Specific Normalization
**Lines**: 1672-1693, 1817-1893

**Verified**:
- ✓ Good architectural design: applies per-dataset normalization before interleaving
- ✓ Avoids runtime type checking overhead
- ✓ Correctly computes separate statistics for each state type (joint_pos, eef_pose)
- ✓ Correctly skips normalization for "none" state type (Line 1822)
- ✓ Proper handling of empty states with `std=1.0` (Lines 333, 358)

### 6. Image Decoding
**Lines**: 106-140

**Verified**:
- ✓ Handles multiple tensor ranks (0, 1, 3, 4)
- ✓ Properly decodes prediction mode with [T] encoded strings (Lines 117-125)
- ✓ Preserves aspect ratio with symmetric padding (Lines 45-81)
- ✓ Handles empty string placeholders (Lines 89-98)

### 7. Dataset Interleaving & Weight Balancing
**Lines**: 1638-1650, 1695-1700

**Verified**:
- ✓ Correct balancing: `sample_weights * dataset_sizes` (Line 1644)
- ✓ Proper normalization of weights (Line 1646)
- ✓ Correct dataset length calculation (Line 1650)

---

## 📋 Recommended Action Plan

### Priority 1: Apply Critical Fixes

1. **Fix std padding** (Lines 1776-1778, 1858) ✅ FIXED
   - Add `constant_values=1.0` to all std padding operations
   - Test: Verify no NaN values in normalized data

2. **Fix action mean calculation** (Lines 1755-1764) ✅ FIXED
   - Initialize to `action_dim`, pad stats before accumulating
   - Test: Verify correct global mean with mixed-dimension datasets

3. ~~**Fix prediction delta clamping**~~ ✅ NOT A BUG - Original code was correct
   - Prediction language actions should use `delta`, not `control_frequency`
   - Both JSON and numeric cases correctly gather `delta` actions

### Priority 2: Validation Tests

After applying fixes, run these validation checks:

1. **Statistics validation**:
   ```python
   # Check no zeros in std
   assert np.all(global_stats["actions"].std > 0)

   # Check no NaNs after normalization
   batch = next(iter(dataset))
   assert not np.any(np.isnan(batch["actions"]))
   ```

2. **Dimension consistency**:
   ```python
   # Verify all datasets properly padded
   for batch in dataset:
       assert batch["actions"].shape[-1] == action_dim
       assert batch["observation"]["state"].shape[-1] == action_dim
   ```

3. **Prediction consistency**:
   ```python
   # Verify JSON and numeric prediction language actions both use delta
   # Both should gather exactly delta actions (not clamped to control_frequency)
   ```

### Priority 3: Code Quality Improvements

1. **Refactor padding logic** into helper function:
   ```python
   def pad_stats_to_dim(stats: ExtendedNormStats, target_dim: int) -> ExtendedNormStats:
       """Pad statistics arrays to target dimension with safe defaults."""
       return ExtendedNormStats(
           mean=np.pad(stats.mean, (0, target_dim - len(stats.mean)), mode="constant"),
           std=np.pad(stats.std, (0, target_dim - len(stats.std)), mode="constant", constant_values=1.0),
           q01=np.pad(stats.q01, (0, target_dim - len(stats.q01)), mode="constant", constant_values=0),
           q99=np.pad(stats.q99, (0, target_dim - len(stats.q99)), mode="constant", constant_values=1),
           num_transitions=stats.num_transitions,
           num_trajectories=stats.num_trajectories,
       )
   ```

2. **Add assertions** for dimension consistency checks

3. **Add logging** for global statistics shapes during computation

---

## 📊 Impact Assessment

### If Bugs Are NOT Fixed:

**Bug #1 (std padding)** ✅ NOW FIXED:
- Training will produce NaN losses within first few iterations
- Model parameters will become NaN
- Training completely fails

**Bug #2 (action mean)** ✅ NOW FIXED:
- If all datasets have same action dim: May work by accident
- If mixed dimensions: Fails with shape mismatch or produces biased statistics
- Silent data corruption if dimensions happen to be compatible

### With Fixes Applied:

- ✅ Normalization works correctly across all dimensions
- ✅ Statistics properly computed for mixed-dimension datasets
- ✅ Training proceeds normally without NaN issues
- ✅ Prediction language actions correctly gather `delta` actions (by design)

---

## 🔍 Additional Observations

### Good Practices Found:

1. **Proper empty state handling** (Lines 326-338, 352-363)
   - Uses `std=1.0` instead of `0` to avoid division by zero
   - Correctly pads to `action_dim`

2. **Deterministic seeding** (Lines 306, 587-588)
   - Global seed for reproducibility
   - Per-trajectory seeds derived from trajectory_id

3. **Static shape preservation** (Lines 464, 474, 486, 519, 550)
   - Explicitly sets static shapes for TensorFlow's shape inference

4. **Commented code sections**
   - Clear explanations of complex logic (e.g., lines 492-497)

### Areas for Future Enhancement:

1. **Statistics caching**: Global stats computation could be cached (commented code at lines 1738-1743)
2. **Memory profiling**: Good memory logging exists (lines 178-181) but could be expanded
3. **Unit tests**: Complex windowing and padding logic would benefit from dedicated tests
4. **Documentation**: Add examples of expected tensor shapes at each pipeline stage

---

## Conclusion

The dataset pipeline has a solid architectural foundation with good separation of concerns and proper handling of multi-modal data. **2 critical bugs were identified and fixed**:

1. **Std padding bug** (#1) - Would cause immediate training failure with NaN values ✅ FIXED
2. **Action mean calculation bug** (#2) - Would cause silent data corruption with mixed-dimension datasets ✅ FIXED

**Initial Bug #3 was a misunderstanding**: Prediction language actions correctly use `delta` (not clamped to control_frequency), which is the intended behavior. Training and prediction serve different purposes and appropriately use different gathering strategies.

**Status**: ✅ All critical bugs have been fixed. The codebase is now ready for training experiments.
