# Language Actions Padding Fix - Summary

## Issues Analyzed in `group_language_actions` (when `self.use_json_actions==False`)

### Issue 1: 2D vs 1D Indices - No Change Needed ✓

**Question:** Should we use 1D indices (like in `add_prediction_pairs`) instead of 2D indices?

**Answer:** No, the 2D indexing approach is actually **more efficient** and correct.

**Reasoning:**
- **2D approach (current):** Uses `tf.gather(traj["raw_action"], summation_indices[:, :trimmed_len])` - a single batched gather operation
- **1D approach (in `add_prediction_pairs`):** Uses `tf.map_fn` to iterate over each timestep separately

**Why 2D is better:**
1. **Performance:** Batched operations are faster than iterating with `tf.map_fn`
2. **Graph optimization:** TensorFlow can optimize a single gather better than multiple serial operations
3. **Memory efficiency:** No intermediate function calls or temporary tensors per timestep

**When to use 1D (tf.map_fn):**
- When the computation per timestep is complex and varies (like in `add_prediction_pairs` where deltas differ per timestep)
- When you need different-sized windows per timestep

**Conclusion:** Keep the 2D indexing approach for `group_language_actions`.

---

### Issue 2: Clamping Logic Bug - FIXED ✓

**Bug Confirmed:** The clamping logic was incorrect, causing the last action to be **repeated** instead of **zero-padded**.

**Root Cause:**
```python
# BUGGY CODE (before fix):
summation_indices = compute_window_indices(traj_len, summation_steps)
actions_window_trim = tf.gather(traj["raw_action"], summation_indices[:, :trimmed_len])
```

The `compute_window_indices` function clamps indices to `traj_len - 1`:
```python
indices = base + offsets
return tf.minimum(indices, sequence_length - 1)  # <-- This causes repetition!
```

**Problem:**
- At timestep `t=7` (last timestep in an 8-step trajectory)
- Wants to gather next 5 actions: indices `[7, 8, 9, 10, 11]`
- After clamping: `[7, 7, 7, 7, 7]`
- Result: Action at index 7 is gathered 5 times → **repetition instead of zero-padding**

**Example from Test Output:**
```
Trajectory length: 8
Summation steps: 5

BUGGY - Timestep 7:
  [[8. 8. 8.]
   [8. 8. 8.]  <-- Repeated!
   [8. 8. 8.]  <-- Repeated!
   [8. 8. 8.]  <-- Repeated!
   [8. 8. 8.]] <-- Repeated!

FIXED - Timestep 7:
  [[8. 8. 8.]
   [0. 0. 0.]  <-- Zero-padded ✓
   [0. 0. 0.]  <-- Zero-padded ✓
   [0. 0. 0.]  <-- Zero-padded ✓
   [0. 0. 0.]] <-- Zero-padded ✓
```

---

## The Fix

**Strategy:**
1. Create indices **without** initial clamping
2. Create a `valid_mask` to track which indices are in-bounds
3. Clamp indices **only for gathering** (to avoid TensorFlow errors)
4. Multiply gathered actions by `valid_mask` to **zero out** invalid positions

**Code (Applied to `cot_rlds_dataset.py:519-550`):**
```python
else:
    # FIX: Create indices without clamping to avoid repeating last action
    # We want zero-padding for out-of-bounds indices, not repetition
    base = tf.broadcast_to(tf.range(trimmed_len)[None], [traj_len, trimmed_len])
    offsets = tf.broadcast_to(tf.range(traj_len)[:, None], [traj_len, trimmed_len])
    indices = base + offsets  # [T, trimmed_len], can exceed traj_len - 1

    # Create mask for valid indices
    valid_mask = indices < traj_len  # [T, trimmed_len]

    # Clamp indices only for gathering (to avoid out-of-bounds errors)
    clamped_indices = tf.minimum(indices, traj_len - 1)

    # Gather actions using clamped indices
    actions_window_trim = tf.gather(traj["raw_action"], clamped_indices)  # [T, trimmed_len, A]

    # Zero out invalid elements (this is the key fix!)
    actions_window_trim = actions_window_trim * tf.cast(
        tf.expand_dims(valid_mask, -1), actions_window_trim.dtype
    )

    # ... rest of padding logic unchanged
```

---

## Impact

**Who is affected:**
- Only datasets with `use_json_actions=False` (non-DROID datasets like OXE datasets)
- Only affects the **last few timesteps** of each trajectory

**Severity:**
- **High** for short trajectories (3-10 steps) where control_frequency < summation_steps
- **Medium** for longer trajectories where only the last `summation_steps` frames are affected
- **Training impact:** The model was trained on incorrect (repeated) language actions for tail timesteps, potentially affecting end-of-episode behavior

**Benefit of fix:**
- Correct zero-padding ensures language actions properly represent "no future action"
- Consistent with the intended design and matches JSON-based language action handling
- Better training signal for end-of-episode scenarios

---

## Testing

The test file `test_language_actions_padding.py` demonstrates:
1. The bug with concrete examples showing repetition
2. The fix with proper zero-padding
3. Edge cases (very short trajectories)
4. Quantitative difference between buggy and fixed versions

**Run test:**
```bash
python test_language_actions_padding.py
```

**Key test results:**
- ✓ Confirms bug exists in current implementation
- ✓ Confirms fix properly zero-pads
- ✓ Shows exact numerical differences
- ✓ Validates edge cases

---

## Conclusion

1. **2D indexing:** Keep it - it's more efficient than 1D (no change needed)
2. **Clamping bug:** Fixed by using valid_mask approach to zero-pad instead of repeat
3. **Testing:** Comprehensive test demonstrates both the bug and the fix
4. **Code location:** `src/openpi_cot/dataloader/cot_rlds_dataset.py:519-550`
