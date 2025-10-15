# Fix Summary: std=0 Bug in normalize_adapter.py

## Problem

Some dimensions in computed norm_stats had `std=0` even though `q01 != q99`, indicating the data **should** vary but the variance calculation collapsed to zero.

## Root Cause

**Float32 accumulation errors** when summing millions of samples.

The variance formula `var = E[X^2] - E[X]^2` requires high precision when:
1. Dataset has millions of samples (e.g., 27M)
2. Computing sums via `.sum()` accumulates float32 rounding errors
3. Even with shifted statistics, accumulated errors can make variance **slightly negative**
4. Negative variance gets clamped to 0, producing std=0

Real example from user's dataset:
- State dimension 0: **27,630,375 samples**
- Range: [0.27, 0.78] (expected std ≈ 0.15)
- **Result in float32**: variance = -0.00001 → clamped to 0 → std = 0
- **Result in float64**: variance = 0.0225 → std = 0.15 ✓

This happens even when data has **large variation** because the formula `E[X^2] - E[X]^2` is numerically unstable when both terms are similar in magnitude.

## The Fix

### 1. Promote input data to float64 before computing statistics (lines 127-134):

```python
# Promote to float64 for numerical stability in variance calculation
# Float32 accumulation errors over millions of samples can cause variance to become
# slightly negative (violating E[X^2] >= E[X]^2), which gets clamped to 0.
actions = actions.astype(np.float64)
if has_state:
    proprios = proprios.astype(np.float64)
```

### 2. Add diagnostic warnings for negative variance (lines 207-230):

```python
# Check for negative variance (indicates numerical instability)
if (a_var < 0).any():
    neg_dims = np.where(a_var < 0)[0]
    logging.warning(
        f"Action dims {neg_dims.tolist()} have negative variance, "
        f"clamping to 0. This indicates numerical instability."
    )
```

### 3. Cast final statistics back to float32 for storage efficiency (lines 258-286)

## Why This Works

- **Float64 has ~16 significant digits** vs float32's ~7
- Computing mean/std/variance in float64 preserves precision for tiny relative variations
- Final statistics are cast back to float32 (mean/std don't need extra precision for storage)

## Test Coverage

`test_normalize_stats_multihost.py` includes:
- **Test 8**: Demonstrates the bug (float32 → std=0) and verifies the fix (float64 → std>0)
- All tests validate multi-host gathering correctness

## Important Caveat

If a dimension **truly collapsed to a single constant value in float32** (only 1 unique value), promoting to float64 won't help - std will still be 0. This is **correct behavior** for truly constant dimensions.

The fix helps when:
- Dimension has 2+ unique float32 values (even if very close)
- Variance calculation would otherwise collapse due to precision

## Verification

Run the test to verify:
```bash
python test_normalize_stats_multihost.py
```

Check Test 8 output for:
```
WITHOUT FIX (compute in float32):
  ❌ FAILURE: Dims [X] have std=0 due to float32 precision!

WITH FIX (promote to float64 before stats):
  ✅ FIXED! All dims now have std > 0
```

## Files Modified

1. `src/openpi_cot/shared/adapters/normalize_adapter.py`
   - Lines 127-134: Promote to float64
   - Lines 207-215: Add warning for negative action variance
   - Lines 222-230: Add warning for negative state variance
   - Lines 263-266: Cast state stats back to float32
   - Lines 289-292: Cast action stats back to float32

2. `test_normalize_stats_multihost.py` (enhanced)
   - Added Tests 6-8 for edge cases
   - Test 8 specifically validates the fix

3. `test_user_specific_case.py` (new)
   - Reproduces the exact scenario from user's dataset
   - Tests with 27M samples to verify float32 accumulation errors

4. `diagnose_dataset_std_zero.py` (new)
   - Diagnostic tool to inspect norm_stats.json
   - Identifies dimensions with std=0 and checks if q01==q99
