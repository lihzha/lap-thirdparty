"""Test script to validate normalize_adapter statistics computation on multi-host TPU."""

import jax
from jax.experimental import multihost_utils as mh
import numpy as np

jax.distributed.initialize()

# Print distributed setup info
num_processes = jax.process_count()
process_id = jax.process_index()

if process_id == 0:
    print("=" * 80)
    print("Multi-Host Statistics Test for normalize_adapter")
    print("=" * 80)
    print(f"JAX Distributed Setup:")
    print(f"  Total processes: {num_processes}")
    print(f"  Local device count: {jax.local_device_count()}")
    print(f"  Global device count: {len(jax.devices())}")
    print("=" * 80)
    print()


def _gather_and_reduce(x: np.ndarray, op: str) -> np.ndarray:
    """Multi-host gather and reduce operation."""
    if jax.process_count() == 1:
        return x
    xs = mh.process_allgather(np.asarray(x), tiled=False)  # shape: [P, ...]
    xs = np.asarray(xs)
    if op == "sum":
        return xs.sum(axis=0)
    if op == "min":
        return xs.min(axis=0)
    if op == "max":
        return xs.max(axis=0)
    raise ValueError(f"Unsupported op: {op}")


def test_basic_gathering():
    """Test basic multi-host gathering operations."""
    if process_id == 0:
        print("Test 1: Basic multi-host gathering")
        print(f"  Testing with {num_processes} processes")

    # Each process has different scalar value
    local_scalar = np.array(process_id + 10, dtype=np.int64)
    result_sum = _gather_and_reduce(local_scalar, "sum")
    expected_sum = sum(range(10, 10 + num_processes))

    if process_id == 0:
        print(f"  Sum result: {result_sum}, expected: {expected_sum}")
        print(f"  ✅ PASS" if result_sum == expected_sum else f"  ❌ FAIL")

    # Test array operations
    local_arr = np.array([float(process_id), float(process_id + 1), float(process_id + 2)])
    result_min = _gather_and_reduce(local_arr, "min")
    result_max = _gather_and_reduce(local_arr, "max")

    if process_id == 0:
        print(f"  Array min: {result_min}, expected: [0. 1. 2.]")
        print(f"  Array max: {result_max}, expected: [{num_processes-1}. {num_processes}. {num_processes+1}.]")
        print()


def test_statistics_old_vs_new():
    """Test OLD (unstable) vs NEW (stable) variance calculation across hosts."""
    if process_id == 0:
        print("Test 2: OLD vs NEW variance calculation on real multi-host data")
        print(f"  Each of {num_processes} processes has different data")

    # Each process generates data with different seed
    np.random.seed(42 + process_id)

    # Different sample sizes per process
    sample_counts = [1000, 1500, 800, 1200, 900, 1100, 950, 1050]
    local_n = sample_counts[process_id % len(sample_counts)]

    # Generate data with known distribution
    local_data = np.random.randn(local_n, 7) * 2.0 + 5.0

    if process_id == 0:
        print(f"  Process {process_id} has {local_n} samples")

    # ===== OLD METHOD: Direct aggregation =====
    local_sum_old = local_data.sum(axis=0)
    local_sumsq_old = np.square(local_data).sum(axis=0)
    local_n_arr = np.array(local_n, dtype=np.int64)

    global_sum_old = _gather_and_reduce(local_sum_old, "sum")
    global_sumsq_old = _gather_and_reduce(local_sumsq_old, "sum")
    global_n = int(_gather_and_reduce(local_n_arr, "sum"))

    old_mean = global_sum_old / global_n
    old_var = global_sumsq_old / global_n - np.square(old_mean)
    old_std = np.sqrt(np.maximum(old_var, 0.0))

    # ===== NEW METHOD: Shifted statistics =====
    local_min = local_data.min(axis=0)
    local_max = local_data.max(axis=0)

    global_min = _gather_and_reduce(local_min, "min")
    global_max = _gather_and_reduce(local_max, "max")
    shift = (global_min + global_max) / 2.0

    shifted_data = local_data - shift
    local_sum_new = shifted_data.sum(axis=0)
    local_sumsq_new = np.square(shifted_data).sum(axis=0)

    global_sum_new = _gather_and_reduce(local_sum_new, "sum")
    global_sumsq_new = _gather_and_reduce(local_sumsq_new, "sum")

    shifted_mean = global_sum_new / global_n
    new_mean = shift + shifted_mean
    new_var = global_sumsq_new / global_n - np.square(shifted_mean)
    new_std = np.sqrt(np.maximum(new_var, 0.0))

    if process_id == 0:
        print(f"  Total samples: {global_n}")
        print(f"  Global min/max: {global_min} / {global_max}")
        print(f"  Shift: {shift}")
        print()
        print(f"  OLD method mean: {old_mean}")
        print(f"  NEW method mean: {new_mean}")
        print(f"  Mean difference: {np.abs(old_mean - new_mean).max():.2e}")
        print()
        print(f"  OLD method std: {old_std}")
        print(f"  NEW method std: {new_std}")
        print(f"  Std difference: {np.abs(old_std - new_std).max():.2e}")
        print()
        if np.abs(old_mean - new_mean).max() < 1e-10 and np.abs(old_std - new_std).max() < 1e-10:
            print("  ✅ PASS: Both methods agree (good for normal distributions)")
        else:
            print("  ⚠️  Methods differ (expected for extreme distributions)")
        print()


def test_extreme_case():
    """Test with large mean + small variance to trigger numerical issues."""
    if process_id == 0:
        print("Test 3: Extreme case - Large mean (1e6) + tiny variance (1e-10)")
        print("  Testing BOTH float64 (test default) and float32 (real data)")

    np.random.seed(100 + process_id)

    # Each process has small dataset with large mean, tiny std
    local_n = 500
    local_data_f64 = np.random.randn(local_n, 5) * 1e-10 + 1e6

    # CRITICAL: Test with float32 like real datasets!
    local_data = local_data_f64.astype(np.float32)

    # ===== OLD METHOD =====
    local_sum_old = local_data.sum(axis=0)
    local_sumsq_old = np.square(local_data).sum(axis=0)
    local_n_arr = np.array(local_n, dtype=np.int64)

    global_sum_old = _gather_and_reduce(local_sum_old, "sum")
    global_sumsq_old = _gather_and_reduce(local_sumsq_old, "sum")
    global_n = int(_gather_and_reduce(local_n_arr, "sum"))

    old_mean = global_sum_old / global_n
    old_var = global_sumsq_old / global_n - np.square(old_mean)
    old_std = np.sqrt(np.maximum(old_var, 0.0))

    # ===== NEW METHOD =====
    local_min = local_data.min(axis=0)
    local_max = local_data.max(axis=0)

    global_min = _gather_and_reduce(local_min, "min")
    global_max = _gather_and_reduce(local_max, "max")
    shift = (global_min + global_max) / 2.0

    shifted_data = local_data - shift
    local_sum_new = shifted_data.sum(axis=0)
    local_sumsq_new = np.square(shifted_data).sum(axis=0)

    global_sum_new = _gather_and_reduce(local_sum_new, "sum")
    global_sumsq_new = _gather_and_reduce(local_sumsq_new, "sum")

    shifted_mean = global_sum_new / global_n
    new_mean = shift + shifted_mean
    new_var = global_sumsq_new / global_n - np.square(shifted_mean)
    new_std = np.sqrt(np.maximum(new_var, 0.0))

    if process_id == 0:
        print(f"  Total samples: {global_n}")
        print(f"  Data dtype: {local_data.dtype}")
        print(f"  Expected: mean ≈ 1e6, std ≈ 1e-10")
        print()
        print(f"  OLD method mean: {old_mean}")
        print(f"  OLD method std:  {old_std}")
        print()
        print(f"  NEW method mean: {new_mean}")
        print(f"  NEW method std:  {new_std}")
        print()

        # Check if any dimension has std=0 (the actual bug!)
        old_has_zero = (old_std == 0).any()
        new_has_zero = (new_std == 0).any()

        if old_has_zero:
            print(f"  ❌ OLD METHOD FAILED: {(old_std == 0).sum()}/{len(old_std)} dims have std=0 (catastrophic cancellation!)")
        else:
            print(f"  ✅ OLD METHOD: All dims have std > 0")

        if new_has_zero:
            print(f"  ❌ NEW METHOD FAILED: {(new_std == 0).sum()}/{len(new_std)} dims have std=0 (float32 precision loss!)")
        else:
            print(f"  ✅ NEW METHOD: All dims have std > 0")

        # Show actual std values to diagnose
        print(f"  NEW std range: [{new_std.min():.2e}, {new_std.max():.2e}]")
        print()


def test_realistic_dataset_pattern():
    """Test with pattern matching real dataset statistics calculation."""
    if process_id == 0:
        print("Test 4: Realistic dataset pattern (trajectories + concatenation + filtering)")

    np.random.seed(200 + process_id)

    # Simulate multiple trajectories per process (like the real code)
    num_trajs = 10 + process_id * 2
    trajectories = []

    # Generate trajectories with float32 (like TensorFlow datasets)
    for _ in range(num_trajs):
        traj_len = np.random.randint(50, 200)
        # Realistic action magnitudes: joint velocities might be [-2, 2]
        # But we'll test with larger magnitudes to stress-test
        traj_data = (np.random.randn(traj_len, 7).astype(np.float32) * 0.001 + 5.0).astype(np.float32)

        # Add some NaN/Inf to test filtering (like real data might have)
        if np.random.rand() < 0.3:
            traj_data[np.random.randint(0, traj_len), np.random.randint(0, 7)] = np.nan

        trajectories.append(traj_data)

    # Concatenate all trajectories (line 117 in normalize_adapter.py)
    all_data = np.concatenate(trajectories, axis=0)

    # Filter non-finite values (lines 118-119)
    mask = np.isfinite(all_data).all(axis=1)
    filtered_data = all_data[mask]

    if process_id == 0:
        print(f"  Process {process_id}: {num_trajs} trajs, {all_data.shape[0]} steps -> {filtered_data.shape[0]} after filtering")

    # Now compute statistics with the NEW (shifted) method
    local_min = filtered_data.min(axis=0)
    local_max = filtered_data.max(axis=0)

    global_min = _gather_and_reduce(local_min, "min")
    global_max = _gather_and_reduce(local_max, "max")
    shift = (global_min + global_max) / 2.0

    shifted_data = filtered_data - shift
    local_sum = shifted_data.sum(axis=0)
    local_sumsq = np.square(shifted_data).sum(axis=0)
    local_n = np.array(filtered_data.shape[0], dtype=np.int64)

    global_sum = _gather_and_reduce(local_sum, "sum")
    global_sumsq = _gather_and_reduce(local_sumsq, "sum")
    global_n = int(_gather_and_reduce(local_n, "sum"))

    shifted_mean = global_sum / global_n
    final_mean = shift + shifted_mean
    final_var = global_sumsq / global_n - np.square(shifted_mean)
    final_std = np.sqrt(np.maximum(final_var, 0.0))

    if process_id == 0:
        print(f"  Total samples: {global_n}")
        print(f"  Data dtype: {filtered_data.dtype}")
        print(f"  Global range: [{global_min.min():.6f}, {global_max.max():.6f}]")
        print(f"  Shift: {shift}")
        print(f"  Mean: {final_mean}")
        print(f"  Std:  {final_std}")
        print()

        has_zero = (final_std == 0).any()
        if has_zero:
            zero_dims = np.where(final_std == 0)[0]
            print(f"  ❌ FAILED: {len(zero_dims)} dims have std=0: {zero_dims}")
            print(f"      This should NOT happen with random data!")
        else:
            print(f"  ✅ PASSED: All dims have std > 0")
            print(f"  Std range: [{final_std.min():.2e}, {final_std.max():.2e}]")
        print()


def test_float32_precision_limit():
    """Test the EXACT failure case: float32 with large magnitude + tiny relative variation."""
    if process_id == 0:
        print("Test 5: Float32 precision limit (the ACTUAL bug)")
        print("  Scenario: values around 100.0 with std around 0.0001")
        print("  This is within float32 representable range, but variance calc can fail")

    np.random.seed(300 + process_id)

    # Generate data that SHOULD be representable in float32
    # Mean around 100, std around 0.0001
    # Coefficient of variation: 0.0001/100 = 1e-6 (should be ok for float32)
    local_n = 1000
    local_data_f64 = np.random.randn(local_n, 7) * 0.0001 + 100.0
    local_data = local_data_f64.astype(np.float32)

    # Compute stats with NEW shifted method in FLOAT32
    local_min = local_data.min(axis=0)
    local_max = local_data.max(axis=0)

    global_min = _gather_and_reduce(local_min, "min")
    global_max = _gather_and_reduce(local_max, "max")
    shift = (global_min + global_max) / 2.0

    shifted_data = local_data - shift
    local_sum = shifted_data.sum(axis=0)
    local_sumsq = np.square(shifted_data).sum(axis=0)
    local_n_arr = np.array(local_n, dtype=np.int64)

    global_sum = _gather_and_reduce(local_sum, "sum")
    global_sumsq = _gather_and_reduce(local_sumsq, "sum")
    global_n = int(_gather_and_reduce(local_n_arr, "sum"))

    shifted_mean = global_sum / global_n
    mean_f32 = shift + shifted_mean
    var_f32 = global_sumsq / global_n - np.square(shifted_mean)
    std_f32 = np.sqrt(np.maximum(var_f32, 0.0))

    # Now compute with FLOAT64 for comparison
    local_data_f64_shifted = local_data_f64 - shift.astype(np.float64)
    local_sum_f64 = local_data_f64_shifted.sum(axis=0)
    local_sumsq_f64 = np.square(local_data_f64_shifted).sum(axis=0)

    global_sum_f64 = _gather_and_reduce(local_sum_f64, "sum")
    global_sumsq_f64 = _gather_and_reduce(local_sumsq_f64, "sum")

    shifted_mean_f64 = global_sum_f64 / global_n
    mean_f64 = shift.astype(np.float64) + shifted_mean_f64
    var_f64 = global_sumsq_f64 / global_n - np.square(shifted_mean_f64)
    std_f64 = np.sqrt(np.maximum(var_f64, 0.0))

    if process_id == 0:
        print(f"  Total samples: {global_n}")
        print(f"  Expected: mean ≈ 100.0, std ≈ 0.0001")
        print()
        print(f"  Float32 mean: {mean_f32}")
        print(f"  Float32 std:  {std_f32}")
        print()
        print(f"  Float64 mean: {mean_f64}")
        print(f"  Float64 std:  {std_f64}")
        print()

        has_zero_f32 = (std_f32 == 0).any()
        has_zero_f64 = (std_f64 == 0).any()

        if has_zero_f32:
            zero_dims = np.where(std_f32 == 0)[0]
            print(f"  ❌ FLOAT32 FAILED: {len(zero_dims)} dims have std=0: {zero_dims}")
            print(f"      This is the BUG in normalize_adapter.py!")
        else:
            print(f"  ✅ Float32: All dims have std > 0")
            print(f"      Std range: [{std_f32.min():.2e}, {std_f32.max():.2e}]")

        if has_zero_f64:
            print(f"  ❌ Float64 also failed (unexpected!)")
        else:
            print(f"  ✅ Float64: All dims have std > 0")
            print(f"      Std range: [{std_f64.min():.2e}, {std_f64.max():.2e}]")

        print()
        print(f"  📊 Relative error in std: {(np.abs(std_f32 - std_f64) / (std_f64 + 1e-15)).max():.2e}")
        print()


def test_constant_dimensions():
    """Test with some dimensions being truly constant (common in real datasets)."""
    if process_id == 0:
        print("Test 6: Constant dimensions (gripper always closed, joint stuck, etc.)")

    np.random.seed(400 + process_id)

    # Generate data where some dimensions are constant
    local_n = 1000
    local_data = np.random.randn(local_n, 7).astype(np.float32) * 0.1 + 5.0

    # Make dimensions 2, 4, 6 EXACTLY constant across all processes
    local_data[:, 2] = 100.0  # Constant at 100.0
    local_data[:, 4] = 0.0    # Constant at 0.0
    local_data[:, 6] = -50.5  # Constant at -50.5

    # Compute statistics
    local_min = local_data.min(axis=0)
    local_max = local_data.max(axis=0)

    global_min = _gather_and_reduce(local_min, "min")
    global_max = _gather_and_reduce(local_max, "max")
    shift = (global_min + global_max) / 2.0

    shifted_data = local_data - shift
    local_sum = shifted_data.sum(axis=0)
    local_sumsq = np.square(shifted_data).sum(axis=0)
    local_n_arr = np.array(local_n, dtype=np.int64)

    global_sum = _gather_and_reduce(local_sum, "sum")
    global_sumsq = _gather_and_reduce(local_sumsq, "sum")
    global_n = int(_gather_and_reduce(local_n_arr, "sum"))

    shifted_mean = global_sum / global_n
    final_mean = shift + shifted_mean
    final_var = global_sumsq / global_n - np.square(shifted_mean)
    final_std = np.sqrt(np.maximum(final_var, 0.0))

    if process_id == 0:
        print(f"  Total samples: {global_n}")
        print(f"  Dimensions 2, 4, 6 are constant (should have std=0)")
        print()
        print(f"  Mean: {final_mean}")
        print(f"  Std:  {final_std}")
        print()

        # Check which dimensions have std=0
        zero_dims = np.where(final_std == 0)[0]
        expected_zero_dims = {2, 4, 6}

        if set(zero_dims) == expected_zero_dims:
            print(f"  ✅ CORRECT: Dims {zero_dims} have std=0 (as expected for constant data)")
        elif len(zero_dims) > len(expected_zero_dims):
            unexpected = set(zero_dims) - expected_zero_dims
            print(f"  ❌ FAILED: Dims {list(unexpected)} have std=0 but should vary!")
            print(f"      This would be the BUG!")
        else:
            missing = expected_zero_dims - set(zero_dims)
            print(f"  ⚠️  Dims {list(missing)} should be constant but have std > 0")

        # Show varying dimensions
        varying_dims = [i for i in range(7) if i not in expected_zero_dims]
        print(f"  Varying dims {varying_dims} std: {final_std[varying_dims]}")
        print()


def test_quantized_values():
    """Test with quantized/integer-like values (e.g., sensor readings 0-255)."""
    if process_id == 0:
        print("Test 7: Quantized values (integer sensor readings cast to float32)")
        print("  Simulating values that look like: 123.0, 124.0, 125.0 (no decimals)")

    np.random.seed(500 + process_id)

    # Generate quantized data: integers cast to float32
    local_n = 1000
    # Integers between 100-200, then cast to float32
    local_data = np.random.randint(100, 200, size=(local_n, 7)).astype(np.float32)

    # Add one dimension that's almost constant (varies by only 1-2 integers)
    local_data[:, 3] = np.random.choice([150.0, 151.0], size=local_n).astype(np.float32)

    # Compute statistics
    local_min = local_data.min(axis=0)
    local_max = local_data.max(axis=0)

    global_min = _gather_and_reduce(local_min, "min")
    global_max = _gather_and_reduce(local_max, "max")
    shift = (global_min + global_max) / 2.0

    shifted_data = local_data - shift
    local_sum = shifted_data.sum(axis=0)
    local_sumsq = np.square(shifted_data).sum(axis=0)
    local_n_arr = np.array(local_n, dtype=np.int64)

    global_sum = _gather_and_reduce(local_sum, "sum")
    global_sumsq = _gather_and_reduce(local_sumsq, "sum")
    global_n = int(_gather_and_reduce(local_n_arr, "sum"))

    shifted_mean = global_sum / global_n
    final_mean = shift + shifted_mean
    final_var = global_sumsq / global_n - np.square(shifted_mean)
    final_std = np.sqrt(np.maximum(final_var, 0.0))

    if process_id == 0:
        print(f"  Total samples: {global_n}")
        print(f"  Data dtype: {local_data.dtype}")
        print(f"  Value range: [{global_min.min():.1f}, {global_max.max():.1f}]")
        print()
        print(f"  Mean: {final_mean}")
        print(f"  Std:  {final_std}")
        print()

        has_zero = (final_std == 0).any()
        if has_zero:
            zero_dims = np.where(final_std == 0)[0]
            print(f"  ⚠️  Dims {zero_dims} have std=0")
            print(f"      Check if these dimensions are truly constant in your dataset!")
        else:
            print(f"  ✅ All dims have std > 0")
            print(f"      Std range: [{final_std.min():.2e}, {final_std.max():.2e}]")
        print()


def test_nearly_constant_float32():
    """Test the edge case: values that vary slightly but round to same float32."""
    if process_id == 0:
        print("Test 8: Nearly-constant values with float32 rounding")
        print("  Values that differ by less than float32 epsilon")

    np.random.seed(600 + process_id)

    # Generate values that are "constant + tiny noise"
    # Float32 has ~7 decimal digits of precision
    # So at magnitude 1000, smallest representable difference is ~1e-4
    base_value = 1000.0
    local_n = 1000

    # Generate in float64, then cast to float32
    local_data_f64 = np.ones((local_n, 7)) * base_value
    # Add noise that's below float32 resolution
    local_data_f64 += np.random.randn(local_n, 7) * 1e-5  # 10x smaller than float32 epsilon at this magnitude

    local_data = local_data_f64.astype(np.float32)

    # Check how many unique values we have after float32 conversion
    unique_counts = [len(np.unique(local_data[:, i])) for i in range(7)]

    # Compute statistics
    local_min = local_data.min(axis=0)
    local_max = local_data.max(axis=0)

    global_min = _gather_and_reduce(local_min, "min")
    global_max = _gather_and_reduce(local_max, "max")
    shift = (global_min + global_max) / 2.0

    shifted_data = local_data - shift
    local_sum = shifted_data.sum(axis=0)
    local_sumsq = np.square(shifted_data).sum(axis=0)
    local_n_arr = np.array(local_n, dtype=np.int64)

    global_sum = _gather_and_reduce(local_sum, "sum")
    global_sumsq = _gather_and_reduce(local_sumsq, "sum")
    global_n = int(_gather_and_reduce(local_n_arr, "sum"))

    shifted_mean = global_sum / global_n
    final_mean = shift + shifted_mean
    final_var = global_sumsq / global_n - np.square(shifted_mean)
    final_std = np.sqrt(np.maximum(final_var, 0.0))

    if process_id == 0:
        print(f"  Total samples: {global_n}")
        print(f"  Base value: {base_value}, noise: ±1e-5")
        print(f"  Unique values per dim after float32: {unique_counts}")
        print()
        print(f"  Mean: {final_mean}")
        print(f"  Std:  {final_std}")
        print()

        has_zero = (final_std == 0).any()
        if has_zero:
            zero_dims = np.where(final_std == 0)[0]
            print(f"  ❌ FAILURE: Dims {zero_dims} have std=0 due to float32 rounding!")
            print(f"      These dimensions collapsed to constant after float32 conversion")
            print(f"      This is the ACTUAL BUG you're seeing!")
        else:
            print(f"  ✅ All dims have std > 0")
            print(f"      Std range: [{final_std.min():.2e}, {final_std.max():.2e}]")
        print()


if __name__ == "__main__":
    test_basic_gathering()
    test_statistics_old_vs_new()
    test_extreme_case()
    test_realistic_dataset_pattern()
    test_float32_precision_limit()
    test_constant_dimensions()
    test_quantized_values()
    test_nearly_constant_float32()

    if process_id == 0:
        print("=" * 80)
        print("All multi-host tests complete!")
        print("=" * 80)
