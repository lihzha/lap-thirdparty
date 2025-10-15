"""Test script to validate normalize_adapter statistics computation."""

import jax

jax.distributed.initialize()

from jax.experimental import multihost_utils as mh
import numpy as np


def test_gather_and_reduce():
    """Test the _gather_and_reduce function logic."""

    def _gather_and_reduce(x: np.ndarray, op: str) -> np.ndarray:
        """Copied from normalize_adapter.py"""
        if getattr(jax, "process_count", lambda: 1)() == 1:
            breakpoint()
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

    # Test 1: Scalar gathering
    print("Test 1: Scalar gathering")
    scalar = np.array(5, dtype=np.int64)
    result = _gather_and_reduce(scalar, "sum")
    print(f"  Input scalar: {scalar}, shape: {scalar.shape}")
    print(f"  Result: {result}, shape: {result.shape}")
    print(f"  Expected: {scalar} (single process)")
    print()

    # Test 2: 1D array gathering
    print("Test 2: 1D array gathering")
    arr = np.array([1.0, 2.0, 3.0])
    result = _gather_and_reduce(arr, "sum")
    print(f"  Input array: {arr}, shape: {arr.shape}")
    print(f"  Result: {result}, shape: {result.shape}")
    print(f"  Expected: {arr} (single process)")
    print()


def test_variance_calculation():
    """Test variance calculation with known data."""

    print("Test 3: Variance calculation with known statistics")

    # Create synthetic data with known mean and std
    np.random.seed(42)

    # Case 1: Normal distribution
    data1 = np.random.randn(10000, 5) * 2.0 + 5.0  # mean=5, std=2
    true_mean1 = data1.mean(axis=0)
    true_std1 = data1.std(axis=0)

    # Compute using the two-pass formula (what the code uses)
    sum1 = data1.sum(axis=0)
    sumsq1 = np.square(data1).sum(axis=0)
    n1 = data1.shape[0]

    computed_mean1 = sum1 / n1
    computed_var1 = sumsq1 / n1 - np.square(computed_mean1)
    computed_std1 = np.sqrt(np.maximum(computed_var1, 0.0))

    # NEW: Compute using stable shifted formula
    shift1 = (data1.min(axis=0) + data1.max(axis=0)) / 2.0
    shifted1 = data1 - shift1
    shifted_sum1 = shifted1.sum(axis=0)
    shifted_sumsq1 = np.square(shifted1).sum(axis=0)
    shifted_mean1 = shifted_sum1 / n1
    stable_mean1 = shift1 + shifted_mean1
    stable_var1 = shifted_sumsq1 / n1 - np.square(shifted_mean1)
    stable_std1 = np.sqrt(np.maximum(stable_var1, 0.0))

    print("  Case 1: Normal distribution (mean=5, std=2)")
    print(f"    True mean: {true_mean1}")
    print(f"    Old computed mean: {computed_mean1}, error: {np.abs(true_mean1 - computed_mean1).max():.2e}")
    print(f"    New stable mean: {stable_mean1}, error: {np.abs(true_mean1 - stable_mean1).max():.2e}")
    print(f"    True std: {true_std1}")
    print(f"    Old computed std: {computed_std1}, error: {np.abs(true_std1 - computed_std1).max():.2e}")
    print(f"    New stable std: {stable_std1}, error: {np.abs(true_std1 - stable_std1).max():.2e}")
    print()

    # Case 2: Constant dimension (should have std=0)
    data2 = np.random.randn(10000, 5) * 2.0 + 5.0
    data2[:, 2] = 3.14159  # Make one dimension constant

    true_mean2 = data2.mean(axis=0)
    true_std2 = data2.std(axis=0)

    sum2 = data2.sum(axis=0)
    sumsq2 = np.square(data2).sum(axis=0)
    n2 = data2.shape[0]

    computed_mean2 = sum2 / n2
    computed_var2 = sumsq2 / n2 - np.square(computed_mean2)
    computed_std2 = np.sqrt(np.maximum(computed_var2, 0.0))

    print("  Case 2: With constant dimension (dim 2)")
    print(f"    True std: {true_std2}")
    print(f"    Computed std: {computed_std2}")
    print(f"    Constant dim (index 2) std: {computed_std2[2]} (should be ~0)")
    print()

    # Case 3: Very small variance (numerical precision test)
    data3 = np.random.randn(10000, 5) * 1e-10 + 1e6  # Large mean, tiny std
    true_mean3 = data3.mean(axis=0)
    true_std3 = data3.std(axis=0)

    sum3 = data3.sum(axis=0)
    sumsq3 = np.square(data3).sum(axis=0)
    n3 = data3.shape[0]

    computed_mean3 = sum3 / n3
    computed_var3 = sumsq3 / n3 - np.square(computed_mean3)
    computed_std3 = np.sqrt(np.maximum(computed_var3, 0.0))

    # NEW: Stable shifted version
    shift3 = (data3.min(axis=0) + data3.max(axis=0)) / 2.0
    shifted3 = data3 - shift3
    shifted_sum3 = shifted3.sum(axis=0)
    shifted_sumsq3 = np.square(shifted3).sum(axis=0)
    shifted_mean3 = shifted_sum3 / n3
    stable_mean3 = shift3 + shifted_mean3
    stable_var3 = shifted_sumsq3 / n3 - np.square(shifted_mean3)
    stable_std3 = np.sqrt(np.maximum(stable_var3, 0.0))

    print("  Case 3: Numerical precision (mean=1e6, std=1e-10) *** CRITICAL TEST ***")
    print(f"    True std: {true_std3}")
    print(f"    OLD method std: {computed_std3}")
    print(f"    NEW method std: {stable_std3}")
    print(f"    Old relative error: {np.abs(true_std3 - computed_std3) / (true_std3 + 1e-15)}")
    print(f"    New relative error: {np.abs(true_std3 - stable_std3) / (true_std3 + 1e-15)}")
    if computed_std3.max() == 0:
        print("    ❌ OLD METHOD: FAILED - returns std=0 due to catastrophic cancellation!")
    if stable_std3.max() > 0 and (np.abs(true_std3 - stable_std3) / true_std3).max() < 0.1:
        print("    ✅ NEW METHOD: PASSED - accurate result with shifted statistics!")
    print()


def test_multi_host_simulation():
    """Simulate multi-host statistics aggregation."""

    print("Test 4: Multi-host statistics simulation")

    np.random.seed(42)

    # Simulate 4 hosts with different amounts of data
    host_data = [
        np.random.randn(1000, 5) * 2.0 + 5.0,  # Host 0
        np.random.randn(1500, 5) * 2.0 + 5.0,  # Host 1
        np.random.randn(800, 5) * 2.0 + 5.0,  # Host 2
        np.random.randn(1200, 5) * 2.0 + 5.0,  # Host 3
    ]

    # Ground truth: compute stats on combined data
    all_data = np.concatenate(host_data, axis=0)
    true_mean = all_data.mean(axis=0)
    true_std = all_data.std(axis=0)

    # Simulate multi-host aggregation
    host_sums = [data.sum(axis=0) for data in host_data]
    host_sumsqs = [np.square(data).sum(axis=0) for data in host_data]
    host_ns = [data.shape[0] for data in host_data]

    # Aggregate
    global_sum = sum(host_sums)
    global_sumsq = sum(host_sumsqs)
    global_n = sum(host_ns)

    computed_mean = global_sum / global_n
    computed_var = global_sumsq / global_n - np.square(computed_mean)
    computed_std = np.sqrt(np.maximum(computed_var, 0.0))

    print(f"  Hosts: 4, Samples: {[len(d) for d in host_data]}")
    print(f"  True mean: {true_mean}")
    print(f"  Computed mean: {computed_mean}")
    print(f"  Mean error: {np.abs(true_mean - computed_mean).max()}")
    print(f"  True std: {true_std}")
    print(f"  Computed std: {computed_std}")
    print(f"  Std error: {np.abs(true_std - computed_std).max()}")
    print()


def test_masking_issue():
    """Test the effect of different masks on actions and states."""

    print("Test 5: Different masking for actions and states")

    np.random.seed(42)

    # Create data where actions and states have NaN in different places
    actions = np.random.randn(1000, 7)
    states = np.random.randn(1000, 14)

    # Add NaN to different rows
    actions[10, :] = np.nan  # Row 10 has NaN in actions
    states[20, :] = np.nan  # Row 20 has NaN in states
    actions[30, 5] = np.nan  # Row 30 has partial NaN in actions

    # Apply separate masks (as in the original code)
    action_mask = np.isfinite(actions).all(axis=1)
    state_mask = np.isfinite(states).all(axis=1)

    actions_clean = actions[action_mask]
    states_clean = states[state_mask]

    print(f"  Original samples: {len(actions)}")
    print(f"  After action mask: {len(actions_clean)} (removed {len(actions) - len(actions_clean)})")
    print(f"  After state mask: {len(states_clean)} (removed {len(states) - len(states_clean)})")
    print(f"  Rows with NaN in actions: {np.where(~action_mask)[0]}")
    print(f"  Rows with NaN in states: {np.where(~state_mask)[0]}")
    print()
    print("  WARNING: Actions and states have different sample counts!")
    print("  This means a_n != s_n, which could cause issues if they should be synchronized.")
    print()


def test_edge_cases():
    """Test edge cases that might cause std=0."""

    print("Test 6: Edge cases")

    # Case 1: Truly constant data
    print("  Case 1: Truly constant data")
    const_data = np.ones((1000, 5)) * 3.14
    const_std = const_data.std(axis=0)
    print(f"    Std of constant data: {const_std} (should be all zeros)")
    print()

    # Case 2: Very limited unique values
    print("  Case 2: Limited unique values")
    limited_data = np.random.choice([0.0, 1.0], size=(1000, 5))
    limited_std = limited_data.std(axis=0)
    print(f"    Std of binary data: {limited_std} (should be ~0.5)")
    print()

    # Case 3: Single sample (edge case)
    print("  Case 3: Single sample")
    single_data = np.array([[1.0, 2.0, 3.0]])
    single_std = single_data.std(axis=0)
    print(f"    Std of single sample: {single_std} (should be all zeros)")
    print()


if __name__ == "__main__":
    print("=" * 80)
    print("Testing normalize_adapter statistics computation")
    print("=" * 80)
    print()

    test_gather_and_reduce()
    test_variance_calculation()
    test_multi_host_simulation()
    test_masking_issue()
    test_edge_cases()

    print("=" * 80)
    print("Tests complete!")
    print("=" * 80)
