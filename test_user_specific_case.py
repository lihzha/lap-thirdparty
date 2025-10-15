"""Test to reproduce the user's specific case: mean=0.53, std=0, q01=0.27, q99=0.78"""

import jax
from jax.experimental import multihost_utils as mh
import numpy as np

jax.distributed.initialize()

process_id = jax.process_index()
num_processes = jax.process_count()


def _gather_and_reduce(x: np.ndarray, op: str) -> np.ndarray:
    if jax.process_count() == 1:
        return x
    xs = mh.process_allgather(np.asarray(x), tiled=False)
    xs = np.asarray(xs)
    if op == "sum":
        return xs.sum(axis=0)
    if op == "min":
        return xs.min(axis=0)
    if op == "max":
        return xs.max(axis=0)
    raise ValueError(f"Unsupported op: {op}")


def test_user_case():
    """Reproduce user's exact scenario."""
    if process_id == 0:
        print("=" * 80)
        print("Testing user's specific case: mean=0.53, std=0, q01=0.27, q99=0.78")
        print("=" * 80)
        print()

    np.random.seed(42 + process_id)

    # Generate data that should produce: mean~0.53, range [0.27, 0.78]
    # This is a fairly uniform distribution
    local_n = 1000

    # Each process generates data in the range [0.27, 0.78]
    local_data = np.random.uniform(0.27, 0.78, size=(local_n, 1)).astype(np.float32)

    if process_id == 0:
        print(f"Process {process_id} data sample: {local_data[:5, 0]}")
        print(f"Process {process_id} local stats: mean={local_data.mean():.4f}, std={local_data.std():.4f}")
        print()

    # Now compute statistics EXACTLY as normalize_adapter.py does
    actions = local_data  # Shape: [N, 1]

    # WITHOUT float64 promotion (original buggy code)
    a_min = actions.min(axis=0)
    a_max = actions.max(axis=0)

    a_min = _gather_and_reduce(a_min, "min")
    a_max = _gather_and_reduce(a_max, "max")
    a_shift = (a_min + a_max) / 2.0

    a_shifted = actions - a_shift
    a_sum = a_shifted.sum(axis=0)
    a_sumsq = np.square(a_shifted).sum(axis=0)
    a_n = np.array(actions.shape[0], dtype=np.int64)

    a_sum = _gather_and_reduce(a_sum, "sum")
    a_sumsq = _gather_and_reduce(a_sumsq, "sum")
    a_n = int(_gather_and_reduce(a_n, "sum"))

    a_shifted_mean = a_sum / max(a_n, 1)
    a_mean_f32 = a_shift + a_shifted_mean
    a_var_f32 = a_sumsq / max(a_n, 1) - np.square(a_shifted_mean)
    a_std_f32 = np.sqrt(np.maximum(a_var_f32, 0.0))

    # WITH float64 promotion (fixed code)
    actions_f64 = actions.astype(np.float64)

    a_min = actions_f64.min(axis=0)
    a_max = actions_f64.max(axis=0)

    a_min = _gather_and_reduce(a_min, "min")
    a_max = _gather_and_reduce(a_max, "max")
    a_shift = (a_min + a_max) / 2.0

    a_shifted = actions_f64 - a_shift
    a_sum = a_shifted.sum(axis=0)
    a_sumsq = np.square(a_shifted).sum(axis=0)

    a_sum = _gather_and_reduce(a_sum, "sum")
    a_sumsq = _gather_and_reduce(a_sumsq, "sum")

    a_shifted_mean = a_sum / max(a_n, 1)
    a_mean_f64 = a_shift + a_shifted_mean
    a_var_f64 = a_sumsq / max(a_n, 1) - np.square(a_shifted_mean)
    a_std_f64 = np.sqrt(np.maximum(a_var_f64, 0.0))

    if process_id == 0:
        print(f"Total samples: {a_n}")
        print(f"Global min/max: {a_min[0]:.4f} / {a_max[0]:.4f}")
        print()
        print("WITHOUT FIX (float32):")
        print(f"  Mean: {a_mean_f32[0]:.4f}")
        print(f"  Var:  {a_var_f32[0]:.6f}")
        print(f"  Std:  {a_std_f32[0]:.6f}")
        if a_var_f32[0] < 0:
            print(f"  ⚠️  NEGATIVE VARIANCE! This gets clamped to 0!")
        if a_std_f32[0] == 0:
            print(f"  ❌ std=0 (BUG reproduced!)")
        print()
        print("WITH FIX (float64):")
        print(f"  Mean: {a_mean_f64[0]:.4f}")
        print(f"  Var:  {a_var_f64[0]:.6f}")
        print(f"  Std:  {a_std_f64[0]:.6f}")
        if a_std_f64[0] == 0:
            print(f"  ❌ Still std=0 (float64 didn't help!)")
        else:
            print(f"  ✅ Fixed!")
        print()

        # Let's also check the raw variance components
        print("DEBUG INFO:")
        print(f"  Shift: {a_shift[0]:.6f}")
        print(f"  a_sum (float32): {_gather_and_reduce(a_shifted.astype(np.float32).sum(axis=0), 'sum')[0]:.6e}")
        print(f"  a_sumsq (float32): {_gather_and_reduce(np.square(a_shifted.astype(np.float32)).sum(axis=0), 'sum')[0]:.6e}")

        # Manual variance check
        actions_all_f32 = local_data.astype(np.float32)
        actions_all_f64 = local_data.astype(np.float64)

        expected_std_f32 = actions_all_f32.std()
        expected_std_f64 = actions_all_f64.std()

        print(f"  Expected std (numpy float32): {expected_std_f32:.6f}")
        print(f"  Expected std (numpy float64): {expected_std_f64:.6f}")


if __name__ == "__main__":
    test_user_case()
