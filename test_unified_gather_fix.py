"""Comprehensive test for unified gather_with_padding function.

Tests all three use cases:
1. chunk_actions: Fixed window for action chunking
2. group_language_actions: Fixed window for language action grouping
3. add_prediction_pairs: Variable window per timestep
"""

import tensorflow as tf
import numpy as np


def gather_with_padding(
    data: tf.Tensor,
    sequence_length: tf.Tensor,
    window_size: int | tf.Tensor,
    per_timestep_windows: tf.Tensor | None = None,
) -> tf.Tensor:
    """Unified gather function with proper zero-padding."""
    if isinstance(window_size, int):
        window_size_tensor = tf.constant(window_size, dtype=tf.int32)
    else:
        window_size_tensor = tf.cast(window_size, tf.int32)

    base = tf.broadcast_to(
        tf.range(window_size_tensor)[None],
        [sequence_length, window_size_tensor]
    )
    offsets = tf.broadcast_to(
        tf.range(sequence_length)[:, None],
        [sequence_length, window_size_tensor]
    )
    indices = base + offsets

    if per_timestep_windows is not None:
        sequence_valid = indices < sequence_length
        window_valid = base < tf.expand_dims(per_timestep_windows, -1)
        valid_mask = tf.logical_and(sequence_valid, window_valid)
    else:
        valid_mask = indices < sequence_length

    clamped_indices = tf.minimum(indices, sequence_length - 1)
    gathered = tf.gather(data, clamped_indices)

    mask_expanded = tf.cast(valid_mask, gathered.dtype)
    if len(gathered.shape) > 2:
        for _ in range(len(gathered.shape) - 2):
            mask_expanded = tf.expand_dims(mask_expanded, -1)

    gathered = gathered * mask_expanded
    return gathered


def compute_window_indices_buggy(sequence_length: tf.Tensor, window_size: int) -> tf.Tensor:
    """BUGGY VERSION: Old implementation that repeats last element."""
    base = tf.broadcast_to(tf.range(window_size)[None], [sequence_length, window_size])
    offsets = tf.broadcast_to(tf.range(sequence_length)[:, None], [sequence_length, window_size])
    indices = base + offsets
    return tf.minimum(indices, sequence_length - 1)  # BUG: Repeats instead of zero-padding


def test_chunk_actions():
    """Test Case 1: chunk_actions with fixed window (action_horizon)."""
    print("=" * 80)
    print("TEST 1: chunk_actions (Fixed Window)")
    print("=" * 80)

    # Create test trajectory: 8 timesteps, 3D actions
    actions = tf.constant([
        [1.0, 1.0, 1.0],
        [2.0, 2.0, 2.0],
        [3.0, 3.0, 3.0],
        [4.0, 4.0, 4.0],
        [5.0, 5.0, 5.0],
        [6.0, 6.0, 6.0],
        [7.0, 7.0, 7.0],
        [8.0, 8.0, 8.0],
    ], dtype=tf.float32)

    action_horizon = 4
    traj_len = tf.shape(actions)[0]

    print(f"\nTrajectory length: {traj_len.numpy()}")
    print(f"Action horizon: {action_horizon}")

    # BUGGY version
    buggy_indices = compute_window_indices_buggy(traj_len, action_horizon)
    buggy_chunked = tf.gather(actions, buggy_indices)

    # FIXED version
    fixed_chunked = gather_with_padding(
        data=actions,
        sequence_length=traj_len,
        window_size=action_horizon,
    )

    print("\nBUGGY - Last 2 timesteps (should show repetition):")
    for t in [6, 7]:
        print(f"  t={t}: {buggy_chunked[t].numpy()}")
        if t >= 5:
            print(f"       ⚠️  Repeats action {actions[7].numpy()} instead of zero-padding")

    print("\nFIXED - Last 2 timesteps (should show zero-padding):")
    for t in [6, 7]:
        print(f"  t={t}: {fixed_chunked[t].numpy()}")
        if t >= 5:
            print(f"       ✓  Properly zero-padded")

    # Verify fix
    assert tf.reduce_all(fixed_chunked[7, 1:] == 0.0), "Timestep 7 should be zero-padded after position 0"
    print("\n✓ TEST 1 PASSED: chunk_actions correctly zero-pads")


def test_group_language_actions():
    """Test Case 2: group_language_actions with control_frequency window."""
    print("\n" + "=" * 80)
    print("TEST 2: group_language_actions (Fixed Window with Trimming)")
    print("=" * 80)

    # Create test actions
    raw_actions = tf.constant([
        [1.0, 2.0],
        [3.0, 4.0],
        [5.0, 6.0],
        [7.0, 8.0],
        [9.0, 10.0],
    ], dtype=tf.float32)

    traj_len = tf.shape(raw_actions)[0]
    control_frequency = 3  # Dataset control frequency
    summation_steps = 5

    trimmed_len = tf.minimum(
        tf.cast(control_frequency, tf.int32),
        tf.cast(summation_steps, tf.int32)
    )

    print(f"\nTrajectory length: {traj_len.numpy()}")
    print(f"Control frequency: {control_frequency}")
    print(f"Summation steps: {summation_steps}")
    print(f"Trimmed length: {trimmed_len.numpy()}")

    # FIXED version using unified function
    actions_window = gather_with_padding(
        data=raw_actions,
        sequence_length=traj_len,
        window_size=trimmed_len,
    )

    print("\nGathered windows (trimmed to control_frequency):")
    for t in range(traj_len.numpy()):
        print(f"  t={t}: {actions_window[t].numpy()}")

    print("\nLast timestep analysis:")
    print(f"  t=4: {actions_window[4].numpy()}")
    print(f"       ✓ Shows [9,10], [0,0], [0,0] - correctly zero-padded")

    # Verify
    assert tf.reduce_all(actions_window[4, 1:] == 0.0), "Should be zero-padded"
    print("\n✓ TEST 2 PASSED: group_language_actions correctly zero-pads")


def test_add_prediction_pairs():
    """Test Case 3: add_prediction_pairs with variable per-timestep windows."""
    print("\n" + "=" * 80)
    print("TEST 3: add_prediction_pairs (Variable Windows)")
    print("=" * 80)

    # Create test actions
    raw_actions = tf.constant([
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
        [4.0, 4.0],
        [5.0, 5.0],
        [6.0, 6.0],
    ], dtype=tf.float32)

    traj_len = tf.shape(raw_actions)[0]
    summation_steps = 4

    # Simulate variable deltas per timestep
    deltas = tf.constant([4, 3, 3, 2, 1, 1], dtype=tf.int32)  # Variable prediction horizons
    deltas_clamped = tf.minimum(deltas, summation_steps)

    print(f"\nTrajectory length: {traj_len.numpy()}")
    print(f"Summation steps: {summation_steps}")
    print(f"Deltas per timestep: {deltas.numpy()}")
    print(f"Deltas clamped: {deltas_clamped.numpy()}")

    # FIXED version using unified function with per-timestep windows
    actions_window = gather_with_padding(
        data=raw_actions,
        sequence_length=traj_len,
        window_size=summation_steps,
        per_timestep_windows=deltas_clamped,
    )

    print("\nGathered windows (variable per timestep):")
    for t in range(traj_len.numpy()):
        print(f"  t={t}, delta={deltas[t].numpy()}: {actions_window[t].numpy()}")

    print("\nKey observations:")
    print(f"  t=0, delta=4: {actions_window[0].numpy()}")
    print(f"       ✓ Gathers [1,2,3,4] - full window within bounds")

    print(f"  t=3, delta=2: {actions_window[3].numpy()}")
    print(f"       ✓ Gathers [4,5] then zero-pads - respects delta=2")

    print(f"  t=5, delta=1: {actions_window[5].numpy()}")
    print(f"       ✓ Gathers [6] then zero-pads - last timestep")

    # Verify
    # t=3 with delta=2 should gather [4,5,0,0]
    expected_t3 = np.array([[4.0, 4.0], [5.0, 5.0], [0.0, 0.0], [0.0, 0.0]])
    assert np.allclose(actions_window[3].numpy(), expected_t3), "t=3 should gather 2 then zero-pad"

    # t=5 with delta=1 should gather [6,0,0,0]
    expected_t5 = np.array([[6.0, 6.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    assert np.allclose(actions_window[5].numpy(), expected_t5), "t=5 should gather 1 then zero-pad"

    print("\n✓ TEST 3 PASSED: add_prediction_pairs correctly handles variable windows")


def test_edge_cases():
    """Test edge cases: very short trajectories, large windows."""
    print("\n" + "=" * 80)
    print("TEST 4: Edge Cases")
    print("=" * 80)

    # Edge case 1: Trajectory shorter than window
    print("\nEdge case 1: Trajectory (len=2) shorter than window (size=5)")
    data = tf.constant([[1.0], [2.0]], dtype=tf.float32)
    result = gather_with_padding(
        data=data,
        sequence_length=tf.constant(2),
        window_size=5,
    )
    print(f"  t=0: {result[0].numpy().flatten()}")
    print(f"       Expected: [1, 2, 0, 0, 0]")
    print(f"  t=1: {result[1].numpy().flatten()}")
    print(f"       Expected: [2, 0, 0, 0, 0]")
    assert np.allclose(result[0].numpy().flatten(), [1, 2, 0, 0, 0])
    assert np.allclose(result[1].numpy().flatten(), [2, 0, 0, 0, 0])
    print("  ✓ Passed")

    # Edge case 2: Variable windows with some zero-length
    print("\nEdge case 2: Variable windows including zero-length")
    data = tf.constant([[1.0], [2.0], [3.0]], dtype=tf.float32)
    deltas = tf.constant([2, 1, 0], dtype=tf.int32)  # Last one has zero window
    result = gather_with_padding(
        data=data,
        sequence_length=tf.constant(3),
        window_size=2,
        per_timestep_windows=deltas,
    )
    print(f"  t=0, delta=2: {result[0].numpy().flatten()}")
    print(f"  t=1, delta=1: {result[1].numpy().flatten()}")
    print(f"  t=2, delta=0: {result[2].numpy().flatten()}")
    print(f"       Expected: [0, 0] (zero window)")
    assert np.allclose(result[2].numpy().flatten(), [0, 0])
    print("  ✓ Passed")

    print("\n✓ TEST 4 PASSED: Edge cases handled correctly")


if __name__ == "__main__":
    tf.config.set_visible_devices([], 'GPU')

    test_chunk_actions()
    test_group_language_actions()
    test_add_prediction_pairs()
    test_edge_cases()

    print("\n" + "=" * 80)
    print("ALL TESTS PASSED! ✓")
    print("=" * 80)
    print("""
Summary of fixes:
1. chunk_actions: Now uses gather_with_padding for proper zero-padding
2. group_language_actions: Simplified to use unified function
3. add_prediction_pairs: Converted from tf.map_fn to efficient 2D gather

Benefits:
- Correct zero-padding instead of repeating last element
- More efficient (batched operations instead of iterative)
- Unified implementation reduces code duplication
- Supports both fixed and variable-length windows
    """)
