"""Test to demonstrate the clamping issue in group_language_actions.

This test compares the BUGGY version (current implementation) with the FIXED version.
"""

import tensorflow as tf
import numpy as np


def compute_window_indices_buggy(sequence_length: tf.Tensor, window_size: int) -> tf.Tensor:
    """BUGGY VERSION: Clamps to last index, causing repetition instead of zero-padding."""
    base = tf.broadcast_to(tf.range(window_size)[None], [sequence_length, window_size])
    offsets = tf.broadcast_to(tf.range(sequence_length)[:, None], [sequence_length, window_size])
    indices = base + offsets
    # BUG: This causes the last element to repeat instead of padding with zeros
    return tf.minimum(indices, sequence_length - 1)


def group_language_actions_buggy(raw_action, summation_steps=10, trimmed_len=10):
    """BUGGY VERSION: Current implementation."""
    traj_len = tf.shape(raw_action)[0]

    # Create indices with clamping
    summation_indices = compute_window_indices_buggy(traj_len, summation_steps)

    # Gather using 2D indices
    actions_window_trim = tf.gather(raw_action, summation_indices[:, :trimmed_len])

    # Pad if needed
    pad_len = summation_steps - trimmed_len

    def _pad_numeric():
        zeros_pad = tf.zeros(
            [tf.shape(actions_window_trim)[0], pad_len, tf.shape(actions_window_trim)[-1]],
            dtype=actions_window_trim.dtype,
        )
        return tf.concat([actions_window_trim, zeros_pad], axis=1)

    actions_window = tf.cond(pad_len > 0, _pad_numeric, lambda: actions_window_trim)

    # Serialize each action
    flat_rows = tf.reshape(actions_window, [-1, tf.shape(actions_window)[-1]])
    serialized_flat = tf.map_fn(
        lambda v: tf.io.serialize_tensor(v),
        flat_rows,
        fn_output_signature=tf.string,
    )
    language_actions = tf.reshape(
        serialized_flat,
        [tf.shape(actions_window)[0], summation_steps],
    )

    return language_actions, actions_window


def group_language_actions_fixed(raw_action, summation_steps=10, trimmed_len=10):
    """FIXED VERSION: Properly pads with zeros instead of repeating last action."""
    traj_len = tf.shape(raw_action)[0]

    # Create indices WITHOUT clamping initially
    base = tf.broadcast_to(tf.range(trimmed_len)[None], [traj_len, trimmed_len])
    offsets = tf.broadcast_to(tf.range(traj_len)[:, None], [traj_len, trimmed_len])
    indices = base + offsets  # [T, trimmed_len], can exceed traj_len - 1

    # Create mask for valid indices
    valid_mask = indices < traj_len  # [T, trimmed_len]

    # Clamp indices only for gathering (to avoid out-of-bounds errors)
    clamped_indices = tf.minimum(indices, traj_len - 1)

    # Gather
    actions_window_trim = tf.gather(raw_action, clamped_indices)  # [T, trimmed_len, A]

    # Zero out invalid elements (this is the key fix!)
    actions_window_trim = actions_window_trim * tf.cast(
        tf.expand_dims(valid_mask, -1), actions_window_trim.dtype
    )

    # Pad if needed
    pad_len = summation_steps - trimmed_len

    def _pad_numeric():
        zeros_pad = tf.zeros(
            [tf.shape(actions_window_trim)[0], pad_len, tf.shape(actions_window_trim)[-1]],
            dtype=actions_window_trim.dtype,
        )
        return tf.concat([actions_window_trim, zeros_pad], axis=1)

    actions_window = tf.cond(pad_len > 0, _pad_numeric, lambda: actions_window_trim)

    # Serialize each action
    flat_rows = tf.reshape(actions_window, [-1, tf.shape(actions_window)[-1]])
    serialized_flat = tf.map_fn(
        lambda v: tf.io.serialize_tensor(v),
        flat_rows,
        fn_output_signature=tf.string,
    )
    language_actions = tf.reshape(
        serialized_flat,
        [tf.shape(actions_window)[0], summation_steps],
    )

    return language_actions, actions_window


def test_clamping_issue():
    """Demonstrate the clamping bug with a concrete example."""

    # Create a small trajectory with 8 timesteps, 3-dimensional actions
    # Use distinctive values so we can see what's being gathered
    raw_action = tf.constant([
        [1.0, 1.0, 1.0],  # t=0
        [2.0, 2.0, 2.0],  # t=1
        [3.0, 3.0, 3.0],  # t=2
        [4.0, 4.0, 4.0],  # t=3
        [5.0, 5.0, 5.0],  # t=4
        [6.0, 6.0, 6.0],  # t=5
        [7.0, 7.0, 7.0],  # t=6
        [8.0, 8.0, 8.0],  # t=7
    ], dtype=tf.float32)

    summation_steps = 5  # Want to gather 5 future steps
    trimmed_len = 5

    print("=" * 80)
    print("TEST: Clamping Issue in group_language_actions")
    print("=" * 80)
    print(f"\nTrajectory length: {raw_action.shape[0]}")
    print(f"Summation steps: {summation_steps}")
    print(f"Action shape: {raw_action.shape}")
    print("\nRaw actions:")
    print(raw_action.numpy())

    # Test BUGGY version
    print("\n" + "=" * 80)
    print("BUGGY VERSION (current implementation)")
    print("=" * 80)
    _, actions_window_buggy = group_language_actions_buggy(raw_action, summation_steps, trimmed_len)
    print("\nActions window shape:", actions_window_buggy.shape)
    print("\nActions window (what gets serialized):")
    actions_buggy_np = actions_window_buggy.numpy()

    for t in range(actions_buggy_np.shape[0]):
        print(f"\nTimestep {t}: should gather actions from t={t} to t={t+4}")
        print(f"  Gathered: {actions_buggy_np[t]}")

        # Highlight the issue at the end
        if t >= 4:
            print(f"  ⚠️  ISSUE: At t={t}, some indices exceed traj_len-1={raw_action.shape[0]-1}")
            print(f"      Instead of zero-padding, it repeats the last action!")

    # Test FIXED version
    print("\n" + "=" * 80)
    print("FIXED VERSION (with proper zero-padding)")
    print("=" * 80)
    _, actions_window_fixed = group_language_actions_fixed(raw_action, summation_steps, trimmed_len)
    print("\nActions window shape:", actions_window_fixed.shape)
    print("\nActions window (what gets serialized):")
    actions_fixed_np = actions_window_fixed.numpy()

    for t in range(actions_fixed_np.shape[0]):
        print(f"\nTimestep {t}: should gather actions from t={t} to t={t+4}")
        print(f"  Gathered: {actions_fixed_np[t]}")

        if t >= 4:
            print(f"  ✓  CORRECT: Properly zero-padded for out-of-bounds indices")

    # Show the difference explicitly
    print("\n" + "=" * 80)
    print("DIFFERENCE BETWEEN BUGGY AND FIXED")
    print("=" * 80)

    for t in range(4, actions_buggy_np.shape[0]):  # Only check timesteps where issue occurs
        print(f"\nTimestep {t}:")
        print(f"  BUGGY:  {actions_buggy_np[t]}")
        print(f"  FIXED:  {actions_fixed_np[t]}")
        diff = np.abs(actions_buggy_np[t] - actions_fixed_np[t]).sum()
        print(f"  Difference: {diff:.2f}")

        # Verify the fix
        expected_valid_count = min(5, raw_action.shape[0] - t)
        actual_nonzero_count = np.count_nonzero(actions_fixed_np[t].sum(axis=-1))
        assert actual_nonzero_count == expected_valid_count, \
            f"Expected {expected_valid_count} non-zero actions, got {actual_nonzero_count}"


def test_edge_cases():
    """Test edge cases where the bug is most apparent."""

    print("\n" + "=" * 80)
    print("EDGE CASE: Very short trajectory (3 timesteps, want 10 summation steps)")
    print("=" * 80)

    raw_action = tf.constant([
        [1.0, 1.0],
        [2.0, 2.0],
        [3.0, 3.0],
    ], dtype=tf.float32)

    summation_steps = 10
    trimmed_len = 10

    print(f"\nTrajectory length: {raw_action.shape[0]}")
    print(f"Summation steps: {summation_steps}")

    _, buggy = group_language_actions_buggy(raw_action, summation_steps, trimmed_len)
    _, fixed = group_language_actions_fixed(raw_action, summation_steps, trimmed_len)

    print("\nBUGGY version - Timestep 0 (should be [1,2,3,0,0,0,0,0,0,0]):")
    print(buggy[0].numpy())

    print("\nFIXED version - Timestep 0 (should be [1,2,3,0,0,0,0,0,0,0]):")
    print(fixed[0].numpy())

    print("\nBUGGY version - Timestep 2 (should be [3,0,0,0,0,0,0,0,0,0]):")
    print(buggy[2].numpy())
    print("  ⚠️  BUGGY: All [3,3,3,...] - repeats last action instead of zero-padding!")

    print("\nFIXED version - Timestep 2 (should be [3,0,0,0,0,0,0,0,0,0]):")
    print(fixed[2].numpy())
    print("  ✓  FIXED: Properly zero-padded!")


if __name__ == "__main__":
    # Disable GPU for this test
    tf.config.set_visible_devices([], 'GPU')

    test_clamping_issue()
    test_edge_cases()

    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80)
    print("""
The bug is confirmed! The current implementation:
1. Uses compute_window_indices which clamps to traj_len-1
2. This causes the last action to be REPEATED instead of zero-padded
3. At timestep t near the end, indices like [t, t+1, ..., traj_len-1, traj_len-1, ...]
   cause the action at traj_len-1 to be gathered multiple times

The fix:
1. Create indices without initial clamping
2. Create a valid_mask to identify which indices are in-bounds
3. Clamp indices only for the gather operation (to avoid errors)
4. Multiply gathered actions by the valid_mask to zero out invalid positions

This ensures proper zero-padding behavior matching the intent of the code.
    """)
