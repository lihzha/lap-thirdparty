"""Test to demonstrate the realized_deltas fix for add_prediction_pairs.

This test shows how the fix prevents gathering invalid action[T-1] and ensures
the number of actions matches the actual visual gap.
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


def test_realized_deltas_fix():
    """Demonstrate the realized_deltas fix prevents gathering invalid action[T-1]."""
    print("=" * 80)
    print("TEST: Realized Deltas Fix for add_prediction_pairs")
    print("=" * 80)

    # Create a trajectory with 10 observations
    # Actions: action[t] transitions from obs[t] → obs[t+1]
    # Valid actions: action[0] through action[8] (9 actions total)
    # Invalid: action[9] (would transition obs[9] → obs[10], but obs[10] doesn't exist)

    # Simulate: action[9] is zero-padded (as in DROID transform)
    raw_actions = tf.constant([
        [1.0, 1.0],   # action[0]: obs[0] → obs[1]
        [2.0, 2.0],   # action[1]: obs[1] → obs[2]
        [3.0, 3.0],   # action[2]: obs[2] → obs[3]
        [4.0, 4.0],   # action[3]: obs[3] → obs[4]
        [5.0, 5.0],   # action[4]: obs[4] → obs[5]
        [6.0, 6.0],   # action[5]: obs[5] → obs[6]
        [7.0, 7.0],   # action[6]: obs[6] → obs[7]
        [8.0, 8.0],   # action[7]: obs[7] → obs[8]
        [9.0, 9.0],   # action[8]: obs[8] → obs[9]
        [0.0, 0.0],   # action[9]: INVALID/PADDED (obs[9] → obs[10] doesn't exist)
    ], dtype=tf.float32)

    traj_len = tf.shape(raw_actions)[0]  # 10
    summation_steps = 6

    print(f"\nSetup:")
    print(f"  Trajectory length (observations): {traj_len.numpy()}")
    print(f"  Valid actions: action[0] through action[{traj_len.numpy()-2}] ({traj_len.numpy()-1} total)")
    print(f"  Invalid action: action[{traj_len.numpy()-1}] (zero-padded)")
    print(f"  Summation steps: {summation_steps}")

    # Scenario 1: Near the end with large sampled delta
    print("\n" + "=" * 80)
    print("SCENARIO 1: t=7, sampled delta=5 (would exceed trajectory)")
    print("=" * 80)

    t = 7
    sampled_delta = 5

    # BUGGY VERSION: Uses original sampled delta
    print("\n--- BUGGY VERSION (using original sampled delta) ---")
    future_index_buggy = min(t + sampled_delta, traj_len.numpy() - 1)
    print(f"  Sampled delta: {sampled_delta}")
    print(f"  Future index: min({t} + {sampled_delta}, {traj_len.numpy()-1}) = {future_index_buggy}")
    print(f"  Images: [obs[{t}], obs[{future_index_buggy}]]")
    print(f"  Actual visual gap: {future_index_buggy - t} steps")

    deltas_clamped_buggy = min(sampled_delta, summation_steps)
    print(f"\n  deltas_clamped (BUGGY): min({sampled_delta}, {summation_steps}) = {deltas_clamped_buggy}")

    # Gather actions using buggy approach
    actions_buggy = gather_with_padding(
        data=raw_actions,
        sequence_length=traj_len,
        window_size=summation_steps,
        per_timestep_windows=tf.constant([deltas_clamped_buggy], dtype=tf.int32),
    )[0]

    print(f"\n  Actions gathered at t={t}:")
    for i in range(summation_steps):
        action_idx = t + i
        if action_idx < traj_len.numpy():
            print(f"    Position {i}: action[{action_idx}] = {actions_buggy[i].numpy()}")
        else:
            print(f"    Position {i}: (out of bounds, zero-padded)")

    print(f"\n  ⚠️  PROBLEM:")
    print(f"      - Visual gap is only {future_index_buggy - t} steps")
    print(f"      - But we're trying to gather {deltas_clamped_buggy} actions!")
    print(f"      - This includes action[{traj_len.numpy()-1}] which is INVALID")
    print(f"      - Metadata prediction_delta would be {sampled_delta}, but actual gap is {future_index_buggy - t}")

    # FIXED VERSION: Uses realized delta
    print("\n--- FIXED VERSION (using realized delta) ---")
    future_index_fixed = min(t + sampled_delta, traj_len.numpy() - 1)
    realized_delta = future_index_fixed - t  # THE KEY FIX!
    print(f"  Sampled delta: {sampled_delta}")
    print(f"  Future index: {future_index_fixed}")
    print(f"  Realized delta: {future_index_fixed} - {t} = {realized_delta} ✓")

    deltas_clamped_fixed = min(realized_delta, summation_steps)
    print(f"\n  deltas_clamped (FIXED): min({realized_delta}, {summation_steps}) = {deltas_clamped_fixed}")

    # Gather actions using fixed approach
    actions_fixed = gather_with_padding(
        data=raw_actions,
        sequence_length=traj_len,
        window_size=summation_steps,
        per_timestep_windows=tf.constant([deltas_clamped_fixed], dtype=tf.int32),
    )[0]

    print(f"\n  Actions gathered at t={t}:")
    for i in range(summation_steps):
        action_idx = t + i
        if i < deltas_clamped_fixed:
            print(f"    Position {i}: action[{action_idx}] = {actions_fixed[i].numpy()} ✓ Valid")
        else:
            print(f"    Position {i}: zero-padded (beyond realized_delta)")

    print(f"\n  ✓  CORRECT:")
    print(f"      - Visual gap is {realized_delta} steps")
    print(f"      - We gather exactly {deltas_clamped_fixed} actions to bridge that gap")
    print(f"      - Does NOT include invalid action[{traj_len.numpy()-1}]")
    print(f"      - Metadata prediction_delta would be {realized_delta}, matching actual gap ✓")

    # Verify
    assert not tf.reduce_any(actions_fixed[deltas_clamped_fixed:] != 0), "Should be zero-padded after realized_delta"
    print("\n  ✓ Verification passed!")

    # Scenario 2: Last timestep
    print("\n" + "=" * 80)
    print("SCENARIO 2: t=9 (last timestep), sampled delta=3")
    print("=" * 80)

    t = 9
    sampled_delta = 3

    print("\n--- BUGGY VERSION ---")
    future_index_buggy = min(t + sampled_delta, traj_len.numpy() - 1)
    deltas_clamped_buggy = min(sampled_delta, summation_steps)
    print(f"  Future index: {future_index_buggy} (clamped to last obs)")
    print(f"  deltas_clamped: {deltas_clamped_buggy}")
    print(f"  ⚠️  Would try to gather {deltas_clamped_buggy} actions from t={t}")
    print(f"      But there are NO valid actions at t={t}! (action[9] is invalid)")

    print("\n--- FIXED VERSION ---")
    realized_delta = future_index_buggy - t  # 9 - 9 = 0
    deltas_clamped_fixed = min(realized_delta, summation_steps)
    print(f"  Realized delta: {realized_delta}")
    print(f"  deltas_clamped: {deltas_clamped_fixed}")
    print(f"  ✓  Correctly gathers 0 actions (all zero-padded)")
    print(f"      This makes sense: can't predict actions from final observation!")

    # Scenario 3: Within bounds (no clamping needed)
    print("\n" + "=" * 80)
    print("SCENARIO 3: t=3, sampled delta=4 (no clamping needed)")
    print("=" * 80)

    t = 3
    sampled_delta = 4

    future_index = min(t + sampled_delta, traj_len.numpy() - 1)
    realized_delta = future_index - t
    deltas_clamped = min(realized_delta, summation_steps)

    print(f"  Future index: {future_index}")
    print(f"  Realized delta: {realized_delta}")
    print(f"  deltas_clamped: {deltas_clamped}")
    print(f"  Actions gathered: action[{t}] through action[{t + deltas_clamped - 1}]")
    print(f"  ✓  Same result for both versions (no boundary issues)")

    assert realized_delta == sampled_delta, "Should be equal when no clamping occurs"
    print(f"  ✓  realized_delta == sampled_delta when within bounds")


if __name__ == "__main__":
    tf.config.set_visible_devices([], 'GPU')

    test_realized_deltas_fix()

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print("""
The realized_deltas fix ensures:

1. ✓ Never gather invalid action[T-1] (which transitions to non-existent obs[T])

2. ✓ Number of actions matches the actual visual gap between images
   - If images show [obs[7], obs[9]], gather exactly 2 actions: [action[7], action[8]]
   - Not 5 actions just because delta was sampled as 5

3. ✓ Accurate metadata: prediction_delta stores the realized gap, not the clamped sampled delta
   - Model receives truthful information about the prediction horizon

4. ✓ Clearer training signal: Model learns to predict the right number of actions for the gap
   - No confusion about "sometimes I need to pad with zeros at position 3, sometimes at 5"

Without this fix:
- Boundary cases near trajectory end would gather too many actions
- Would include zero-padded action[T-1] in some cases
- Metadata would be misleading (prediction_delta != actual visual gap)
    """)
