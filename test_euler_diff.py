"""
Test suite for euler_diff function to verify correctness in transforms.py usage.

This test verifies that:
1. euler_diff correctly computes relative rotations
2. The relative rotation satisfies: R(angles2) * R(angles_rel) = R(angles1)
3. Edge cases like angle wraparound are handled correctly
4. Usage in transforms is mathematically sound
"""

import numpy as np
import tensorflow as tf

from openpi_cot.dataloader.oxe_utils.data_utils import euler_diff
from openpi_cot.dataloader.oxe_utils.data_utils import _R_from_euler_xyz, _euler_xyz_from_R


def test_euler_diff_basic():
    """Test basic euler_diff functionality."""
    print("\n=== Test 1: Basic euler_diff functionality ===")

    # Test case: simple rotation
    angles1 = tf.constant([[0.1, 0.2, 0.3]], dtype=tf.float32)
    angles2 = tf.constant([[0.0, 0.0, 0.0]], dtype=tf.float32)

    diff = euler_diff(angles1, angles2)
    print(f"angles1: {angles1.numpy()}")
    print(f"angles2: {angles2.numpy()}")
    print(f"euler_diff: {diff.numpy()}")

    # Verify: R2 * R(diff) should equal R1
    R1 = _R_from_euler_xyz(angles1)
    R2 = _R_from_euler_xyz(angles2)
    R_diff = _R_from_euler_xyz(diff)

    R_reconstructed = tf.linalg.matmul(R2, R_diff)

    error = tf.reduce_max(tf.abs(R1 - R_reconstructed))
    print(f"Reconstruction error: {error.numpy()}")
    assert error < 1e-5, f"Reconstruction error too large: {error}"
    print("✓ Basic test passed")


def test_euler_diff_roundtrip():
    """Test that euler_diff satisfies the roundtrip property."""
    print("\n=== Test 2: Roundtrip property ===")

    # Random angles
    np.random.seed(42)
    angles1 = tf.constant(np.random.uniform(-np.pi, np.pi, (10, 3)), dtype=tf.float32)
    angles2 = tf.constant(np.random.uniform(-np.pi, np.pi, (10, 3)), dtype=tf.float32)

    # Compute diff
    diff = euler_diff(angles1, angles2)

    # Verify roundtrip: R2 * R(diff) = R1
    R1 = _R_from_euler_xyz(angles1)
    R2 = _R_from_euler_xyz(angles2)
    R_diff = _R_from_euler_xyz(diff)

    R_reconstructed = tf.linalg.matmul(R2, R_diff)

    max_error = tf.reduce_max(tf.abs(R1 - R_reconstructed))
    mean_error = tf.reduce_mean(tf.abs(R1 - R_reconstructed))

    print(f"Max reconstruction error: {max_error.numpy()}")
    print(f"Mean reconstruction error: {mean_error.numpy()}")

    assert max_error < 1e-4, f"Max error too large: {max_error}"
    print("✓ Roundtrip test passed")


def test_euler_diff_identity():
    """Test that diff of identical angles gives zero (identity rotation)."""
    print("\n=== Test 3: Identity case ===")

    angles = tf.constant([[0.5, -0.3, 1.2]], dtype=tf.float32)
    diff = euler_diff(angles, angles)

    print(f"angles: {angles.numpy()}")
    print(f"euler_diff(angles, angles): {diff.numpy()}")

    # The diff should represent identity rotation (all zeros or 2π multiples)
    R_diff = _R_from_euler_xyz(diff)
    R_identity = tf.eye(3, dtype=tf.float32)

    error = tf.reduce_max(tf.abs(R_diff - R_identity))
    print(f"Difference from identity matrix: {error.numpy()}")

    assert error < 1e-5, f"Identity test failed: {error}"
    print("✓ Identity test passed")


def test_euler_diff_sequential():
    """Test sequential application matches direct computation."""
    print("\n=== Test 4: Sequential composition ===")

    angles0 = tf.constant([[0.0, 0.0, 0.0]], dtype=tf.float32)
    angles1 = tf.constant([[0.1, 0.2, 0.1]], dtype=tf.float32)
    angles2 = tf.constant([[0.2, 0.4, 0.3]], dtype=tf.float32)

    # Compute sequential diffs
    diff_01 = euler_diff(angles1, angles0)
    diff_12 = euler_diff(angles2, angles1)

    print(f"angles0: {angles0.numpy()}")
    print(f"angles1: {angles1.numpy()}")
    print(f"angles2: {angles2.numpy()}")
    print(f"diff(angles1, angles0): {diff_01.numpy()}")
    print(f"diff(angles2, angles1): {diff_12.numpy()}")

    # Verify: R0 * R(diff_01) * R(diff_12) = R2
    R0 = _R_from_euler_xyz(angles0)
    R2_expected = _R_from_euler_xyz(angles2)

    R_diff_01 = _R_from_euler_xyz(diff_01)
    R_diff_12 = _R_from_euler_xyz(diff_12)

    R2_computed = tf.linalg.matmul(tf.linalg.matmul(R0, R_diff_01), R_diff_12)

    error = tf.reduce_max(tf.abs(R2_expected - R2_computed))
    print(f"Sequential composition error: {error.numpy()}")

    assert error < 1e-5, f"Sequential composition failed: {error}"
    print("✓ Sequential composition test passed")


def test_euler_diff_batch():
    """Test batched computation."""
    print("\n=== Test 5: Batch processing ===")

    # Create a batch of angle pairs
    batch_size = 100
    np.random.seed(123)
    angles1 = tf.constant(np.random.uniform(-np.pi, np.pi, (batch_size, 3)), dtype=tf.float32)
    angles2 = tf.constant(np.random.uniform(-np.pi, np.pi, (batch_size, 3)), dtype=tf.float32)

    # Compute diffs for entire batch
    diffs = euler_diff(angles1, angles2)

    print(f"Batch size: {batch_size}")
    print(f"Diffs shape: {diffs.shape}")

    # Verify each element in batch
    R1 = _R_from_euler_xyz(angles1)
    R2 = _R_from_euler_xyz(angles2)
    R_diffs = _R_from_euler_xyz(diffs)

    R_reconstructed = tf.linalg.matmul(R2, R_diffs)

    errors = tf.reduce_max(tf.abs(R1 - R_reconstructed), axis=[1, 2])
    max_error = tf.reduce_max(errors)
    mean_error = tf.reduce_mean(errors)

    print(f"Max error across batch: {max_error.numpy()}")
    print(f"Mean error across batch: {mean_error.numpy()}")

    assert max_error < 1e-4, f"Batch test failed with max error: {max_error}"
    print("✓ Batch processing test passed")


def test_euler_diff_usage_in_transforms():
    """
    Test the typical usage pattern in transforms.py:
    Computing movement actions from consecutive EEF states.
    """
    print("\n=== Test 6: Usage pattern in transforms ===")

    # Simulate a trajectory with 5 timesteps
    # State format: [x, y, z, roll, pitch, yaw]
    np.random.seed(456)

    # Create smooth trajectory (small incremental changes)
    positions = np.cumsum(np.random.uniform(-0.01, 0.01, (5, 3)), axis=0)
    orientations = np.cumsum(np.random.uniform(-0.05, 0.05, (5, 3)), axis=0)

    eef_states = np.concatenate([positions, orientations], axis=1).astype(np.float32)
    eef_states_tf = tf.constant(eef_states, dtype=tf.float32)

    print(f"EEF trajectory shape: {eef_states.shape}")

    # Compute movement actions like in transforms.py
    position_deltas = eef_states_tf[1:, :3] - eef_states_tf[:-1, :3]
    rotation_deltas = euler_diff(
        eef_states_tf[1:, 3:6],  # next orientation
        eef_states_tf[:-1, 3:6]  # current orientation
    )

    movement_actions = tf.concat([position_deltas, rotation_deltas], axis=-1)

    print(f"Movement actions shape: {movement_actions.shape}")
    print(f"Movement actions:\n{movement_actions.numpy()}")

    # Verify: reconstructing the trajectory from movement actions
    reconstructed_positions = [tf.constant(positions[0:1], dtype=tf.float32)]
    reconstructed_orientations_R = [_R_from_euler_xyz(tf.constant(orientations[0:1], dtype=tf.float32))]

    for i in range(len(movement_actions)):
        # Position: simple addition
        next_pos = reconstructed_positions[-1] + position_deltas[i:i+1]
        reconstructed_positions.append(next_pos)

        # Orientation: matrix multiplication
        R_current = reconstructed_orientations_R[-1]
        R_delta = _R_from_euler_xyz(rotation_deltas[i:i+1])
        R_next = tf.linalg.matmul(R_current, R_delta)
        reconstructed_orientations_R.append(R_next)

    # Convert back to Euler angles
    reconstructed_orientations = []
    for R in reconstructed_orientations_R:
        euler = _euler_xyz_from_R(R)
        reconstructed_orientations.append(euler)

    reconstructed_orientations_tf = tf.concat(reconstructed_orientations, axis=0)

    # Compare with original
    R_original = _R_from_euler_xyz(tf.constant(orientations, dtype=tf.float32))
    R_reconstructed = _R_from_euler_xyz(reconstructed_orientations_tf)

    orientation_error = tf.reduce_max(tf.abs(R_original - R_reconstructed))

    print(f"Orientation reconstruction error: {orientation_error.numpy()}")

    assert orientation_error < 1e-4, f"Transform usage test failed: {orientation_error}"
    print("✓ Transform usage pattern test passed")


def test_euler_diff_angle_wraparound():
    """Test handling of angle wraparound (e.g., -π to π transition)."""
    print("\n=== Test 7: Angle wraparound ===")

    # Test cases where angles wrap around ±π
    test_cases = [
        # (angles1, angles2, description)
        ([[3.0, 0.0, 0.0]], [[-3.0, 0.0, 0.0]], "Near ±π wraparound on roll"),
        ([[0.0, 3.0, 0.0]], [[0.0, -3.0, 0.0]], "Near ±π wraparound on pitch"),
        ([[0.0, 0.0, 3.0]], [[0.0, 0.0, -3.0]], "Near ±π wraparound on yaw"),
    ]

    for angles1, angles2, desc in test_cases:
        print(f"\n  Testing: {desc}")
        angles1_tf = tf.constant(angles1, dtype=tf.float32)
        angles2_tf = tf.constant(angles2, dtype=tf.float32)

        diff = euler_diff(angles1_tf, angles2_tf)

        # Verify using rotation matrices
        R1 = _R_from_euler_xyz(angles1_tf)
        R2 = _R_from_euler_xyz(angles2_tf)
        R_diff = _R_from_euler_xyz(diff)

        R_reconstructed = tf.linalg.matmul(R2, R_diff)

        error = tf.reduce_max(tf.abs(R1 - R_reconstructed))
        print(f"  angles1: {angles1}")
        print(f"  angles2: {angles2}")
        print(f"  diff: {diff.numpy()}")
        print(f"  error: {error.numpy()}")

        assert error < 1e-4, f"Wraparound test failed for {desc}: {error}"

    print("✓ Angle wraparound test passed")


if __name__ == "__main__":
    print("=" * 70)
    print("Testing euler_diff correctness")
    print("=" * 70)

    test_euler_diff_basic()
    test_euler_diff_roundtrip()
    test_euler_diff_identity()
    test_euler_diff_sequential()
    test_euler_diff_batch()
    test_euler_diff_usage_in_transforms()
    test_euler_diff_angle_wraparound()

    print("\n" + "=" * 70)
    print("All tests passed! ✓")
    print("euler_diff is correctly implemented and used in transforms.py")
    print("=" * 70)
