"""
Debug test to identify the issue with euler_diff.
"""

import numpy as np
import tensorflow as tf

from openpi_cot.dataloader.oxe_utils.data_utils import euler_diff, _R_from_euler_xyz, _euler_xyz_from_R


def test_euler_diff_implementation():
    """Debug the euler_diff implementation."""
    print("\n=== Debugging euler_diff ===")

    # Simple test case
    angles1 = tf.constant([[0.1, 0.2, 0.3]], dtype=tf.float32)
    angles2 = tf.constant([[0.0, 0.0, 0.0]], dtype=tf.float32)

    print(f"angles1: {angles1.numpy()}")
    print(f"angles2: {angles2.numpy()}")

    # Build rotation matrices
    R1 = _R_from_euler_xyz(angles1)
    R2 = _R_from_euler_xyz(angles2)

    print(f"\nR1:\n{R1.numpy()[0]}")
    print(f"\nR2:\n{R2.numpy()[0]}")

    # Compute Rrel = R2^T * R1
    Rrel = tf.linalg.matmul(R2, R1, transpose_a=True)
    print(f"\nRrel = R2^T * R1:\n{Rrel.numpy()[0]}")

    # Extract Euler angles using the CURRENT implementation in euler_diff
    r20 = Rrel[..., 2, 0]
    r21 = Rrel[..., 2, 1]
    r22 = Rrel[..., 2, 2]
    r10 = Rrel[..., 1, 0]
    r00 = Rrel[..., 0, 0]

    theta_current = tf.asin(-r20)
    phi_current = tf.atan2(r21, r22)
    psi_current = tf.atan2(r10, r00)
    euler_current = tf.stack([phi_current, theta_current, psi_current], axis=-1)

    print(f"\nEuler from current implementation: {euler_current.numpy()}")

    # Extract Euler angles using the CORRECT method (with gimbal lock handling)
    euler_correct = _euler_xyz_from_R(Rrel)
    print(f"Euler from _euler_xyz_from_R: {euler_correct.numpy()}")

    # Now test reconstruction with both methods
    print("\n--- Testing reconstruction with CURRENT method ---")
    R_diff_current = _R_from_euler_xyz(euler_current)
    R_reconstructed_current = tf.linalg.matmul(R2, R_diff_current)
    error_current = tf.reduce_max(tf.abs(R1 - R_reconstructed_current))
    print(f"R_diff_current:\n{R_diff_current.numpy()[0]}")
    print(f"Reconstruction error: {error_current.numpy()}")

    print("\n--- Testing reconstruction with CORRECT method ---")
    R_diff_correct = _R_from_euler_xyz(euler_correct)
    R_reconstructed_correct = tf.linalg.matmul(R2, R_diff_correct)
    error_correct = tf.reduce_max(tf.abs(R1 - R_reconstructed_correct))
    print(f"R_diff_correct:\n{R_diff_correct.numpy()[0]}")
    print(f"Reconstruction error: {error_correct.numpy()}")

    # Compare
    print("\n" + "="*70)
    print("DIAGNOSIS:")
    print(f"  Current implementation error: {error_current.numpy():.10f}")
    print(f"  Correct implementation error: {error_correct.numpy():.10f}")
    print("="*70)

    if error_correct < 1e-5:
        print("\n✓ The issue is confirmed: euler_diff should use _euler_xyz_from_R()")
        print("  instead of the simplified extraction that doesn't handle gimbal lock.")
    else:
        print("\n✗ Unexpected result - further investigation needed")


def test_proposed_fix():
    """Test the proposed fix for euler_diff."""
    print("\n\n=== Testing Proposed Fix ===")

    @tf.function
    def euler_diff_fixed(angles1, angles2, order="xyz", degrees=False):
        """
        Fixed version of euler_diff that uses _euler_xyz_from_R.
        """
        if degrees:
            angles1 = tf.math.multiply(angles1, tf.constant(np.pi, dtype=tf.float32) / 180.0)
            angles2 = tf.math.multiply(angles2, tf.constant(np.pi, dtype=tf.float32) / 180.0)

        # Build rotation matrices
        R1 = _R_from_euler_xyz(angles1)
        R2 = _R_from_euler_xyz(angles2)

        # Compute relative rotation: Rrel = R2^T * R1
        Rrel = tf.linalg.matmul(R2, R1, transpose_a=True)

        # Extract Euler angles using the robust method with gimbal lock handling
        euler_rel = _euler_xyz_from_R(Rrel)

        if degrees:
            euler_rel = tf.math.multiply(euler_rel, 180.0 / tf.constant(np.pi, dtype=tf.float32))

        return euler_rel

    # Test with multiple cases
    test_cases = [
        ([[0.1, 0.2, 0.3]], [[0.0, 0.0, 0.0]], "Simple case"),
        ([[0.5, -0.3, 1.2]], [[0.2, 0.1, -0.5]], "Random case 1"),
        ([[1.5, 0.8, -1.0]], [[-0.3, 1.2, 0.4]], "Random case 2"),
        ([[0.0, 0.0, 0.0]], [[0.1, 0.2, 0.3]], "Inverted simple case"),
    ]

    all_passed = True
    for angles1_data, angles2_data, desc in test_cases:
        print(f"\nTest: {desc}")
        angles1 = tf.constant(angles1_data, dtype=tf.float32)
        angles2 = tf.constant(angles2_data, dtype=tf.float32)

        diff = euler_diff_fixed(angles1, angles2)

        # Verify: R2 * R(diff) = R1
        R1 = _R_from_euler_xyz(angles1)
        R2 = _R_from_euler_xyz(angles2)
        R_diff = _R_from_euler_xyz(diff)

        R_reconstructed = tf.linalg.matmul(R2, R_diff)
        error = tf.reduce_max(tf.abs(R1 - R_reconstructed))

        print(f"  angles1: {angles1.numpy()}")
        print(f"  angles2: {angles2.numpy()}")
        print(f"  diff: {diff.numpy()}")
        print(f"  error: {error.numpy():.10f}")

        if error < 1e-5:
            print(f"  ✓ PASS")
        else:
            print(f"  ✗ FAIL")
            all_passed = False

    print("\n" + "="*70)
    if all_passed:
        print("✓ All tests passed with the fixed implementation!")
    else:
        print("✗ Some tests failed")
    print("="*70)


if __name__ == "__main__":
    print("="*70)
    print("Debugging euler_diff Implementation")
    print("="*70)

    test_euler_diff_implementation()
    test_proposed_fix()

    print("\n\nRECOMMENDATION:")
    print("-" * 70)
    print("Replace the Euler extraction in euler_diff() with:")
    print("    euler_rel = _euler_xyz_from_R(Rrel)")
    print("\nInstead of the current manual extraction that doesn't handle")
    print("gimbal lock properly.")
    print("-" * 70)
