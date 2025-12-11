import numpy as np
import tensorflow as tf

from openpi_cot.datasets.base_dataset import _euler_xyz_extrinsic_to_matrix
from openpi_cot.datasets.base_dataset import sum_actions
from openpi_cot.datasets.utils.data_utils import _R_from_euler_xyz
from openpi_cot.datasets.utils.data_utils import euler_diff


def test_euler_diff_and_sum_actions_reconstruct_pose():
    num_poses = 8
    rng = np.random.default_rng(seed=7)

    # Build a random pose trajectory [x, y, z, roll, pitch, yaw].
    xyz = rng.uniform(-0.2, 0.2, size=(num_poses, 3)).astype(np.float32)
    rpy = rng.uniform(-np.pi, np.pi, size=(num_poses, 3)).astype(np.float32)
    eef_state = tf.constant(np.concatenate([xyz, rpy], axis=-1), dtype=tf.float32)

    # Deltas between consecutive poses use the production euler_diff implementation.
    movement_actions = tf.concat(
        [
            eef_state[1:, :3] - eef_state[:-1, :3],
            euler_diff(eef_state[1:, 3:6], eef_state[:-1, 3:6]),
        ],
        axis=-1,
    )

    for k in range(num_poses - 1):
        R1 = _R_from_euler_xyz(eef_state[k, 3:])
        dr = _R_from_euler_xyz(movement_actions[k, 3:])

        R2_hat = tf.linalg.matmul(R1, dr, transpose_a=False)
        # R2_hat = _euler_xyz_from_R(R2_hat)
        R2 = _R_from_euler_xyz(eef_state[k + 1, 3:])

        np.testing.assert_allclose(
            R2_hat.numpy(),
            R2.numpy(),
            rtol=1e-5,
            atol=1e-5,
        )

    summed_action = sum_actions(movement_actions[None], [7])[0]
    final_translation_hat = eef_state[0, :3] + summed_action[:3]
    final_translation = eef_state[-1, :3]
    np.testing.assert_allclose(
        final_translation_hat.numpy(),
        final_translation.numpy(),
        rtol=1e-5,
        atol=1e-5,
    )

    # Recover absolute orientations by composing the starting matrix with the summed deltas.
    rotation_deltas = summed_action[3:6]
    R_delta = _euler_xyz_extrinsic_to_matrix(rotation_deltas)
    R_initial = _euler_xyz_extrinsic_to_matrix(eef_state[0, 3:6])
    R_absolute_final_hat = tf.linalg.matmul(R_initial, R_delta)
    R_absolute_final = _euler_xyz_extrinsic_to_matrix(eef_state[-1, 3:6])

    np.testing.assert_allclose(
        R_absolute_final_hat.numpy(),
        R_absolute_final.numpy(),
        rtol=1e-5,
        atol=1e-5,
    )
