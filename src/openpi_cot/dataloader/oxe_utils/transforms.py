"""
transforms.py

Defines a registry of per-dataset standardization transforms for each dataset in Open-X Embodiment.

Transforms adopt the following structure:
    Input: Dictionary of *batched* features (i.e., has leading time dimension)
    Output: Dictionary `step` =>> {
        "observation": {
            <image_keys, depth_image_keys>
            State (in chosen state representation)
        },
        "action": Action (in chosen action representation),
        "language_instruction": str
    }
"""

from typing import Any

# from openpi_cot.dataloader.oxe_utils.rlds.oxe.utils.droid_utils import droid_baseact_transform
# from openpi_cot.dataloader.oxe_utils.rlds.oxe.utils.droid_utils import droid_finetuning_transform
import tensorflow as tf


def binarize_gripper_actions(actions: tf.Tensor, threshold: float = 0.95) -> tf.Tensor:
    """
    Converts gripper actions from continuous to binary values (0 and 1).

    We exploit that fact that most of the time, the gripper is fully open (near 1.0) or fully closed (near 0.0). As it
    transitions between the two, it sometimes passes through a few intermediate values. We relabel those intermediate
    values based on the state that is reached _after_ those intermediate values.

    In the edge case that the trajectory ends with an intermediate value, we give up on binarizing and relabel that
    chunk of intermediate values as the last action in the trajectory.

    The `scan_fn` implements the following logic:
        new_actions = np.empty_like(actions)
        carry = actions[-1]
        for i in reversed(range(actions.shape[0])):
            if in_between_mask[i]:
                carry = carry
            else:
                carry = float(open_mask[i])
            new_actions[i] = carry
    """
    open_mask, closed_mask = actions > threshold, actions < (1 - threshold)
    in_between_mask = tf.logical_not(tf.logical_or(open_mask, closed_mask))
    is_open_float = tf.cast(open_mask, tf.float32)

    def scan_fn(carry, i):
        return tf.cond(in_between_mask[i], lambda: carry, lambda: is_open_float[i])

    return tf.scan(scan_fn, tf.range(tf.shape(actions)[0]), tf.cast(actions[-1], tf.float32), reverse=True)


def invert_gripper_actions(actions: tf.Tensor) -> tf.Tensor:
    return 1 - actions


def rel2abs_gripper_actions(actions: tf.Tensor) -> tf.Tensor:
    """
    Converts relative gripper actions (+1 for closing, -1 for opening) to absolute actions (0 = closed; 1 = open).

    Assumes that the first relative gripper is not redundant (i.e. close when already closed)!
    """
    # Note =>> -1 for closing, 1 for opening, 0 for no change
    opening_mask, closing_mask = actions < -0.1, actions > 0.1
    thresholded_actions = tf.where(opening_mask, 1, tf.where(closing_mask, -1, 0))

    def scan_fn(carry, i):
        return tf.cond(thresholded_actions[i] == 0, lambda: carry, lambda: thresholded_actions[i])

    # If no relative grasp, assumes open for whole trajectory
    start = -1 * thresholded_actions[tf.argmax(thresholded_actions != 0, axis=0)]
    start = tf.cond(start == 0, lambda: 1, lambda: start)

    # Note =>> -1 for closed, 1 for open
    new_actions = tf.scan(scan_fn, tf.range(tf.shape(actions)[0]), start)
    new_actions = tf.cast(new_actions, tf.float32) / 2 + 0.5

    return new_actions


# === Bridge-V2 =>> Dataset-Specific Transform ===
def relabel_bridge_actions(traj: dict[str, Any]) -> dict[str, Any]:
    """Relabels actions to use reached proprioceptive state; discards last timestep (no-action)."""
    movement_actions = traj["observation"]["state"][1:, :6] - traj["observation"]["state"][:-1, :6]
    traj_truncated = tf.nest.map_structure(lambda x: x[:-1], traj)
    traj_truncated["action"] = tf.concat([movement_actions, traj["action"][:-1, -1:]], axis=1)

    return traj_truncated


def bridge_v2_oxe_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    """
    Applies to version of Bridge V2 in Open X-Embodiment mixture.

    Note =>> In original Bridge V2 dataset, the first timestep has an all-zero action, so we remove it!
    """
    # for key in trajectory:
    #     if key == "traj_metadata":
    #         continue
    #     if key in ["observation", "action"]:
    #         for key2 in trajectory[key]:
    #             trajectory[key][key2] = trajectory[key][key2][1:]
    #     else:
    #         trajectory[key] = trajectory[key][1:]

    # trajectory["action"] = tf.concat(
    #     (
    #         trajectory["action"]["world_vector"],
    #         trajectory["action"]["rotation_delta"],
    #         tf.cast(trajectory["action"]["open_gripper"][:, None], tf.float32),
    #     ),
    #     axis=-1,
    # )
    # # print(trajectory.keys(), trajectory['observation'].keys())
    # trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    # trajectory = relabel_bridge_actions(trajectory)
    # trajectory["observation"]["EEF_state"] = trajectory["observation"]["state"][:, :6]
    # trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]

    # print("bridge", trajectory.keys())
    # return trajectory
    """
    Applies to original version of Bridge V2 from the official project website.

    Note =>> In original Bridge V2 dataset, the first timestep has an all-zero action, so we remove it!
    """
    from openpi_cot.dataloader.oxe_utils.data_utils import euler_diff

    for key in trajectory:
        if key == "traj_metadata":
            continue
        if key == "observation":
            for key2 in trajectory[key]:
                trajectory[key][key2] = trajectory[key][key2][1:]
        else:
            trajectory[key] = trajectory[key][1:]

    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            binarize_gripper_actions(trajectory["action"][:, -1])[:, None],
        ],
        axis=1,
    )
    # trajectory = relabel_bridge_actions(trajectory)
    trajectory["observation"]["EEF_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]

    movement_actions = tf.concat(
        (
            trajectory["observation"]["EEF_state"][1:, :3] - trajectory["observation"]["EEF_state"][:-1, :3],
            euler_diff(
                trajectory["observation"]["EEF_state"][1:, 3:6],
                trajectory["observation"]["EEF_state"][:-1, 3:6],
            ),
        ),
        axis=-1,
    )
    traj_truncated = tf.nest.map_structure(lambda x: x[:-1], trajectory)
    traj_truncated["action"] = tf.concat([movement_actions, trajectory["action"][:-1, -1:]], axis=1)

    return traj_truncated


def bridge_orig_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    """
    Applies to original version of Bridge V2 from the official project website.

    Note =>> In original Bridge V2 dataset, the first timestep has an all-zero action, so we remove it!
    """
    for key in trajectory:
        if key == "traj_metadata":
            continue
        if key == "observation":
            for key2 in trajectory[key]:
                trajectory[key][key2] = trajectory[key][key2][1:]
        else:
            trajectory[key] = trajectory[key][1:]

    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            binarize_gripper_actions(trajectory["action"][:, -1])[:, None],
        ],
        axis=1,
    )
    # print(trajectory.keys(), trajectory['observation'].keys())
    trajectory = relabel_bridge_actions(trajectory)
    trajectory["observation"]["EEF_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    return trajectory


def ppgm_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            binarize_gripper_actions(trajectory["action"][:, -1])[:, None],
        ],
        axis=1,
    )
    trajectory["observation"]["EEF_state"] = trajectory["observation"]["cartesian_position"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["gripper_position"][:, -1:]
    return trajectory


def rt1_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    import tensorflow_graphics.geometry.transformation as tft

    from openpi_cot.dataloader.oxe_utils.data_utils import euler_diff

    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action[:, None],
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]

    trajectory["observation"]["eef_state"] = tf.concat(
        (
            trajectory["observation"]["base_pose_tool_reached"][:, :3],
            tft.euler.from_quaternion(trajectory["observation"]["base_pose_tool_reached"][:, 3:7]),
        ),
        axis=-1,
    )

    movement_actions = tf.concat(
        (
            trajectory["observation"]["base_pose_tool_reached"][1:, :3]
            - trajectory["observation"]["base_pose_tool_reached"][:-1, :3],
            euler_diff(
                tft.euler.from_quaternion(trajectory["observation"]["base_pose_tool_reached"][1:, 3:7]),
                tft.euler.from_quaternion(trajectory["observation"]["base_pose_tool_reached"][:-1, 3:7]),
            ),
        ),
        axis=-1,
    )
    # movement_actions = transform_actions_xyz(movement_actions)
    traj_truncated = tf.nest.map_structure(lambda x: x[:-1], trajectory)
    traj_truncated["action"] = tf.concat([movement_actions, trajectory["action"][:-1, -1:]], axis=1)

    return traj_truncated


def kuka_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action[:, None],
        ),
        axis=-1,
    )
    # decode compressed state
    eef_value = tf.io.decode_compressed(
        trajectory["observation"]["clip_function_input/base_pose_tool_reached"],
        compression_type="ZLIB",
    )
    eef_value = tf.io.decode_raw(eef_value, tf.float32)
    trajectory["observation"]["clip_function_input/base_pose_tool_reached"] = tf.reshape(eef_value, (-1, 7))
    gripper_value = tf.io.decode_compressed(trajectory["observation"]["gripper_closed"], compression_type="ZLIB")
    gripper_value = tf.io.decode_raw(gripper_value, tf.float32)
    trajectory["observation"]["gripper_closed"] = tf.reshape(gripper_value, (-1, 1))
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    # )  # delete uninformative language instruction
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def taco_play_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    from openpi_cot.dataloader.oxe_utils.data_utils import euler_diff

    trajectory["observation"]["state_eef"] = trajectory["observation"]["robot_obs"][:, :6]
    trajectory["observation"]["state_gripper"] = trajectory["observation"]["robot_obs"][:, 7:8]
    trajectory["action"] = trajectory["action"]["rel_actions_world"]

    # invert gripper action + clip, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            tf.clip_by_value(trajectory["action"][:, -1:], 0, 1),
        ),
        axis=-1,
    )

    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]

    movement_actions = tf.concat(
        (
            trajectory["observation"]["state_eef"][1:, :3] - trajectory["observation"]["state_eef"][:-1, :3],
            euler_diff(
                trajectory["observation"]["state_eef"][1:, 3:6],
                trajectory["observation"]["state_eef"][:-1, 3:6],
            ),
        ),
        axis=-1,
    )
    traj_truncated = tf.nest.map_structure(lambda x: x[:-1], trajectory)
    traj_truncated["action"] = tf.concat([movement_actions, trajectory["action"][:-1, -1:]], axis=1)

    return traj_truncated


def jaco_play_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    from openpi_cot.dataloader.oxe_utils.data_utils import euler_diff

    trajectory["observation"]["state_eef"] = trajectory["observation"]["end_effector_cartesian_pos"][:, :6]
    trajectory["observation"]["state_gripper"] = trajectory["observation"]["end_effector_cartesian_pos"][:, -1:]

    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            tf.zeros_like(trajectory["action"]["world_vector"]),
            gripper_action[:, None],
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]

    movement_actions = tf.concat(
        (
            trajectory["observation"]["state_eef"][1:, :3] - trajectory["observation"]["state_eef"][:-1, :3],
            euler_diff(
                trajectory["observation"]["state_eef"][1:, 3:6],
                trajectory["observation"]["state_eef"][:-1, 3:6],
            ),
        ),
        axis=-1,
    )
    traj_truncated = tf.nest.map_structure(lambda x: x[:-1], trajectory)
    traj_truncated["action"] = tf.concat([movement_actions, trajectory["action"][:-1, -1:]], axis=1)

    return traj_truncated


def berkeley_cable_routing_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            tf.zeros_like(trajectory["action"]["world_vector"][:, :1]),
        ),
        axis=-1,
    )
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    # )  # delete uninformative language instruction
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def roboturk_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    # invert absolute gripper action, +1 = open, 0 = close
    gripper_action = invert_gripper_actions(tf.clip_by_value(trajectory["action"]["gripper_closedness_action"], 0, 1))

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action,
        ),
        axis=-1,
    )
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    # )  # delete uninformative language instruction
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def nyu_door_opening_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, 0]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action[:, None],
        ),
        axis=-1,
    )
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    # )  # delete uninformative language instruction
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def viola_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    # make gripper action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"][:, None]
    gripper_action = tf.clip_by_value(gripper_action, 0, 1)
    gripper_action = invert_gripper_actions(gripper_action)

    # trajectory["action"] = tf.concat(
    #     (
    #         trajectory["action"]["world_vector"],
    #         trajectory["action"]["rotation_delta"],
    #         gripper_action,
    #     ),
    #     axis=-1,
    # )
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    # )  # delete uninformative language instruction
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]

    from openpi_cot.dataloader.oxe_utils.data_utils import euler_diff
    from openpi_cot.dataloader.oxe_utils.data_utils import matrix_to_xyzrpy

    # Reshape from column-major flattened format and transpose to row-major
    state_matrix = tf.reshape(trajectory["observation"]["ee_states"][:, -16:], [-1, 4, 4])
    # state_matrix = tf.transpose(state_matrix, [0, 2, 1])
    state_matrix = tf.transpose(state_matrix, [0, 2, 1])  # Transpose to convert column-major to row-major
    trajectory["observation"]["state"] = tf.concat(
        (matrix_to_xyzrpy(state_matrix), trajectory["observation"]["gripper_states"]),
        axis=-1,
    )

    movement_actions = tf.concat(
        (
            trajectory["observation"]["state"][1:, :3] - trajectory["observation"]["state"][:-1, :3],
            euler_diff(
                trajectory["observation"]["state"][1:, 3:6],
                trajectory["observation"]["state"][:-1, 3:6],
            ),
        ),
        axis=-1,
    )
    traj_truncated = tf.nest.map_structure(lambda x: x[:-1], trajectory)
    traj_truncated["action"] = tf.concat([movement_actions, gripper_action[:-1]], axis=1)
    return traj_truncated

    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["language_instruction"]), ""
    # )  # delete uninformative language instruction
    return trajectory

    return trajectory


def berkeley_autolab_ur5_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    import tensorflow_graphics.geometry.transformation as tft

    from openpi_cot.dataloader.oxe_utils.data_utils import euler_diff

    trajectory["observation"]["state"] = trajectory["observation"]["robot_state"][:, 6:14]
    trajectory["observation"]["depth"] = trajectory["observation"].pop("image_with_depth")

    # make gripper action absolute action, +1 = open, 0 = close
    gripper_action = trajectory["action"]["gripper_closedness_action"]
    gripper_action = rel2abs_gripper_actions(gripper_action)

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            gripper_action[:, None],
        ),
        axis=-1,
    )

    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]

    movement_actions = tf.concat(
        (
            trajectory["observation"]["state"][1:, :3] - trajectory["observation"]["state"][:-1, :3],
            euler_diff(
                tft.euler.from_quaternion(trajectory["observation"]["state"][1:, 3:7]),
                tft.euler.from_quaternion(trajectory["observation"]["state"][:-1, 3:7]),
            ),
        ),
        axis=-1,
    )
    traj_truncated = tf.nest.map_structure(lambda x: x[:-1], trajectory)
    traj_truncated["action"] = tf.concat([movement_actions, trajectory["action"][:-1, -1:]], axis=1)

    return traj_truncated


def toto_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            tf.cast(trajectory["action"]["open_gripper"][:, None], tf.float32),
        ),
        axis=-1,
    )
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["observation"]["natural_language_instruction"]), ""
    # )  # delete uninformative language instruction
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def language_table_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    # default to "open" gripper
    trajectory["action"] = tf.concat(
        (
            trajectory["action"],
            tf.zeros_like(trajectory["action"]),
            tf.zeros_like(trajectory["action"]),
            tf.ones_like(trajectory["action"][:, :1]),
        ),
        axis=-1,
    )

    # decode language instruction
    instruction_bytes = trajectory["observation"]["instruction"]
    instruction_encoded = tf.strings.unicode_encode(instruction_bytes, output_encoding="UTF-8")
    # Remove trailing padding --> convert RaggedTensor to regular Tensor.
    trajectory["language_instruction"] = tf.strings.split(instruction_encoded, "\x00")[:, :1].to_tensor()[:, 0]
    return trajectory


def pusht_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["world_vector"],
            trajectory["action"]["rotation_delta"],
            trajectory["action"]["gripper_closedness_action"][:, None],
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]
    return trajectory


def stanford_kuka_multimodal_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["observation"]["depth_image"] = trajectory["observation"]["depth_image"][..., 0]
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tf.zeros_like(trajectory["action"][:, :3]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def nyu_rot_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][..., :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][..., -1:]
    trajectory["action"] = trajectory["action"][..., :7]
    return trajectory


def stanford_hydra_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    # invert gripper action, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(trajectory["action"][:, -1:]),
        ),
        axis=-1,
    )

    trajectory["observation"]["eef_state"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :3],
            trajectory["observation"]["state"][:, 7:10],
        ),
        axis=-1,
    )
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -3:-2]
    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["language_instruction"]), ""
    # )  # delete uninformative language instruction
    return trajectory


def austin_buds_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    # # invert gripper action + clip, +1 = open, 0 = close
    # trajectory["action"] = tf.concat(
    #     (
    #         trajectory["action"][:, :6],
    #         invert_gripper_actions(tf.clip_by_value(trajectory["action"][:, -1:], 0, 1)),
    #     ),
    #     axis=-1,
    # )

    # trajectory["observation"]["state"] = trajectory["observation"]["state"][:, :8]
    # # trajectory["language_instruction"] = tf.fill(
    # #     tf.shape(trajectory["language_instruction"]), ""
    # # )  # delete uninformative language instruction
    # return trajectory

    from openpi_cot.dataloader.oxe_utils.data_utils import euler_diff
    from openpi_cot.dataloader.oxe_utils.data_utils import matrix_to_xyzrpy

    # Reshape from column-major flattened format and transpose to row-major
    state_matrix = tf.reshape(trajectory["observation"]["state"][:, -16:], [-1, 4, 4])
    state_matrix = tf.transpose(state_matrix, [0, 2, 1])  # Transpose to convert column-major to row-major
    trajectory["observation"]["state"] = tf.concat(
        (matrix_to_xyzrpy(state_matrix), trajectory["observation"]["state"][:, 7:8]),
        axis=-1,
    )

    movement_actions = tf.concat(
        (
            trajectory["observation"]["state"][1:, :3] - trajectory["observation"]["state"][:-1, :3],
            euler_diff(
                trajectory["observation"]["state"][1:, 3:6],
                trajectory["observation"]["state"][:-1, 3:6],
            ),
        ),
        axis=-1,
    )
    traj_truncated = tf.nest.map_structure(lambda x: x[:-1], trajectory)
    traj_truncated["action"] = tf.concat(
        [movement_actions, invert_gripper_actions(tf.clip_by_value(trajectory["action"][1:, -1:], 0, 1))], axis=1
    )

    # Randomly pad empty language instructions with fallback text
    fallback_instructions = tf.constant(
        [
            "Do something useful.",
            "Complete the task.",
            "Perform the task.",
            "Carry out the objective.",
            "Execute the current task.",
            "Accomplish the goal.",
            "Proceed with the task.",
            "Handle the task at hand.",
            "Continue the operation.",
            "Fulfill the task.",
            "Take meaningful steps.",
            "Demonstrate useful behavior.",
            "Act in a useful manner.",
            "Engage in productive actions.",
            "Make useful moves.",
            "Undertake useful actions.",
            "Behave purposefully.",
            "Start the activity.",
        ],
        dtype=tf.string,
    )

    # Check if language instruction is empty and replace with random fallback
    traj_len = tf.shape(traj_truncated["language_instruction"])[0]
    instruction = traj_truncated["language_instruction"][0]
    is_empty = tf.logical_or(
        tf.equal(tf.strings.length(tf.strings.strip(instruction)), 0),
        tf.equal(instruction, tf.constant("", dtype=tf.string)),
    )

    random_fallback = tf.random.shuffle(fallback_instructions)[0]
    selected_instruction = tf.cond(is_empty, lambda: random_fallback, lambda: instruction)
    traj_truncated["language_instruction"] = tf.fill([traj_len], selected_instruction)

    return traj_truncated


def nyu_franka_play_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["observation"]["depth"] = tf.cast(trajectory["observation"]["depth"][..., 0], tf.float32)
    trajectory["observation"]["depth_additional_view"] = tf.cast(
        trajectory["observation"]["depth_additional_view"][..., 0], tf.float32
    )
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][:, -6:]

    # clip gripper action, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, -8:-2],
            tf.clip_by_value(trajectory["action"][:, -2:-1], 0, 1),
        ),
        axis=-1,
    )

    print("nyu", trajectory.keys())
    print("nyu obs", trajectory["observation"].keys())

    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["language_instruction"]), ""
    # )  # delete uninformative language instruction
    return trajectory


def maniskill_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][..., 7:8]
    return trajectory


def furniture_bench_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    import tensorflow_graphics.geometry.transformation as tft

    from openpi_cot.dataloader.oxe_utils.data_utils import euler_diff

    trajectory["observation"]["state"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :7],
            trajectory["observation"]["state"][:, -1:],
        ),
        axis=-1,
    )

    # invert gripper action + clip, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tft.euler.from_quaternion(trajectory["action"][:, 3:7]),
            invert_gripper_actions(tf.clip_by_value(trajectory["action"][:, -1:], 0, 1)),
        ),
        axis=-1,
    )

    movement_actions = tf.concat(
        (
            trajectory["observation"]["state"][1:, :3] - trajectory["observation"]["state"][:-1, :3],
            euler_diff(
                tft.euler.from_quaternion(trajectory["observation"]["state"][1:, 3:7]),
                tft.euler.from_quaternion(trajectory["observation"]["state"][:-1, 3:7]),
            ),
        ),
        axis=-1,
    )
    traj_truncated = tf.nest.map_structure(lambda x: x[:-1], trajectory)
    traj_truncated["action"] = tf.concat([movement_actions, trajectory["action"][:-1, -1:]], axis=1)

    return traj_truncated


def cmu_franka_exploration_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., :-1]
    return trajectory


def ucsd_kitchen_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    # from openpi_cot.dataloader.oxe_utils.data_utils import euler_diff

    # trajectory["observation"]["joint_state"] = trajectory["observation"]["state"][:, :7]

    # movement_actions = tf.concat(
    #     (
    #         trajectory["action"][1:, :3] - trajectory["action"][:-1, :3],
    #         euler_diff(
    #             trajectory["action"][1:, 3:6],
    #             trajectory["action"][:-1, 3:6],
    #         ),
    #         trajectory["action"][1:, 6:7],
    #     ),
    #     axis=-1,
    # )
    # traj_truncated = tf.nest.map_structure(lambda x: x[:-1], trajectory)
    # traj_truncated["action"] = movement_actions
    # return traj_truncated
    trajectory["action"] = trajectory["action"][..., :-1]
    return trajectory


def ucsd_pick_place_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tf.zeros_like(trajectory["action"][:, :3]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def austin_sailor_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    # # invert gripper action + clip, +1 = open, 0 = close
    # trajectory["action"] = tf.concat(
    #     (
    #         trajectory["action"][:, :6],
    #         invert_gripper_actions(tf.clip_by_value(trajectory["action"][:, -1:], 0, 1)),
    #     ),
    #     axis=-1,
    # )

    # # trajectory["language_instruction"] = tf.fill(
    # #     tf.shape(trajectory["language_instruction"]), ""
    # # )  # delete uninformative language instruction
    # return trajectory

    from openpi_cot.dataloader.oxe_utils.data_utils import euler_diff
    from openpi_cot.dataloader.oxe_utils.data_utils import matrix_to_xyzrpy

    trajectory["observation"]["state"] = tf.concat(
        (matrix_to_xyzrpy(trajectory["observation"]["state_ee"]), trajectory["observation"]["state"][:, -1:]),
        axis=-1,
    )

    movement_actions = tf.concat(
        (
            trajectory["observation"]["state"][1:, :3] - trajectory["observation"]["state"][:-1, :3],
            euler_diff(
                trajectory["observation"]["state"][1:, 3:6],
                trajectory["observation"]["state"][:-1, 3:6],
            ),
        ),
        axis=-1,
    )
    traj_truncated = tf.nest.map_structure(lambda x: x[:-1], trajectory)
    traj_truncated["action"] = tf.concat(
        [movement_actions, invert_gripper_actions(tf.clip_by_value(trajectory["action"][1:, -1:], 0, 1))], axis=1
    )

    # Randomly pad empty language instructions with fallback text
    fallback_instructions = tf.constant(
        [
            "Do something useful.",
            "Complete the task.",
            "Perform the task.",
            "Carry out the objective.",
            "Execute the current task.",
            "Accomplish the goal.",
            "Proceed with the task.",
            "Handle the task at hand.",
            "Continue the operation.",
            "Fulfill the task.",
            "Take meaningful steps.",
            "Demonstrate useful behavior.",
            "Act in a useful manner.",
            "Engage in productive actions.",
            "Make useful moves.",
            "Undertake useful actions.",
            "Behave purposefully.",
            "Start the activity.",
        ],
        dtype=tf.string,
    )

    # Check if language instruction is empty and replace with random fallback
    traj_len = tf.shape(traj_truncated["language_instruction"])[0]
    instruction = traj_truncated["language_instruction"][0]
    is_empty = tf.logical_or(
        tf.equal(tf.strings.length(tf.strings.strip(instruction)), 0),
        tf.equal(instruction, tf.constant("", dtype=tf.string)),
    )

    random_fallback = tf.random.shuffle(fallback_instructions)[0]
    selected_instruction = tf.cond(is_empty, lambda: random_fallback, lambda: instruction)
    traj_truncated["language_instruction"] = tf.fill([traj_len], selected_instruction)

    return traj_truncated


def austin_sirius_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    from openpi_cot.dataloader.oxe_utils.data_utils import euler_diff
    from openpi_cot.dataloader.oxe_utils.data_utils import matrix_to_xyzrpy

    # invert gripper action + clip, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(tf.clip_by_value(trajectory["action"][:, -1:], 0, 1)),
        ),
        axis=-1,
    )

    trajectory["observation"]["state"] = tf.concat(
        (matrix_to_xyzrpy(trajectory["observation"]["state_ee"]), trajectory["observation"]["state"][:, -1:]),
        axis=-1,
    )

    movement_actions = tf.concat(
        (
            trajectory["observation"]["state"][1:, :3] - trajectory["observation"]["state"][:-1, :3],
            euler_diff(
                trajectory["observation"]["state"][1:, 3:6],
                trajectory["observation"]["state"][:-1, 3:6],
            ),
        ),
        axis=-1,
    )
    traj_truncated = tf.nest.map_structure(lambda x: x[:-1], trajectory)
    traj_truncated["action"] = tf.concat([movement_actions, trajectory["action"][:-1, -1:]], axis=1)

    # Randomly pad empty language instructions with fallback text
    fallback_instructions = tf.constant(
        [
            "Do something useful.",
            "Complete the task.",
            "Perform the task.",
            "Carry out the objective.",
            "Execute the current task.",
            "Accomplish the goal.",
            "Proceed with the task.",
            "Handle the task at hand.",
            "Continue the operation.",
            "Fulfill the task.",
            "Take meaningful steps.",
            "Demonstrate useful behavior.",
            "Act in a useful manner.",
            "Engage in productive actions.",
            "Make useful moves.",
            "Undertake useful actions.",
            "Behave purposefully.",
            "Start the activity.",
        ],
        dtype=tf.string,
    )

    # Check if language instruction is empty and replace with random fallback
    traj_len = tf.shape(traj_truncated["language_instruction"])[0]
    instruction = traj_truncated["language_instruction"][0]
    is_empty = tf.logical_or(
        tf.equal(tf.strings.length(tf.strings.strip(instruction)), 0),
        tf.equal(instruction, tf.constant("", dtype=tf.string)),
    )

    random_fallback = tf.random.shuffle(fallback_instructions)[0]
    selected_instruction = tf.cond(is_empty, lambda: random_fallback, lambda: instruction)
    traj_truncated["language_instruction"] = tf.fill([traj_len], selected_instruction)

    return traj_truncated

    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["language_instruction"]), ""
    # )  # delete uninformative language instruction


def bc_z_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    from openpi_cot.dataloader.oxe_utils.data_utils import coordinate_transform_bcz
    from openpi_cot.dataloader.oxe_utils.data_utils import euler_diff

    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["future/xyz_residual"][:, :3],
            trajectory["action"]["future/axis_angle_residual"][:, :3],
            invert_gripper_actions(tf.cast(trajectory["action"]["future/target_close"][:, :1], tf.float32)),
        ),
        axis=-1,
    )
    trajectory["language_instruction"] = trajectory["observation"]["natural_language_instruction"]

    trajectory["observation"]["eef_state"] = coordinate_transform_bcz(
        tf.concat(
            (
                trajectory["observation"]["present/xyz"][:, :3],
                trajectory["observation"]["present/axis_angle"][:, :3],
            ),
            axis=-1,
        )
    )

    # movement_actions = tf.concat(
    #     (
    #         trajectory["observation"]["present/xyz"][1:, :3] - trajectory["observation"]["present/xyz"][:-1, :3],
    #         euler_diff(
    #             axis_angle_to_euler(trajectory["observation"]["present/axis_angle"][1:, :3]),
    #             axis_angle_to_euler(trajectory["observation"]["present/axis_angle"][:-1, :3]),
    #         ),
    #     ),
    #     axis=-1,
    # )
    movement_actions = tf.concat(
        (
            trajectory["observation"]["present/xyz"][1:, :3] - trajectory["observation"]["present/xyz"][:-1, :3],
            euler_diff(
                trajectory["observation"]["present/axis_angle"][1:, :3],
                trajectory["observation"]["present/axis_angle"][:-1, :3],
            ),
        ),
        axis=-1,
    )
    movement_actions = coordinate_transform_bcz(movement_actions)
    traj_truncated = tf.nest.map_structure(lambda x: x[:-1], trajectory)
    traj_truncated["action"] = tf.concat([movement_actions, trajectory["action"][:-1, -1:]], axis=1)

    return traj_truncated


def tokyo_pr2_opening_fridge_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    trajectory["action"] = trajectory["action"][..., :-1]
    return trajectory


def tokyo_pr2_tabletop_manipulation_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    trajectory["action"] = trajectory["action"][..., :-1]
    return trajectory


def utokyo_xarm_pick_place_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    return trajectory


def utokyo_xarm_bimanual_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., -7:]
    return trajectory


def robo_net_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["observation"]["eef_state"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :4],
            tf.zeros_like(trajectory["observation"]["state"][:, :2]),
        ),
        axis=-1,
    )
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :4],
            tf.zeros_like(trajectory["action"][:, :2]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def berkeley_mvp_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    return trajectory


def berkeley_rpt_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    return trajectory


def kaist_nonprehensible_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["observation"]["state"] = trajectory["observation"]["state"][:, -7:]
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            tf.zeros_like(trajectory["action"][:, :1]),
        ),
        axis=-1,
    )
    return trajectory


def stanford_mask_vit_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["observation"]["eef_state"] = tf.concat(
        (
            trajectory["observation"]["end_effector_pose"][:, :4],
            tf.zeros_like(trajectory["observation"]["end_effector_pose"][:, :2]),
        ),
        axis=-1,
    )
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["end_effector_pose"][:, -1:]
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :4],
            tf.zeros_like(trajectory["action"][:, :2]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def tokyo_lsmo_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    return trajectory


def dlr_sara_pour_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    return trajectory


def dlr_sara_grid_clamp_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["observation"]["state"] = trajectory["observation"]["state"][:, :6]
    return trajectory


def dlr_edan_shared_control_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    from openpi_cot.dataloader.oxe_utils.data_utils import euler_diff
    from openpi_cot.dataloader.oxe_utils.data_utils import zxy_to_xyz_tf

    # invert gripper action, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            zxy_to_xyz_tf(trajectory["action"][:, 3:6]),
            invert_gripper_actions(trajectory["action"][:, -1:]),
        ),
        axis=-1,
    )

    trajectory["observation"]["state"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :3],
            zxy_to_xyz_tf(trajectory["observation"]["state"][:, 3:6]),
            trajectory["observation"]["state"][:, -1:],
        ),
        axis=-1,
    )

    movement_actions = tf.concat(
        (
            trajectory["observation"]["state"][1:, :3] - trajectory["observation"]["state"][:-1, :3],
            euler_diff(
                zxy_to_xyz_tf(trajectory["observation"]["state"][1:, 3:6]),
                zxy_to_xyz_tf(trajectory["observation"]["state"][:-1, 3:6]),
            ),
        ),
        axis=-1,
    )

    traj_truncated = tf.nest.map_structure(lambda x: x[:-1], trajectory)
    traj_truncated["action"] = tf.concat([movement_actions, trajectory["action"][:-1, -1:]], axis=1)

    return traj_truncated


def asu_table_top_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["ground_truth_states"]["EE"]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    return trajectory


def robocook_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["observation"]["eef_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    return trajectory


def imperial_wristcam_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., :-1]
    return trajectory


def iamlab_pick_insert_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    import tensorflow_graphics.geometry.transformation as tft

    trajectory["observation"]["joint_state"] = trajectory["observation"]["state"][:, :7]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, 7:8]
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tft.euler.from_quaternion(trajectory["action"][:, 3:7]),
            trajectory["action"][:, 7:8],
        ),
        axis=-1,
    )
    return trajectory


def uiuc_d3field_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"],
            tf.zeros_like(trajectory["action"]),
            tf.zeros_like(trajectory["action"][:, :1]),
        ),
        axis=-1,
    )
    return trajectory


def utaustin_mutex_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    # trajectory["observation"]["state"] = trajectory["observation"]["state"][:, :8]

    # invert gripper action + clip, +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :6],
            invert_gripper_actions(tf.clip_by_value(trajectory["action"][:, -1:], 0, 1)),
        ),
        axis=-1,
    )

    from openpi_cot.dataloader.oxe_utils.data_utils import euler_diff
    from openpi_cot.dataloader.oxe_utils.data_utils import matrix_to_xyzrpy

    # Reshape from column-major flattened format and transpose to row-major
    state_matrix = tf.reshape(trajectory["observation"]["state"][:, -16:], [-1, 4, 4])
    state_matrix = tf.transpose(state_matrix, [0, 2, 1])  # Transpose to convert column-major to row-major
    trajectory["observation"]["state"] = tf.concat(
        (matrix_to_xyzrpy(state_matrix), trajectory["observation"]["state"][:, 7:8]),
        axis=-1,
    )

    movement_actions = tf.concat(
        (
            trajectory["observation"]["state"][1:, :3] - trajectory["observation"]["state"][:-1, :3],
            euler_diff(
                trajectory["observation"]["state"][1:, 3:6],
                trajectory["observation"]["state"][:-1, 3:6],
            ),
        ),
        axis=-1,
    )
    traj_truncated = tf.nest.map_structure(lambda x: x[:-1], trajectory)
    traj_truncated["action"] = tf.concat([movement_actions, trajectory["action"][:-1, -1:]], axis=1)
    return traj_truncated

    # trajectory["language_instruction"] = tf.fill(
    #     tf.shape(trajectory["language_instruction"]), ""
    # )  # delete uninformative language instruction
    # return trajectory


def berkeley_fanuc_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    import tensorflow_graphics.geometry.transformation as tft

    # trajectory["observation"]["joint_state"] = trajectory["observation"]["state"][:, :6]
    # trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, 6:7]

    # dataset does not store gripper actions, so use gripper state info, invert so +1 = open, 0 = close
    trajectory["action"] = tf.concat(
        (
            trajectory["action"],
            invert_gripper_actions(trajectory["observation"]["state"][:, 6:7]),
        ),
        axis=-1,
    )
    # return trajectory

    trajectory["observation"]["state"] = tf.concat(
        (
            trajectory["observation"]["end_effector_state"][:, :3],
            tft.euler.from_quaternion(trajectory["observation"]["end_effector_state"][:, 3:7]),
            invert_gripper_actions(trajectory["observation"]["state"][:, 6:7]),
        ),
        axis=-1,
    )

    return trajectory

    # movement_actions = tf.concat(
    #     (
    #         trajectory["observation"]["state"][1:, :3] - trajectory["observation"]["state"][:-1, :3],
    #         euler_diff(
    #             trajectory["observation"]["state"][1:, 3:6],
    #             trajectory["observation"]["state"][:-1, 3:6],
    #         ),
    #     ),
    #     axis=-1,
    # )
    # traj_truncated = tf.nest.map_structure(lambda x: x[:-1], trajectory)
    # traj_truncated["action"] = tf.concat([movement_actions, trajectory["observation"]["state"][:-1, -1:]], axis=1)
    # return traj_truncated


def cmu_playing_with_food_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    import tensorflow_graphics.geometry.transformation as tft

    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            tft.euler.from_quaternion(trajectory["action"][:, 3:7]),
            trajectory["action"][:, -1:],
        ),
        axis=-1,
    )
    return trajectory


def playfusion_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :3],
            trajectory["action"][:, -4:],
        ),
        axis=-1,
    )
    return trajectory


def cmu_stretch_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    from openpi_cot.dataloader.oxe_utils.data_utils import euler_diff

    trajectory["observation"]["eef_state"] = tf.concat(
        (
            trajectory["observation"]["state"][:, :3],
            tf.zeros_like(trajectory["observation"]["state"][:, :3]),
        ),
        axis=-1,
    )
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -1:]
    trajectory["action"] = trajectory["action"][..., :-1]

    movement_actions = tf.concat(
        (
            trajectory["observation"]["eef_state"][1:, :3] - trajectory["observation"]["eef_state"][:-1, :3],
            euler_diff(
                trajectory["observation"]["eef_state"][1:, 3:6], trajectory["observation"]["eef_state"][:-1, 3:6]
            ),
        ),
        axis=-1,
    )

    traj_truncated = tf.nest.map_structure(lambda x: x[:-1], trajectory)
    traj_truncated["action"] = tf.concat([movement_actions, trajectory["action"][:-1, -1:]], axis=1)

    return traj_truncated


def gnm_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["observation"]["state"] = tf.concat(
        (
            trajectory["observation"]["position"],
            tf.zeros_like(trajectory["observation"]["state"][:, :3]),
            trajectory["observation"]["yaw"],
        ),
        axis=-1,
    )
    trajectory["action"] = tf.concat(
        (
            trajectory["action"],
            tf.zeros_like(trajectory["action"]),
            tf.zeros_like(trajectory["action"]),
            tf.zeros_like(trajectory["action"][:, :1]),
        ),
        axis=-1,
    )
    return trajectory


def fmb_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    import tensorflow_graphics.geometry.transformation as tft

    from openpi_cot.dataloader.oxe_utils.data_utils import euler_diff

    # every input feature is batched, ie has leading batch dimension
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["eef_pose"][:, :3],
            tft.euler.from_quaternion(trajectory["observation"]["eef_pose"][:, 3:7]),
            trajectory["observation"]["state_gripper_pose"][..., None],
        ),
        axis=-1,
    )

    movement_actions = tf.concat(
        (
            trajectory["observation"]["proprio"][1:, :3] - trajectory["observation"]["proprio"][:-1, :3],
            euler_diff(trajectory["observation"]["proprio"][1:, 3:6], trajectory["observation"]["proprio"][:-1, 3:6]),
        ),
        axis=-1,
    )

    traj_truncated = tf.nest.map_structure(lambda x: x[:-1], trajectory)
    traj_truncated["action"] = tf.concat([movement_actions, trajectory["action"][:-1, -1:]], axis=1)

    return traj_truncated


def dobbe_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    # every input feature is batched, ie has leading batch dimension

    from openpi_cot.dataloader.oxe_utils.data_utils import coordinate_transform_dobbe
    from openpi_cot.dataloader.oxe_utils.data_utils import euler_diff

    # every input feature is batched, ie has leading batch dimension
    trajectory["observation"]["proprio"] = tf.concat(
        (
            coordinate_transform_dobbe(trajectory["observation"]["state"][:, :6]),
            trajectory["observation"]["state"][:, -1:],
        ),
        axis=-1,
    )

    movement_actions = tf.concat(
        (
            trajectory["observation"]["proprio"][1:, :3] - trajectory["observation"]["proprio"][:-1, :3],
            euler_diff(trajectory["observation"]["proprio"][1:, 3:6], trajectory["observation"]["proprio"][:-1, 3:6]),
        ),
        axis=-1,
    )
    traj_truncated = tf.nest.map_structure(lambda x: x[:-1], trajectory)
    traj_truncated["action"] = tf.concat([movement_actions, trajectory["action"][:-1, -1:]], axis=1)

    return traj_truncated


def roboset_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    # every input feature is batched, ie has leading batch dimension
    trajectory["observation"]["proprio"] = trajectory["observation"]["state"]

    # gripper action is in -1...1 --> clip to 0...1, flip
    gripper_action = trajectory["action"][:, -1:]
    gripper_action = invert_gripper_actions(tf.clip_by_value(gripper_action, 0, 1))

    trajectory["action"] = tf.concat(
        (
            trajectory["action"][:, :7],
            gripper_action,
        ),
        axis=-1,
    )
    return trajectory


def rh20t_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["action"] = tf.concat(
        (
            trajectory["action"]["tcp_base"],
            tf.cast(trajectory["action"]["gripper"][:, None], tf.float32),
        ),
        axis=-1,
    )
    trajectory["observation"]["proprio"] = tf.concat(
        (
            trajectory["observation"]["tcp_base"],
            trajectory["observation"]["gripper_width"][..., None],
        ),
        axis=-1,
    )
    return trajectory


def tdroid_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            binarize_gripper_actions(trajectory["action"][:, -1])[:, None],
        ],
        axis=1,
    )
    trajectory["observation"]["EEF_state"] = trajectory["observation"]["cartesian_position"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["gripper_position"][:, -1:]
    return trajectory


def libero_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    # gripper action is in -1 (open)...1 (close) --> clip to 0...1, flip --> +1 = open, 0 = close
    gripper_action = trajectory["action"][:, -1:]
    gripper_action = invert_gripper_actions(tf.clip_by_value(gripper_action, 0, 1))

    trajectory["action"] = tf.concat(
        [
            trajectory["action"][:, :6],
            gripper_action,
        ],
        axis=1,
    )
    trajectory["observation"]["EEF_state"] = trajectory["observation"]["state"][:, :6]
    trajectory["observation"]["gripper_state"] = trajectory["observation"]["state"][:, -2:]  # 2D gripper state
    return trajectory


def human_dataset_transform(sample: dict[str, Any]) -> dict[str, Any]:
    """
    Transforms human data into the expected format by adding dummy actions.

    Args:
        sample (Dict[str, Any]): A dictionary containing human data observations.

    Returns:
        Dict[str, Any]: Transformed sample with dummy actions added.
    """
    # Extract the observation from the sample
    observation = sample["observation"]
    print("ego4d", sample.keys())
    print("ego4d obs", sample["observation"].keys())
    # print('sample["observation"]', sample["observation"]['image'].shape[0])
    # observation["state"] = tf.zeros((2, 7), dtype=tf.float32)

    # Create a dummy action tensor with all zeros
    # Assuming the action space is 7D (6D for EEF + 1D for gripper)
    # dummy_action = tf.zeros((2, 7), dtype=tf.float32)

    # Add the dummy action to the sample
    # sample["action"] = dummy_action

    # Split the observation state into EEF_state and gripper_state
    # sample["observation"]["EEF_state"] = observation["state"][:, :6]
    # sample["observation"]["gripper_state"] = observation["state"][:, -1:]

    return sample


def sample_r1_lite_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    trajectory["action"] = trajectory["action"][..., :14]  # exclude torso action
    # Apply R1 Lite specific transformations
    return trajectory


def agibot_dataset_transform(trajectory: dict[str, Any]) -> dict[str, Any]:
    from openpi_cot.dataloader.oxe_utils.data_utils import euler_diff

    movement_actions = tf.concat(
        (
            trajectory["observation"]["state"][1:, :3] - trajectory["observation"]["state"][:-1, :3],
            euler_diff(
                trajectory["observation"]["state"][1:, 3:6],
                trajectory["observation"]["state"][:-1, 3:6],
            ),
            invert_gripper_actions(trajectory["action"][:-1, 6:7]),
            trajectory["observation"]["state"][1:, 7:10] - trajectory["observation"]["state"][:-1, 7:10],
            euler_diff(
                trajectory["observation"]["state"][1:, 10:13],
                trajectory["observation"]["state"][:-1, 10:13],
            ),
            invert_gripper_actions(trajectory["action"][:-1, 13:14]),
        ),
        axis=-1,
    )
    traj_truncated = tf.nest.map_structure(lambda x: x[:-1], trajectory)
    traj_truncated["action"] = movement_actions

    return traj_truncated


# === Registry ===
OXE_STANDARDIZATION_TRANSFORMS = {
    "bridge_v2_oxe": bridge_v2_oxe_dataset_transform,
    "bridge_orig": bridge_orig_dataset_transform,
    "bridge_dataset": bridge_orig_dataset_transform,
    "ppgm": ppgm_dataset_transform,
    "ppgm_static": ppgm_dataset_transform,
    "ppgm_wrist": ppgm_dataset_transform,
    "fractal20220817_data": rt1_dataset_transform,
    "kuka": kuka_dataset_transform,
    "taco_play": taco_play_dataset_transform,
    "jaco_play": jaco_play_dataset_transform,
    "berkeley_cable_routing": berkeley_cable_routing_dataset_transform,
    "roboturk": roboturk_dataset_transform,
    "nyu_door_opening_surprising_effectiveness": nyu_door_opening_dataset_transform,
    "viola": viola_dataset_transform,
    "berkeley_autolab_ur5": berkeley_autolab_ur5_dataset_transform,
    "toto": toto_dataset_transform,
    "language_table": language_table_dataset_transform,
    "columbia_cairlab_pusht_real": pusht_dataset_transform,
    "stanford_kuka_multimodal_dataset_converted_externally_to_rlds": stanford_kuka_multimodal_dataset_transform,
    "nyu_rot_dataset_converted_externally_to_rlds": nyu_rot_dataset_transform,
    "stanford_hydra_dataset_converted_externally_to_rlds": stanford_hydra_dataset_transform,
    "austin_buds_dataset_converted_externally_to_rlds": austin_buds_dataset_transform,
    "nyu_franka_play_dataset_converted_externally_to_rlds": nyu_franka_play_dataset_transform,
    "maniskill_dataset_converted_externally_to_rlds": maniskill_dataset_transform,
    "furniture_bench_dataset_converted_externally_to_rlds": furniture_bench_dataset_transform,
    "cmu_franka_exploration_dataset_converted_externally_to_rlds": cmu_franka_exploration_dataset_transform,
    "ucsd_kitchen_dataset_converted_externally_to_rlds": ucsd_kitchen_dataset_transform,
    "ucsd_pick_and_place_dataset_converted_externally_to_rlds": ucsd_pick_place_dataset_transform,
    "austin_sailor_dataset_converted_externally_to_rlds": austin_sailor_dataset_transform,
    "austin_sirius_dataset_converted_externally_to_rlds": austin_sirius_dataset_transform,
    "bc_z": bc_z_dataset_transform,
    "utokyo_pr2_opening_fridge_converted_externally_to_rlds": tokyo_pr2_opening_fridge_dataset_transform,
    "utokyo_pr2_tabletop_manipulation_converted_externally_to_rlds": tokyo_pr2_tabletop_manipulation_dataset_transform,
    "utokyo_xarm_pick_and_place_converted_externally_to_rlds": utokyo_xarm_pick_place_dataset_transform,
    "utokyo_xarm_bimanual_converted_externally_to_rlds": utokyo_xarm_bimanual_dataset_transform,
    "robo_net": robo_net_dataset_transform,
    "berkeley_mvp_converted_externally_to_rlds": berkeley_mvp_dataset_transform,
    "berkeley_rpt_converted_externally_to_rlds": berkeley_rpt_dataset_transform,
    "kaist_nonprehensile_converted_externally_to_rlds": kaist_nonprehensible_dataset_transform,
    "stanford_mask_vit_converted_externally_to_rlds": stanford_mask_vit_dataset_transform,
    "tokyo_u_lsmo_converted_externally_to_rlds": tokyo_lsmo_dataset_transform,
    "dlr_sara_pour_converted_externally_to_rlds": dlr_sara_pour_dataset_transform,
    "dlr_sara_grid_clamp_converted_externally_to_rlds": dlr_sara_grid_clamp_dataset_transform,
    "dlr_edan_shared_control_converted_externally_to_rlds": dlr_edan_shared_control_dataset_transform,
    "asu_table_top_converted_externally_to_rlds": asu_table_top_dataset_transform,
    "stanford_robocook_converted_externally_to_rlds": robocook_dataset_transform,
    "imperialcollege_sawyer_wrist_cam": imperial_wristcam_dataset_transform,
    "iamlab_cmu_pickup_insert_converted_externally_to_rlds": iamlab_pick_insert_dataset_transform,
    "uiuc_d3field": uiuc_d3field_dataset_transform,
    "utaustin_mutex": utaustin_mutex_dataset_transform,
    "berkeley_fanuc_manipulation": berkeley_fanuc_dataset_transform,
    "cmu_playing_with_food": cmu_playing_with_food_dataset_transform,
    "cmu_play_fusion": playfusion_dataset_transform,
    "cmu_stretch": cmu_stretch_dataset_transform,
    "berkeley_gnm_recon": gnm_dataset_transform,
    "berkeley_gnm_cory_hall": gnm_dataset_transform,
    "berkeley_gnm_sac_son": gnm_dataset_transform,
    # "droid": droid_baseact_transform,
    "droid": lambda x: x,
    "fmb": fmb_dataset_transform,
    "dobbe": dobbe_dataset_transform,
    "roboset": roboset_dataset_transform,
    "rh20t": rh20t_dataset_transform,
    ### T-DROID datasets
    "tdroid_carrot_in_bowl": tdroid_dataset_transform,
    "tdroid_pour_corn_in_pot": tdroid_dataset_transform,
    "tdroid_flip_pot_upright": tdroid_dataset_transform,
    "tdroid_move_object_onto_plate": tdroid_dataset_transform,
    "tdroid_knock_object_over": tdroid_dataset_transform,
    "tdroid_cover_object_with_towel": tdroid_dataset_transform,
    ### DROID Finetuning datasets
    # "droid_wipe": droid_finetuning_transform,
    ### LIBERO datasets (modified versions)
    "libero_spatial_no_noops": libero_dataset_transform,
    "libero_object_no_noops": libero_dataset_transform,
    "libero_goal_no_noops": libero_dataset_transform,
    "libero_10_no_noops": libero_dataset_transform,
    "libero_10_no_noops_mini": libero_dataset_transform,
    "libero_goal_no_noops_mini": libero_dataset_transform,
    "libero_goal_no_noops_half": libero_dataset_transform,
    "libero_10_no_noops_half": libero_dataset_transform,
    "libero_goal_no_noops_quad": libero_dataset_transform,
    "libero_10_no_noops_quad": libero_dataset_transform,
    "libero_combined": libero_dataset_transform,
    ### Human Dataset
    "ego4d_split_1": human_dataset_transform,
    "ego4d_split_2": human_dataset_transform,
    "ego4d_split_3": human_dataset_transform,
    "ego4d_split_4": human_dataset_transform,
    "sample_r1_lite": sample_r1_lite_dataset_transform,
    "agibot_dataset": agibot_dataset_transform,
}
