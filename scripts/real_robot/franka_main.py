# ruff: noqa


import dataclasses
import faulthandler
import numpy as np
from droid.robot_env import RobotEnv
import tyro
from scipy.spatial.transform import Rotation as R
import sys

sys.path.append(".")
from shared import BaseEvalRunner, Args
from openpi_client import image_tools
import cv2

faulthandler.enable()


def binarize_gripper_actions_np(actions: np.ndarray, threshold: float = 0.95) -> np.ndarray:
    """
    Convert continuous gripper actions to binary (0 or 1) using backward propagation logic.
    """
    actions = actions.astype(np.float32)
    n = actions.shape[0]
    new_actions = np.zeros_like(actions)

    open_mask = actions > threshold
    closed_mask = actions < (1 - threshold)
    in_between_mask = ~(open_mask | closed_mask)

    carry = actions[-1] > threshold  # carry as boolean (True=open)

    for i in reversed(range(n)):
        if not in_between_mask[i]:
            carry = open_mask[i]
        new_actions[i] = float(carry)

    return new_actions


def invert_gripper_actions_np(actions: np.ndarray) -> np.ndarray:
    """Invert gripper binary actions: 1 → 0, 0 → 1."""
    return 1.0 - actions

class FrankaEvalRunner(BaseEvalRunner):
    robot_type: str = "panda"

    def __init__(self, args):
        super().__init__(args)
        self.side_image_name = "right_image"

    def init_env(self):
        return RobotEnv(
            robot_type=self.robot_type,
            action_space="cartesian_position",
            gripper_action_space="position",
        )

    def _extract_observation(self, obs_dict, *, save_to_disk=False):
        image_observations = obs_dict["image"]
        # In addition to image observations, also capture the proprioceptive state
        robot_state = obs_dict["robot_state"]
        cartesian_position = np.array(robot_state["cartesian_position"])
        gripper_position = np.array([robot_state["gripper_position"]])
        # gripper_position = binarize_gripper_actions_np(invert_gripper_actions_np(gripper_position), threshold=0.5)
        gripper_position = binarize_gripper_actions_np(gripper_position)

        print("Gripper position:", gripper_position)

        # if gripper_position > 0.2:
        #     gripper_position = 1.0
        # else:
        #     gripper_position = 0.0

        right_image = image_observations["31425515_left"][:,:, :3][..., ::-1]
        wrist_image = image_observations["1"][::-1, ::-1, ::-1] # rotate 180
        # wrist_image = image_observations["1"][:, :, ::-1]

        if not self.args.use_raw:
            right_image = right_image[None]
            wrist_image = wrist_image[None]

        return {
            # "right_image": image_observations["0"][..., ::-1][None],  # Convert BGR to RGB
            "right_image": right_image,
            "wrist_image": wrist_image,
            # "wrist_image": np.rot90(image_observations["1"][..., ::-1], k=1), # rotate 90 degrees
            # "wrist_image": np.rot90(image_observations["14846828_left"][::-1,:, :3][..., ::-1], k=0)[None], # flip
            # "wrist_image": np.rot90(image_observations["14846828_left"][..., :3][..., ::-1], k=2)[None], # rotate 180 degrees
            # "wrist_image": image_observations["1"][..., ::-1][::-1],  # flip vertically, up -> dowm
            # "wrist_image": rotate_x_degrees(image_observations["1"][..., ::-1], -15.5),
            # "wrist_image": image_observations["14846828_left"][..., :3][..., ::-1],  # drop alpha channel and convert BGR to RGB
            "cartesian_position": cartesian_position,
            "gripper_position": np.array(gripper_position),
            "state": np.concatenate([cartesian_position, gripper_position]),
            "joint_position": np.array(robot_state["joint_positions"]),
        }

    def get_action_from_response(self, response, curr_obs, use_quaternions=True):
        return super().get_action_from_response(response, curr_obs, use_quaternions=use_quaternions)


def rotate_x_degrees(img, degrees):
    # Rotation parameters
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)

    # Compute rotation matrix
    M = cv2.getRotationMatrix2D(center, degrees, 1.0)

    # Compute new bounding box to fit entire image
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int((h * sin_a) + (w * cos_a))
    new_h = int((h * cos_a) + (w * sin_a))

    # Adjust rotation matrix to account for translation
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # Rotate with black padding
    rotated = cv2.warpAffine(
        img, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0)
    )
    cv2.imwrite("rotated_debug.png", rotated)
    return rotated


class FrankaExtrEvalRunner(FrankaEvalRunner):
    def __init__(self, args):
        super().__init__(args)

    def set_extrinsics(self):
        extrinsics = [
            0.19344852600560863,
            -0.4704189157280809,
            0.968999457340307,
            -2.3815317980812005,
            0.1557806728117621,
            -0.6502647332046341,
        ]
        # Turn (pos, quat_wxyz) into 4x4 cam->base extrinsics matrix
        pos = np.array(extrinsics[:3], dtype=float)
        roll, pitch, yaw = extrinsics[3:6]
        r = R.from_euler("xyz", [roll, pitch, yaw], degrees=False)
        quat_xyzw = r.as_quat()  # (x,y,z,w)
        rot_mat = R.from_quat(quat_xyzw).as_matrix()
        cam_to_base_extrinsics_matrix = np.eye(4, dtype=float)
        cam_to_base_extrinsics_matrix[:3, :3] = rot_mat
        cam_to_base_extrinsics_matrix[:3, 3] = pos
        return cam_to_base_extrinsics_matrix


class FrankaUpstreamEvalRunner(FrankaEvalRunner):
    def __init__(self, args):
        super().__init__(args)

    def init_env(self):
        return RobotEnv(
            robot_type=self.robot_type,
            action_space="joint_velocity",
            gripper_action_space="position",
        )
    
    def _extract_observation(self, obs_dict, *, save_to_disk=False):
        image_observations = obs_dict["image"]
        # In addition to image observations, also capture the proprioceptive state
        robot_state = obs_dict["robot_state"]
        cartesian_position = np.array(robot_state["cartesian_position"])
        gripper_position = np.array([robot_state["gripper_position"]])
        gripper_position = binarize_gripper_actions_np(gripper_position)

        print("Gripper position:", gripper_position)

        right_image = image_observations["31425515_left"][:,:, :3][..., ::-1]
        wrist_image = image_observations["1"][..., ::-1]

        return {
            "right_image": right_image,
            "wrist_image": wrist_image,
            "cartesian_position": cartesian_position,
            "gripper_position": np.array(gripper_position),
            "state": np.concatenate([cartesian_position, gripper_position]),
            "joint_position": np.array(robot_state["joint_positions"]),
        }



    def obs_to_request(self, curr_obs, instruction):

        gripper_position = curr_obs["gripper_position"]

        request = {
            "observation/exterior_image_1_left": image_tools.resize_with_pad(curr_obs[self.side_image_name], 224, 224),
            "observation/cartesian_position": curr_obs["cartesian_position"],
            "observation/gripper_position": gripper_position,
            "observation/joint_position": curr_obs["joint_position"],
            "prompt": instruction,
        }
        if self.args.use_wrist_camera:
            request["observation/wrist_image_left"] = image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224)
        return request


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    print(args)
    if args.run_upstream:
        eval_runner = FrankaUpstreamEvalRunner(args)
        eval_runner.run_upstream()
    else:
        if args.in_camera_frame:
            print("Running in camera frame")
            eval_runner = FrankaExtrEvalRunner(args)
        else:
            print("Running in base frame")
            eval_runner = FrankaEvalRunner(args)
        eval_runner.run()
    eval_runner = FrankaEvalRunner(args)
