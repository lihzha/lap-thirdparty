# ruff: noqa
import numpy as np
from openpi_client import image_tools
from aloha.real_env import RealEnv
import tyro
import sys

sys.path.append(".")
from shared import BaseEvalRunner, Args
from PIL import Image

from helpers import binarize_gripper_actions_np, euler_to_rot6d, invert_gripper_actions_np

class AlohaEvalRunner(BaseEvalRunner):
    def __init__(self, args):
        super().__init__(args)

    def _extract_observation(self, obs_dict, save_to_disk=False):
        image_observations = obs_dict["image"]
        base_image = image_observations["cam_high"]
        right_image = image_observations["cam_right_wrist"]
        # left_image = left_image[..., ::-1]
        right_image = image_tools.resize_with_pad(right_image[..., ::-1], 224, 224)
        wrist_image = image_tools.resize_with_pad(wrist_image[..., ::-1], 224, 224)
        if save_to_disk:
            combined_image = np.concatenate([base_image, right_image], axis=1)
            combined_image = Image.fromarray(combined_image)
            combined_image.save("robot_camera_views.png")

        # wrist_image = wrist_image[..., ::-1]
        # In addition to image observations, also capture the proprioceptive state
        robot_state = obs_dict["robot_state"]
        cartesian_position = np.array(robot_state["cartesian_position"])
        euler = cartesian_position[3:6].copy()
        cartesian_position = np.concatenate([cartesian_position[:3], euler_to_rot6d(euler)])
        joint_position = np.array(robot_state["joint_positions"])
        gripper_position = np.array([robot_state["gripper_position"]])
        gripper_position = binarize_gripper_actions_np(invert_gripper_actions_np(gripper_position), threshold=0.5)
        # Create one combined image to make live viewing easy
        return {
            "right_image": right_image,
            "wrist_image": wrist_image,
            "cartesian_position": cartesian_position,
            "joint_position": joint_position,
            "gripper_position": gripper_position,
            "state": np.concatenate([cartesian_position, gripper_position]),
            "euler": euler,
        }


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    print(args)
    eval_runner = AlohaEvalRunner(args)
    eval_runner.run_upstream()
  
