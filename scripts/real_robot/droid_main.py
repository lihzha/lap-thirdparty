# ruff: noqa
import numpy as np
from openpi_client import image_tools
from droid.robot_env import RobotEnv
import tyro
from scipy.spatial.transform import Rotation as R
import sys

sys.path.append(".")
from shared import BaseEvalRunner, Args
from helpers import binarize_gripper_actions_np, invert_gripper_actions_np


class DroidEvalRunner(BaseEvalRunner):
    def __init__(self, args):
        super().__init__(args)
        self.side_image_name = f"{self.args.external_camera}_image"

    def _extract_observation(self, obs_dict):
        image_observations = obs_dict["image"]
        left_image, right_image, wrist_image = None, None, None
        for key in image_observations:
            # Note the "left" below refers to the left camera in the stereo pair.
            # The model is only trained on left stereo cams, so we only feed those.
            if self.args.left_camera_id in key and "left" in key:
                left_image = image_observations[key]
            elif self.args.right_camera_id in key and "left" in key:
                right_image = image_observations[key]
            elif self.args.wrist_camera_id in key and "left" in key:
                wrist_image = image_observations[key]
        # Drop the alpha dimension
        # left_image = left_image[..., :3]
        right_image = right_image[..., :3]
        wrist_image = wrist_image[..., :3]
        # Convert to RGB
        # left_image = left_image[..., ::-1]
        right_image = right_image[..., ::-1]
        wrist_image = wrist_image[::-1, ::-1, ::-1]
        # In addition to image observations, also capture the proprioceptive state
        robot_state = obs_dict["robot_state"]
        cartesian_position = np.array(robot_state["cartesian_position"])
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
        }

class DroidUpstreamEvalRunner(DroidEvalRunner):
    def __init__(self, args):
        super().__init__(args)

    def init_env(self):
        return RobotEnv(
            action_space="joint_velocity",
            gripper_action_space="position",
        )
    
    def obs_to_request(self, curr_obs, instruction):

        request = {
            "observation/exterior_image_1_left": image_tools.resize_with_pad(
                    curr_obs[self.side_image_name][0], 224, 224
                ),
            "observation/cartesian_position": curr_obs["cartesian_position"],
            "observation/gripper_position": curr_obs["gripper_position"],
            "observation/joint_position": curr_obs["joint_position"],
            "prompt": instruction,
        }
        if self.args.use_wrist_camera:
            request["observation/wrist_image_left"] = image_tools.resize_with_pad(curr_obs["wrist_image"][0], 224, 224)
        return request


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    print(args)
    if args.run_upstream:
        eval_runner = DroidUpstreamEvalRunner(args)
        eval_runner.run_upstream()
    else:
        print("Running in base frame")
        eval_runner = DroidEvalRunner(args)
        eval_runner.run()
