# ruff: noqa
import numpy as np
import tensorflow as tf
import tyro
from scipy.spatial.transform import Rotation as R
import sys
from openpi_client import image_tools

sys.path.append(".")
from shared import BaseEvalRunner, Args
from helpers import binarize_gripper_actions_np, euler_to_rot6d

class FrankaEvalRunner(BaseEvalRunner):

    def __init__(self, args):
        super().__init__(args)
        self.side_image_name = "right_image"

    def _extract_observation(self, obs_dict):
        image_observations = obs_dict["image"]
        # In addition to image observations, also capture the proprioceptive state
        robot_state = obs_dict["robot_state"]
        cartesian_position = np.array(robot_state["cartesian_position"])
        euler = cartesian_position[3:6].copy()
        cartesian_position = np.concatenate([cartesian_position[:3], euler_to_rot6d(euler)])
        gripper_position = np.array([robot_state["gripper_position"]])
        # print("Gripper position:", gripper_position)
        # gripper_position = binarize_gripper_actions_np(invert_gripper_actions_np(gripper_position), threshold=0.5)
        gripper_position = binarize_gripper_actions_np(gripper_position)


        right_image = image_observations["31425515_left"][:,:, :3][..., ::-1]
        wrist_image = image_observations["1"][::-1, ::-1, ::-1] # rotate 180

        if self.args.right_image_encoding == "tf_jpeg":
            right_tensor = tf.convert_to_tensor(right_image)
            if right_tensor.dtype != tf.uint8:
                right_tensor = tf.image.convert_image_dtype(right_tensor, tf.uint8, saturate=True)
            right_image = tf.io.decode_jpeg(tf.io.encode_jpeg(right_tensor, quality=95), channels=3).numpy()
        else:
            right_image = right_image

        if self.args.wrist_image_encoding == "tf_jpeg":
            wrist_tensor = tf.convert_to_tensor(wrist_image)
            if wrist_tensor.dtype != tf.uint8:
                wrist_tensor = tf.image.convert_image_dtype(wrist_tensor, tf.uint8, saturate=True)
            wrist_image = tf.io.decode_jpeg(tf.io.encode_jpeg(wrist_tensor, quality=95), channels=3).numpy()
        else:
            wrist_image = wrist_image


        return {
            "right_image": right_image,
            "wrist_image": wrist_image,
            "cartesian_position": cartesian_position,
            "euler": euler,
            "gripper_position": np.array(gripper_position),
            "state": np.concatenate([cartesian_position, gripper_position]),
            "joint_position": np.array(robot_state["joint_positions"]),
        }

    def get_action_from_response(self, response, curr_obs, use_quaternions=True):
        return super().get_action_from_response(response, curr_obs, use_quaternions=use_quaternions)


class FrankaUpstreamEvalRunner(FrankaEvalRunner):
    def __init__(self, args):
        super().__init__(args)

    def init_env(self):
        from droid.robot_env import RobotEnv
        return RobotEnv(
            action_space="joint_velocity",
            gripper_action_space="position",
        )
    
    def _extract_observation(self, obs_dict):
        image_observations = obs_dict["image"]
        # In addition to image observations, also capture the proprioceptive state
        robot_state = obs_dict["robot_state"]
        cartesian_position = np.array(robot_state["cartesian_position"])
        print("Gripper position:", gripper_position)
        gripper_position = np.array([robot_state["gripper_position"]])
        gripper_position = binarize_gripper_actions_np(gripper_position)


        right_image = image_observations["31425515_left"][:,:, :3][..., ::-1]
        wrist_image = image_observations["1"][:, :, ::-1]

        if self.args.right_image_encoding == "tf_jpeg":
            right_tensor = tf.convert_to_tensor(right_image)
            if right_tensor.dtype != tf.uint8:
                right_tensor = tf.image.convert_image_dtype(right_tensor, tf.uint8, saturate=True)
            right_image = tf.io.decode_jpeg(tf.io.encode_jpeg(right_tensor, quality=95), channels=3).numpy()
        else:
            right_image = right_image

        if self.args.wrist_image_encoding == "tf_jpeg":
            wrist_tensor = tf.convert_to_tensor(wrist_image)
            if wrist_tensor.dtype != tf.uint8:
                wrist_tensor = tf.image.convert_image_dtype(wrist_tensor, tf.uint8, saturate=True)
            wrist_image = tf.io.decode_jpeg(tf.io.encode_jpeg(wrist_tensor, quality=95), channels=3).numpy()
        else:
            wrist_image = wrist_image

        return {
            "right_image": right_image,
            "wrist_image": wrist_image,
            "cartesian_position": cartesian_position,
            "gripper_position": np.array(gripper_position),
            "state": np.concatenate([cartesian_position, gripper_position]),
            "joint_position": np.array(robot_state["joint_positions"]),
        }



    def obs_to_request(self, curr_obs, instruction):

        request = {
            "observation/exterior_image_1_left": image_tools.resize_with_pad(curr_obs[self.side_image_name], 224, 224),
            "observation/cartesian_position": curr_obs["cartesian_position"],
            "observation/gripper_position": curr_obs["gripper_position"],
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
        print("Running in base frame")
        eval_runner = FrankaEvalRunner(args)
        eval_runner.run()
    eval_runner = FrankaEvalRunner(args)
