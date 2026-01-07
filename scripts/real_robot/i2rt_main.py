# ruff: noqa
import numpy as np
import tensorflow as tf
import tyro
from scipy.spatial.transform import Rotation as R
import sys
from openpi_client import image_tools

sys.path.append(".")
from shared import BaseEvalRunner, Args, IMAGE_KEYS
from helpers import binarize_gripper_actions_np, euler_to_rot6d
from PIL import Image

def _encode_image_if_needed(image, encoding: str):
    if encoding != "tf_jpeg":
        return image
    tensor = tf.convert_to_tensor(image)
    if tensor.dtype != tf.uint8:
        tensor = tf.image.convert_image_dtype(tensor, tf.uint8, saturate=True)
    return tf.io.decode_jpeg(tf.io.encode_jpeg(tensor, quality=95), channels=3).numpy()


class I2rtEvalRunner(BaseEvalRunner):

    CHUNK_STEPS=8

    def __init__(self, args):
        super().__init__(args)
        self.side_image_name = "right_image"
    
    def init_env(self):
        from i2rt.ee_pose_env_real import RealRobotEndEffectorEnv, EnvConfig
        return RealRobotEndEffectorEnv(EnvConfig(control_mode="absolute"))
    
    def process_gripper_action(self, action, curr_obs):
        return curr_obs["gripper_position"] if len(action) == 6 else action[..., -1]
    

    def _extract_observation(self, obs_dict, save_to_disk=False):
        image_observations = obs_dict["image"]
        # In addition to image observations, also capture the proprioceptive state
        robot_state = obs_dict["robot_state"]
        cartesian_position = np.array(robot_state["cartesian_position"])
        euler = cartesian_position[3:6].copy()
        # cartesian_position = np.concatenate([cartesian_position[:3], euler])
        # cartesian_position = np.concatenate([cartesian_position[:3], euler])
        cartesian_position = np.concatenate([cartesian_position[:3], euler_to_rot6d(euler)])
        gripper_position = np.array([robot_state["gripper_position"]])
        # print("Gripper position:", gripper_position)
        # gripper_position = binarize_gripper_actions_np(invert_gripper_actions_np(gripper_position), threshold=0.5)
        gripper_position = binarize_gripper_actions_np(gripper_position)

        right_image = image_observations["31425515_left"][:,:, :3][..., ::-1]
        # wrist_image = image_observations["1"][::-1, ::-1, ::-1] # rotate 180
        right_image = _encode_image_if_needed(right_image, self.args.right_image_encoding)
        # wrist_image = _encode_image_if_needed(wrist_image, self.args.wrist_image_encoding)

        # wrist_image = wrist_image[-360:]

        right_image = image_tools.resize_with_pad(right_image, 224, 224)
        # wrist_image = image_tools.resize_with_pad(wrist_image, 224, 224)

        if save_to_disk:
            combined_image = Image.fromarray(right_image)
            combined_image.save("robot_camera_views.png")

        return {
            "right_image": right_image,
            "cartesian_position": cartesian_position,
            "euler": euler,
            "gripper_position": np.array(gripper_position),
            "state": np.concatenate([cartesian_position, gripper_position]),
            "joint_position": np.array(robot_state["joint_positions"]),
        }
    
    def obs_to_request(self, curr_obs, instruction):
        request = {
            "observation": {
                IMAGE_KEYS[0]: curr_obs[self.side_image_name],
                "cartesian_position": curr_obs["cartesian_position"],
                "gripper_position": curr_obs["gripper_position"],
                "joint_position": curr_obs["joint_position"],
                "state": curr_obs["state"],
            },
            "prompt": instruction,
            "batch_size": None,
        }
        return request

class I2rtUpstreamEvalRunner(I2rtEvalRunner):
    def __init__(self, args):
        super().__init__(args)

    def init_env(self):
        from i2rt.ee_pose_env_real import RealRobotEndEffectorEnv, EnvConfig
        return RealRobotEndEffectorEnv(EnvConfig())
    
    def _extract_observation(self, obs_dict, save_to_disk=False):
        image_observations = obs_dict["image"]
        # In addition to image observations, also capture the proprioceptive state
        robot_state = obs_dict["robot_state"]
        cartesian_position = np.array(robot_state["cartesian_position"])
        gripper_position = np.array([robot_state["gripper_position"]])
        gripper_position = 1- binarize_gripper_actions_np(gripper_position)

        right_image = image_observations["31425515_left"][:,:, :3][..., ::-1]
        # wrist_image = image_observations["1"][:, :, ::-1]
        right_image = _encode_image_if_needed(right_image, self.args.right_image_encoding)
        # wrist_image = _encode_image_if_needed(wrist_image, self.args.wrist_image_encoding)

        right_image = image_tools.resize_with_pad(right_image, 224, 224)
        # wrist_image = image_tools.resize_with_pad(wrist_image, 224, 224)

        if save_to_disk:
            combined_image = Image.fromarray(right_image)
            combined_image.save("robot_camera_views.png")

        return {
            "right_image": right_image,
            "cartesian_position": cartesian_position,
            "gripper_position": np.array(gripper_position),
            "state": np.concatenate([cartesian_position, gripper_position]),
            "joint_position": np.array(robot_state["joint_positions"]),
        }



    def obs_to_request(self, curr_obs, instruction):

        request = {
            "observation/exterior_image_1_left": curr_obs[self.side_image_name],
            "observation/cartesian_position": curr_obs["cartesian_position"],
            "observation/gripper_position": curr_obs["gripper_position"],
            "observation/joint_position": curr_obs["joint_position"],
            "prompt": instruction,
        }
        # if self.args.use_wrist_camera:
        #     request["observation/wrist_image_left"] =curr_obs["wrist_image"]
        return request


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    print(args)
    if args.run_upstream:
        eval_runner = I2rtUpstreamEvalRunner(args)
        eval_runner.run_upstream()
    else:
        print("Running in base frame")
        eval_runner = I2rtEvalRunner(args)
        eval_runner.run()
