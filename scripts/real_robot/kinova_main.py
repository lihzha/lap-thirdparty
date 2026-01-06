# ruff: noqa

import tyro
from openpi_client import image_tools
import numpy as np
import sys
from PIL import Image
sys.path.append(".")
from shared import BaseEvalRunner, Args, IMAGE_KEYS
from kinova_robot_env import RobotEnv

class KinovaEvalRunner(BaseEvalRunner):
    def __init__(self, args):
        super().__init__(args)
        self.side_image_name = "right_image"

    def init_env(self):
        return RobotEnv()

    def _extract_observation(self, obs_dict, save_to_disk=False):

        right_image = image_tools.resize_with_pad(obs_dict["base_image"], 224, 224)
        wrist_image = image_tools.resize_with_pad(obs_dict["wrist_image"], 224, 224)

        if save_to_disk:
            combined_image = np.concatenate([right_image, wrist_image], axis=1)
            combined_image = Image.fromarray(combined_image)
            combined_image.save("robot_camera_views.png")
        
        return {
            "right_image": right_image,
            "wrist_image": wrist_image,
            "cartesian_position": obs_dict["cartesian_position"],
            "gripper_position": obs_dict["gripper_position"],
            "state": obs_dict["state"],
            "joint_position": obs_dict["joint_position"],
        }

    def obs_to_request(self, curr_obs, instruction):
        request = {
            "observation": {
                IMAGE_KEYS[0]: curr_obs[self.side_image_name],
                "state": curr_obs["state"],
            },
            "prompt": instruction,
            "batch_size": None,
        }
        if self.args.use_wrist_camera:
            request["observation"][IMAGE_KEYS[1]] = curr_obs["wrist_image"]
        return request


class KinovaUpstreamEvalRunner(KinovaEvalRunner):
    def __init__(self, args):
        super().__init__(args)

    def init_env(self):
        return RobotEnv()
    
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
        eval_runner = KinovaUpstreamEvalRunner(args)
        eval_runner.run_upstream()
    else:
        
        print("Running in base frame")
        eval_runner = KinovaEvalRunner(args)
        eval_runner.run()
