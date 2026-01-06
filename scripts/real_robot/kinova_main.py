# ruff: noqa

import contextlib
import datetime
import faulthandler
import signal
import sys
import time
import cv2
import numpy as np
import tqdm
import tyro
from moviepy.editor import ImageSequenceClip
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from scipy.spatial.transform import Rotation as R
import zmq
import faulthandler
import numpy as np
from openpi_client import image_tools
import tyro
from scipy.spatial.transform import Rotation as R
import sys
import cv2
import zmq
sys.path.append(".")
from shared import BaseEvalRunner, Args, IMAGE_KEYS
from kinova_robot_env import RobotEnv

@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Temporarily prevent keyboard interrupts by delaying them until after the protected code."""
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


faulthandler.enable()
# DROID data collection frequency -- we slow down execution to match this frequency
DROID_CONTROL_FREQUENCY = 15


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


class KinovaEvalRunner(BaseEvalRunner):
    def __init__(self, args):
        super().__init__(args)
        self.side_image_name = f"{self.args.external_camera}_image"

    def init_env(self):
        return RobotEnv()

    def _extract_observation(self, obs_dict, save_to_disk=False):
        
        return {
            "right_image": obs_dict["base_image"][None],
            "wrist_image": obs_dict["wrist_image"][None],
            "cartesian_position": obs_dict["cartesian_position"],
            "gripper_position": obs_dict["gripper_position"],
            "state": obs_dict["state"],
            "joint_position": obs_dict["joint_position"],
        }

    def obs_to_request(self, curr_obs, instruction):
        request = {
            "observation": {
                IMAGE_KEYS[0]: image_tools.resize_with_pad(curr_obs[self.side_image_name], 224, 224),
                "state": curr_obs["state"],
            },
            "prompt": instruction,
            "batch_size": None,
        }
        if self.args.use_wrist_camera:
            request["observation"][IMAGE_KEYS[1]] = image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224)
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
