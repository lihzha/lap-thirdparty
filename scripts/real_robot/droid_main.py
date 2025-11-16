# ruff: noqa
import faulthandler
import numpy as np
from openpi_client import image_tools
from droid.robot_env import RobotEnv
import tyro
from scipy.spatial.transform import Rotation as R
import sys
import cv2

sys.path.append(".")
from shared import BaseEvalRunner, Args, IMAGE_KEYS

AXIS_PERM = np.array([0, 2, 1])
AXIS_SIGN = np.array([1, 1, 1])
faulthandler.enable()
# DROID data collection frequency -- we slow down execution to match this frequency
DROID_CONTROL_FREQUENCY = 15


import numpy as np

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


class DroidEvalRunner(BaseEvalRunner):
    def __init__(self, args):
        super().__init__(args)
        self.side_image_name = f"{self.args.external_camera}_image"

    def init_env(self):
        return RobotEnv(
            action_space="cartesian_position",
            gripper_action_space="position",
        )

    def _extract_observation(self, obs_dict, save_to_disk=False):
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
        wrist_image = wrist_image[..., ::-1]
        # wrist_image = np.rot90(wrist_image, k=2)
        # wrist_image = wrist_image[::-1]
        # In addition to image observations, also capture the proprioceptive state
        robot_state = obs_dict["robot_state"]
        cartesian_position = np.array(robot_state["cartesian_position"])
        joint_position = np.array(robot_state["joint_positions"])
        gripper_position = np.array([robot_state["gripper_position"]])
        gripper_position = binarize_gripper_actions_np(invert_gripper_actions_np(gripper_position), threshold=0.5)
        # Save the images to disk so that they can be viewed live while the robot is running
        # Create one combined image to make live viewing easy
        return {
            # "left_image": left_image,
            "right_image": right_image[None],
            "wrist_image": wrist_image[None],
            "cartesian_position": cartesian_position,
            "joint_position": joint_position,
            "gripper_position": gripper_position,
            "state": np.concatenate([cartesian_position, gripper_position]),
        }
    # def _extract_observation(self, obs_dict, save_to_disk=False):
    #     image_observations = obs_dict["image"]
    #     left_image, right_image, wrist_image = None, None, None
    #     for key in image_observations:
    #         # Note the "left" below refers to the left camera in the stereo pair.
    #         # The model is only trained on left stereo cams, so we only feed those.
    #         if self.args.wrist_camera_id in key and "left" in key:
    #             wrist_image = image_observations[key]
    #         if key == "0":
    #             right_image = image_observations[key]
    #     # Drop the alpha dimension
    #     # left_image = left_image[..., :3]
    #     right_image = right_image[..., :3]
    #     wrist_image = wrist_image[..., :3]
    #     # Convert to RGB
    #     # left_image = left_image[..., ::-1]
    #     right_image = right_image[..., ::-1]
    #     wrist_image = wrist_image[..., ::-1]
    #     # In addition to image observations, also capture the proprioceptive state
    #     robot_state = obs_dict["robot_state"]
    #     cartesian_position = np.array(robot_state["cartesian_position"])
    #     joint_position = np.array(robot_state["joint_positions"])
    #     gripper_position = np.array([robot_state["gripper_position"]])
    #     # Save the images to disk so that they can be viewed live while the robot is running
    #     # Create one combined image to make live viewing easy
    #     return {
    #         # "left_image": left_image,
    #         "right_image": right_image,
    #         "wrist_image": wrist_image,
    #         "cartesian_position": cartesian_position,
    #         "joint_position": joint_position,
    #         "gripper_position": gripper_position,
    #         "state": np.concatenate([cartesian_position, gripper_position]),
    #     }



class DroidExtrEvalRunner(DroidEvalRunner):
    def __init__(self, args):
        super().__init__(args)

    def set_extrinsics(self):
        extrinsics = [
            0.15297898357307485,
            -0.46533509871932016,
            0.488261593272068,
            -2.1393635673966935,
            -0.01094009086539871,
            -0.7221553756328747,
        ]
        if len(extrinsics) == 6:
            extrinsics = extrinsics[:3] + R.from_euler("xyz", extrinsics[3:6], degrees=False).as_quat().tolist()
        # Turn (pos, quat_wxyz) into 4x4 cam->base extrinsics matrix
        pos = np.array(extrinsics[:3], dtype=float)
        x, y, z, w = extrinsics[3:7]
        quat_xyzw = np.array([x, y, z, w], dtype=float)  # convert (w,x,y,z) -> (x,y,z,w)
        rot_mat = R.from_quat(quat_xyzw).as_matrix()
        cam_to_base_extrinsics_matrix = np.eye(4, dtype=float)
        cam_to_base_extrinsics_matrix[:3, :3] = rot_mat
        cam_to_base_extrinsics_matrix[:3, 3] = pos
        return cam_to_base_extrinsics_matrix


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
        if args.in_camera_frame:
            print("Running in camera frame")
            eval_runner = DroidExtrEvalRunner(args)
        else:
            print("Running in base frame")
            eval_runner = DroidEvalRunner(args)
        eval_runner.run()
