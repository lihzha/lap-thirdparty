# ruff: noqa
import numpy as np
import collections
import tyro
import sys
import dm_env
from PIL import Image

# Aloha / Interbotix imports
from aloha.real_env import RealEnv
from aloha.robot_utils import load_yaml_file
from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    get_interbotix_global_node,
    robot_startup,
    robot_shutdown,
    InterbotixRobotNode
)
from interbotix_common_modules.common_robot.exceptions import InterbotixException
import interbotix_common_modules.angle_manipulation as ang

# Shared / OpenPI imports
from openpi.training import config
from openpi_client import image_tools
sys.path.append(".")
from shared import BaseEvalRunner, Args
from helpers import binarize_gripper_actions_np, euler_to_rot6d, invert_gripper_actions_np

class AlohaEvalRunner(BaseEvalRunner):
    def __init__(self, args):
        super().__init__(args)
        self.env = None
        self.left_arm = None
        
    def init_env(self):
        try:
            node = get_interbotix_global_node()
            if node is None:
                node = create_interbotix_global_node('aloha')
        except:
            node = create_interbotix_global_node('aloha')

        # Load the left-only configuration
        real_config = load_yaml_file(
            config_type='robot', 
            name='aloha_left_only', 
            base_path='/home/aloha/interbotix_ws/src/aloha/config'
        ).get('robot', {})

        # Use standard RealEnv (Joint Space)
        self.env = RealEnv(
            node=node,
            setup_robots=True,
            setup_base=False,
            config=real_config
        )

        try:
            robot_startup(node)
        except InterbotixException:
            pass

        ts = self.env.reset()
        
        breakpoint()
        return self.env

    def _extract_observation(self, obs_dict, save_to_disk=False):
        image_observations = obs_dict["images"]

        # Camera keys may vary based on your config, usually 'camera_high' and 'camera_wrist_left'
        top_image = image_observations["camera_high"]
        left_image = image_observations["camera_wrist_left"]

        # Cropping/Resizing
        start_col, end_col = 104, 744
        top_image = top_image[:, start_col:end_col]
        left_image = left_image[:, start_col:end_col]
        left_image = image_tools.resize_with_pad(left_image, 224, 224)
        top_image = image_tools.resize_with_pad(top_image, 224, 224)

        # Standard Aloha rotations
        left_image = left_image[:, :, ::-1]
        top_image = top_image.transpose(1, 0, 2)[::-1, :, ::-1]

        if save_to_disk:
            combined_image = np.concatenate([top_image, left_image], axis=1)
            Image.fromarray(combined_image).save("robot_camera_views.png")

        # Get Cartesian State for the model if required
        R_T_curr = self.left_arm.get_ee_pose()
        cartesian_xyz = R_T_curr[:3, 3]
        euler = ang.rotationMatrixToEulerAngles(R_T_curr[:3, :3])
        cartesian_position = np.concatenate([cartesian_xyz, euler_to_rot6d(euler)])
        
        # Gripper handling (assumes left-only joint index 6)
        gripper_pos = np.array([obs_dict['qpos'][6]])
        gripper_binary = binarize_gripper_actions_np(invert_gripper_actions_np(gripper_pos), threshold=0.5)

        return {
            "right_image": top_image, # 'right' often refers to high cam in some OpenPI configs
            "wrist_image": left_image,
            "state": np.concatenate([cartesian_position, gripper_binary]),
            "cartesian_position": cartesian_position,
            "gripper_position": gripper_binary,
            "euler": euler,
            "qpos": obs_dict['qpos']
        }

    def run_step(self, action):
        """
        Takes a Cartesian action [dx, dy, dz, droll, dpitch, dyaw, gripper_target]
        and converts it to a Joint Space step for RealEnv.
        """
        # 1. Calculate target pose matrix
        current_T = self.left_arm.get_ee_pose()
        R_curr = current_T[:3, :3]
        R_delta = ang.eulerAnglesToRotationMatrix(action[3:6])
        
        target_R = R_curr @ R_delta
        target_xyz = current_T[:3, 3] + action[:3]
        
        target_T = np.eye(4)
        target_T[:3, :3] = target_R
        target_T[:3, 3] = target_xyz

        # 2. Local IK Call (execute=False)
        # This solves IK but doesn't send the command yet
        joint_targets, success = self.left_arm.set_ee_pose_matrix(
            target_T, 
            execute=False, 
            blocking=False
        )

        if success:
            # 3. Construct joint action [q0, q1, q2, q3, q4, q5, gripper]
            # action[6] is the gripper value from your model
            full_joint_action = np.append(joint_targets, action[6])
            
            # 4. Step the real environment
            ts = self.env.step(full_joint_action)
            return ts
        else:
            print("[WARN] IK Solution not found. Skipping step.")
            return self.env.get_observation()

if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    eval_runner = AlohaEvalRunner(args)
    
    try:
        eval_runner.run()
    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt received. Cleaning up...")
    finally:
        if hasattr(eval_runner, 'env') and eval_runner.env:
            for bot in eval_runner.env.follower_bots:
                bot.arm.core.robot_stop_moving()
        
        node = get_interbotix_global_node()
        if node:
            robot_shutdown(node)
        print("[INFO] Shutdown complete. Safe to restart.")