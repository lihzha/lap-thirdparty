# ruff: noqa
import numpy as np
import collections
import tyro
import sys
import dm_env
import time
import rclpy
from PIL import Image
from typing import Dict

# Aloha / Interbotix imports
from aloha.real_env import make_real_env, RealEnv
from aloha.robot_utils import (
    FOLLOWER_GRIPPER_JOINT_CLOSE,
    load_yaml_file,
    move_arms,
    move_grippers,
    START_ARM_POSE,
    torque_on,
)
from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    robot_shutdown,
    robot_startup,
)
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
import interbotix_common_modules.angle_manipulation as ang

# Shared / OpenPI imports
from openpi.training import config as openpi_config # Avoid collision with local var
from openpi_client import image_tools
sys.path.append(".")
from shared import BaseEvalRunner, Args
from helpers import binarize_gripper_actions_np, euler_to_rot6d, invert_gripper_actions_np

def opening_ceremony(robots: Dict[str, InterbotixManipulatorXS], dt: float) -> None:
    """
    Stabilized initialization logic ported from record_episodes.py.
    Clears hardware errors and sets the gripper limits.
    """
    follower_bots = {name: bot for name, bot in robots.items() if "follower" in name}
    
    for name, follower_bot in follower_bots.items():
        print(f"[INFO] Initializing {name}...")
        # Hard Reset (Clears hardware errors from yesterday)
        follower_bot.core.robot_reboot_motors("single", "gripper", True)
        follower_bot.core.robot_set_operating_modes("group", "arm", "position")
        follower_bot.core.robot_set_operating_modes("single", "gripper", "current_based_position")
        follower_bot.core.robot_set_motor_registers("single", "gripper", "current_limit", 300)

        # Enable torque
        torque_on(follower_bot)

        # Move follower to the starting arm position
        start_arm_qpos = START_ARM_POSE[:6]
        move_arms(
            bot_list=[follower_bot],
            target_pose_list=[start_arm_qpos],
            moving_time=4.0,
            dt=dt,
        )

        # Move follower gripper to the starting position
        move_grippers(
            [follower_bot],
            [FOLLOWER_GRIPPER_JOINT_CLOSE],
            moving_time=0.5,
            dt=dt,
        )
    print("[INFO] Opening ceremony complete. Robots torqued and ready.")

class AlohaEvalRunner(BaseEvalRunner):
    def __init__(self, args):
        self.dt = 1 / 50  # 50Hz for Dynamixel bus stability
        # super().__init__ triggers self.init_env()
        super().__init__(args)

    def init_env(self):
        print("[INFO] Initializing Hardware (Trossen-style)...")
        
        # 1. Initialize ROS node
        self.node = create_interbotix_global_node("aloha")
        
        # 2. Load yaml config
        robot_config = load_yaml_file(
            config_type='robot', 
            name='aloha_left_only', 
            base_path='/home/aloha/interbotix_ws/src/aloha/config'
        ).get('robot', {})

        # 3. Create the environment
        # We store it in self.env because BaseEvalRunner expects that attribute
        self.env = make_real_env(
            node=self.node,
            setup_robots=False, # We handle this in ceremony
            setup_base=False,
            torque_base=False,
            config=robot_config,
        )
        
        # 4. Startup Handshake
        robot_startup(self.node)
        opening_ceremony(self.env.robots, dt=self.dt)

        # 5. Reference for IK
        self.left_arm = next(bot for name, bot in self.env.robots.items() if "follower" in name).arm
        
        # 6. MONKEY PATCH: Redirect env.step to our local IK-aware method
        # This allows BaseEvalRunner._rollout_once to call self.env.step(action)
        # while actually executing our joint-space logic.
        self.env.step = self.ik_step
        
        return self.env

    def ik_step(self, action):
        """
        Intercepts Cartesian actions from the policy, converts to joint space via IK,
        and executes at a stable 50Hz frequency.
        """
        t0 = time.time()
        
        # A. Cartesian to Pose Matrix
        current_T = self.left_arm.get_ee_pose()
        R_delta = ang.eulerAnglesToRotationMatrix(action[3:6])
        target_T = np.eye(4)
        target_T[:3, :3] = current_T[:3, :3] @ R_delta
        target_T[:3, 3] = current_T[:3, 3] + action[:3]

        # B. Solve IK (Execute=False performs math locally)
        joint_targets, success = self.left_arm.set_ee_pose_matrix(
            target_T, execute=False, blocking=False
        )

        if success:
            # action[6] is the binarized gripper command from the base class
            full_joint_action = np.append(joint_targets, action[6])
            
            # C. Step the hardware using the original RealEnv.step method
            # We call type(self.env).step to avoid recursion with our patch
            ts = RealEnv.step(self.env, full_joint_action)
            
            # D. Frequency Limit: Ensures we don't overwhelm the Dynamixel bus
            time.sleep(max(0, self.dt - (time.time() - t0)))
            return ts
        else:
            print("[WARN] IK Solution not found - arm staying in place.")
            return self.env.get_observation()

    def _extract_observation(self, obs_dict, save_to_disk=False):
        image_observations = obs_dict["images"]
        top_image = image_observations["camera_high"]
        left_image = image_observations["camera_wrist_left"]

        # Cropping and resizing for model input
        start_col, end_col = 104, 744
        top_image = top_image[:, start_col:end_col]
        left_image = left_image[:, start_col:end_col]
        left_image = image_tools.resize_with_pad(left_image, 224, 224)
        top_image = image_tools.resize_with_pad(top_image, 224, 224)
        
        # Color/Frame correction
        left_image = left_image[:, :, ::-1]
        top_image = top_image.transpose(1, 0, 2)[::-1, :, ::-1]

        if save_to_disk:
            combined_image = np.concatenate([top_image, left_image], axis=1)
            Image.fromarray(combined_image).save("robot_camera_views.png")

        # State Extraction for Policy Inference
        R_T_curr = self.left_arm.get_ee_pose()
        euler = ang.rotationMatrixToEulerAngles(R_T_curr[:3, :3])
        cartesian_6d = np.concatenate([R_T_curr[:3, 3], euler_to_rot6d(euler)])
        
        qpos = obs_dict['qpos']
        gripper_pos = np.array([qpos[6]])
        gripper_binary = binarize_gripper_actions_np(invert_gripper_actions_np(gripper_pos), threshold=0.5)

        return {
            "right_image": top_image,
            "wrist_image": left_image,
            "cartesian_position": cartesian_6d,
            "gripper_position": gripper_binary,
            "joint_position": qpos[:6],
            "state": np.concatenate([cartesian_6d, gripper_binary]),
            "euler": euler,
            "qpos": qpos
        }

if __name__ == "__main__":
    # Ensure tyro parses the Args class defined in your shared.py
    args: Args = tyro.cli(Args)
    eval_runner = AlohaEvalRunner(args)
    
    try:
        # Calls the loop in BaseEvalRunner
        eval_runner.run()
    except KeyboardInterrupt:
        print("\n[INFO] KeyboardInterrupt. Stopping robots...")
    finally:
        # Standard Interbotix cleanup
        robot_shutdown()
        print("[INFO] Shutdown complete. Safe to restart.")