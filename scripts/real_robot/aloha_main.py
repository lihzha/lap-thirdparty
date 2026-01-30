# ruff: noqa
import os
import sys
import time
import numpy as np
import torch
import tyro
from PIL import Image
from typing import Dict
from openpi_client import image_tools

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
    get_interbotix_global_node,
    robot_startup,
    robot_shutdown
)
from interbotix_xs_modules.xs_robot.arm import InterbotixManipulatorXS
import interbotix_common_modules.angle_manipulation as ang

sys.path.append(".")
from shared import BaseEvalRunner, Args
from helpers import binarize_gripper_actions_np, euler_to_rot6d, invert_gripper_actions_np

def opening_ceremony(robots: Dict[str, InterbotixManipulatorXS], dt: float) -> None:
    follower_bots = {name: bot for name, bot in robots.items() if "follower" in name}
    for name, follower_bot in follower_bots.items():
        follower_bot.core.robot_reboot_motors("single", "gripper", True)
        follower_bot.core.robot_set_operating_modes("group", "arm", "position")
        follower_bot.core.robot_set_operating_modes("single", "gripper", "current_based_position")
        follower_bot.core.robot_set_motor_registers("single", "gripper", "current_limit", 300)
        torque_on(follower_bot)

        move_arms(bot_list=[follower_bot], target_pose_list=[START_ARM_POSE[:6]], moving_time=4.0, dt=dt)
        move_grippers([follower_bot], [FOLLOWER_GRIPPER_JOINT_CLOSE], moving_time=0.5, dt=dt)
    print("[INFO] Opening ceremony complete.")

class AlohaEvalRunner(BaseEvalRunner):
    def __init__(self, args):
        self.dt = 1 / 15 
        super().__init__(args)

    def init_env(self):
        # 1. Clean Node Handling1 
        try:
            node = get_interbotix_global_node()
            if node is None:
                node = create_interbotix_global_node('aloha')
        except:
            node = create_interbotix_global_node('aloha')

        # 2. Load Config
        real_config = load_yaml_file(
            config_type='robot', 
            name='aloha_left_only', 
            base_path='/home/aloha/interbotix_ws/src/aloha/config'
        ).get('robot', {})

        # 3. Create Env (setup_robots=True matches the working script)
        self.env = make_real_env(
            node=node, 
            setup_robots=True, 
            setup_base=False, 
            config=real_config
        )

        # 4. Startup & Reset
        robot_startup(node)
        self.env.reset()
        
        opening_ceremony(self.env.robots, dt=self.dt)

        self.left_arm = next(bot for name, bot in self.env.robots.items() if "follower" in name).arm
        
        self.env.step = self.ik_step
        
        return self.env
    
    def ik_step(self, action):
        print(f"[DEBUG] Absolute Action Received: {action}")
        t0 = time.time()

        target_T = np.eye(4)
        target_T[:3, :3] = ang.euler_angles_to_rotation_matrix(action[3:6])
        target_T[:3, 3] = action[:3]

        current_qpos = self.env.get_qpos()
        current_joints = current_qpos[:6]

        joint_targets, success = self.left_arm.set_ee_pose_matrix(
            target_T, 
            custom_guess=current_joints, 
            execute=False
        )

        if success:
            diff = joint_targets - current_joints
            
            if np.any(np.abs(diff) > 0.5):
                print(f"[WARNING] Large joint jump detected! Diff: {diff}")
                
            clamped_diff = np.clip(diff, -0.05, 0.05) 
            safe_joints = current_joints + clamped_diff
            
            full_joint_action = np.append(safe_joints, action[6])
            ts = RealEnv.step(self.env, full_joint_action)
        else:
            print(f"[DEBUG] IK Failure at Target: {action[:3]}")
            ts = self.env.get_observation()

        time.sleep(max(0, self.dt - (time.time() - t0)))
        return ts

    def _extract_observation(self, obs_dict, save_to_disk=False):
        # TISL Resolution Crop
        top_image = obs_dict["images"]["camera_high"][:, 104:744]
        left_image = obs_dict["images"]["camera_wrist_left"][:, 104:744]

        top_image = image_tools.resize_with_pad(top_image[:, :, ::-1], 224, 224)
        left_image = image_tools.resize_with_pad(left_image[:, :, ::-1], 224, 224)

        if save_to_disk:
            combined_image = np.concatenate([top_image, left_image], axis=1)
            combined_image = Image.fromarray(combined_image)
            combined_image.save("robot_camer_views.png")

        R_T_curr = self.left_arm.get_ee_pose()
        euler = ang.rotation_matrix_to_euler_angles(R_T_curr[:3, :3])
        cartesian_6d = np.concatenate([R_T_curr[:3, 3], euler_to_rot6d(euler)])
        
        qpos = obs_dict['qpos']
        gripper_binary = binarize_gripper_actions_np(invert_gripper_actions_np(np.array([qpos[6]])), 0.5)

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
    args: Args = tyro.cli(Args)
    eval_runner = AlohaEvalRunner(args)
    try:
        eval_runner.run()
    finally:
        robot_shutdown()