# ruff: noqa
import numpy as np
import torch
import os
import pickle
import time
import tyro
import sys
from typing import Dict

# Aloha / Interbotix imports
from aloha.real_env import make_real_env, RealEnv
from aloha.robot_utils import load_yaml_file, move_grippers
from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    get_interbotix_global_node,
    robot_startup,
)
from interbotix_common_modules.common_robot.exceptions import InterbotixException

# Shared / OpenPI imports
sys.path.append(".")
from shared import BaseEvalRunner, Args
from helpers import binarize_gripper_actions_np, euler_to_rot6d, invert_gripper_actions_np
import interbotix_common_modules.angle_manipulation as ang

class AlohaEvalRunner(BaseEvalRunner):
    def __init__(self, args):
        self.dt = 1 / 50 
        super().__init__(args)

    def init_env(self):
        """
        Mirroring the exact environment setup process from the working script.
        """
        # 1. Node Handling
        try:
            node = get_interbotix_global_node()
        except:
            node = create_interbotix_global_node('aloha')

        # 2. Load Config
        # Using 'aloha_left_only' but following the working script's load pattern
        real_config = load_yaml_file(
            config_type='robot', 
            name='aloha_left_only', 
            base_path='/home/aloha/interbotix_ws/src/aloha/config'
        ).get('robot', {})

        # 3. Env Creation
        # The working script sets setup_robots=True and setup_base=False
        self.env = make_real_env(
            node=node, 
            setup_robots=True, 
            setup_base=False, 
            config=real_config
        )

        # 4. Robot Startup
        try:
            robot_startup(node)
        except InterbotixException:
            # The working script passes on this exception
            pass

        # 5. Arm Reference for IK
        # In the working script, env.reset() is usually called here.
        self.left_arm = next(bot for name, bot in self.env.robots.items() if "follower" in name).arm
        
        # Monkey-patch to ensure our IK and rate limiting are used by BaseEvalRunner
        self.env.step = self.ik_step
        
        return self.env

    def ik_step(self, action):
        """
        Handles the Cartesian-to-Joint translation and frequency capping.
        """
        t0 = time.time()
        
        # A. Pose Math
        current_T = self.left_arm.get_ee_pose()
        R_delta = ang.eulerAnglesToRotationMatrix(action[3:6])
        target_T = np.eye(4)
        target_T[:3, :3] = current_T[:3, :3] @ R_delta
        target_T[:3, 3] = current_T[:3, 3] + action[:3]

        # B. Solve IK
        joint_targets, success = self.left_arm.set_ee_pose_matrix(
            target_T, execute=False, blocking=False
        )

        if success:
            # action[6] is the binarized gripper from BaseEvalRunner
            full_joint_action = np.append(joint_targets, action[6])
            
            # C. Physical Step
            # We call the base class RealEnv.step to avoid infinite recursion
            ts = RealEnv.step(self.env, full_joint_action)
            
            # D. Frequency Limit (50Hz)
            time.sleep(max(0, self.dt - (time.time() - t0)))
            return ts
        else:
            print("[WARN] IK Solution not found - holding position.")
            return self.env.get_observation()

    def _extract_observation(self, obs_dict, save_to_disk=False):
        # Image Processing (Minimal TISL Crop)
        top_image = obs_dict["images"]["camera_high"][:, 104:744]
        left_image = obs_dict["images"]["camera_wrist_left"][:, 104:744]

        # Standard State logic
        R_T_curr = self.left_arm.get_ee_pose()
        euler = ang.rotationMatrixToEulerAngles(R_T_curr[:3, :3])
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
        from interbotix_common_modules.common_robot.robot import robot_shutdown
        robot_shutdown()