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
from aloha.robot_utils import load_yaml_file
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
        # super().__init__ triggers self.init_env()
        super().__init__(args)

    def init_env(self):
        """
        Mirroring the exact environment setup and reset process from the working script.
        """
        # 1. Node Handling
        try:
            node = get_interbotix_global_node()
        except:
            node = create_interbotix_global_node('aloha')

        # 2. Load Config
        real_config = load_yaml_file(
            config_type='robot', 
            name='aloha_left_only', 
            base_path='/home/aloha/interbotix_ws/src/aloha/config'
        ).get('robot', {})

        # 3. Env Creation
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
            pass

        # 5. CRITICAL: Physical Reset
        # This starts the hardware loop and populates the first observation
        print("[INFO] Resetting environment...")
        self.env.reset()

        # 6. Arm Reference for IK
        self.left_arm = next(bot for name, bot in self.env.robots.items() if "follower" in name).arm
        
        # 7. Monkey-patch the env.step call for the BaseEvalRunner
        self.env.step = self.ik_step
        
        return self.env

    def ik_step(self, action):
        """
        Translates Cartesian policy output to Joint-space hardware commands.
        """
        t0 = time.time()
        
        # A. Calculate Target Matrix
        current_T = self.left_arm.get_ee_pose()
        R_delta = ang.eulerAnglesToRotationMatrix(action[3:6])
        target_T = np.eye(4)
        target_T[:3, :3] = current_T[:3, :3] @ R_delta
        target_T[:3, 3] = current_T[:3, 3] + action[:3]

        # B. Solve IK locally
        joint_targets, success = self.left_arm.set_ee_pose_matrix(
            target_T, execute=False, blocking=False
        )

        if success:
            # Action[6] is the binarized gripper value from BaseEvalRunner
            full_joint_action = np.append(joint_targets, action[6])
            
            # C. Step the physical hardware
            # Calling RealEnv.step directly to avoid infinite recursion with the patch
            ts = RealEnv.step(self.env, full_joint_action)
            
            # D. Maintain Frequency (50Hz)
            time.sleep(max(0, self.dt - (time.time() - t0)))
            return ts
        else:
            print("[WARN] IK Solution not found - arm staying still.")
            return self.env.get_observation()

    def _extract_observation(self, obs_dict, save_to_disk=False):
        # Image Processing (TISL Resolution Crop)
        top_image = obs_dict["images"]["camera_high"][:, 104:744]
        left_image = obs_dict["images"]["camera_wrist_left"][:, 104:744]

        # State/Proprioception Extraction
        R_T_curr = self.left_arm.get_ee_pose()
        euler = ang.rotationMatrixToEulerAngles(R_T_curr[:3, :3])
        cartesian_6d = np.concatenate([R_T_curr[:3, 3], euler_to_rot6d(euler)])
        
        qpos = obs_dict['qpos']
        # Aloha grippers are usually index 6 for a single arm
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
        # This triggers the rollout loop in shared.py
        eval_runner.run()
    finally:
        from interbotix_common_modules.common_robot.robot import robot_shutdown
        robot_shutdown()
        print("[INFO] Hardware shutdown complete.")