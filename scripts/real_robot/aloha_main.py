# ruff: noqa
import numpy as np
import collections
from openpi.training import config
from openpi_client import image_tools
from aloha.real_env import RealEnv
from aloha.robot_utils import move_arms, START_ARM_POSE
import interbotix_common_modules.angle_manipulation as ang
from aloha.robot_utils import move_grippers, load_yaml_file # requires aloha
from interbotix_common_modules.common_robot.robot import (
    create_interbotix_global_node,
    get_interbotix_global_node,
    robot_startup,
    InterbotixRobotNode
)
from interbotix_common_modules.common_robot.exceptions import InterbotixException
import tyro
import sys
import dm_env

sys.path.append(".")
from shared import BaseEvalRunner, Args
from PIL import Image

from helpers import binarize_gripper_actions_np, euler_to_rot6d, invert_gripper_actions_np

def make_cartesian_real_env(
    node: InterbotixRobotNode = None,
    setup_robots: bool = True,
    setup_base: bool = False,
    torque_base: bool = False,
    config: dict = None,
):
    if node is None:
        node = get_interbotix_global_node()
        if node is None:
            node = create_interbotix_global_node('aloha')
    env = RealEnvCartesian(
        node=node,
        setup_robots=setup_robots,
        setup_base=setup_base,
        torque_base=torque_base,
        config=config,
    )
    return env

class RealEnvCartesian(RealEnv):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        right_arm_init_R_T = np.array([[9.98300963e-01, 1.27577457e-03,-5.82542630e-02, 2.55826044e-01], [-1.09735611e-03, 9.99994609e-01, 3.09464090e-03, -3.15635386e-04], [5.82578970e-02, -3.02545732e-03, 9.98296982e-01, 2.88863282e-01], [0.00000000e+00,0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])
        left_arm_init_R_T = np.array([[0.87965611, 0.02124257, 0.47513565, 0.33629917], [-0.01259074, 0.99969204, -0.02138442, -0.00444487], [-0.47544359, 0.01282863, 0.87965267, 0.2516704], [0., 0., 0., 1.]])
        self.follower_bots[0].arm.set_ee_pose_matrix(right_arm_init_R_T, moving_time=10.0, accel_time=1.0, blocking=True)
        self.follower_bots[1].arm.set_ee_pose_matrix(left_arm_init_R_T, moving_time=10.0, accel_time=1.0, blocking=True)

    def set_relative_ee(self, dx=0, dy=0, dz=0, droll=0, dpitch=0, dyaw=0, custom_guess=None, execute=True, moving_time=None, accel_time=None, blocking=True):
        
        R_T_curr = self.follower_bots[0].arm.get_ee_pose()
        R_curr = R_T_curr[:3, :3]
        R_delta = ang.eulerAnglesToRotationMatrix([droll, dpitch, dyaw])

        R_target = R_curr @ R_delta
        euler_r_target = ang.rotationMatrixToEulerAngles(R_target)
        
        return self.follower_bots[0].arm.set_ee_pose_components(
            x=R_T_curr[0, 3] + dx,
            y=R_T_curr[1, 3] + dy,
            z=R_T_curr[2, 3] + dz,
            roll=euler_r_target[0],
            pitch=euler_r_target[1],
            yaw=euler_r_target[2],
            custom_guess=custom_guess,
            execute=execute,
            moving_time=moving_time,
            accel_time=accel_time,
            blocking=blocking,
        )

    def get_observation(self, get_base_vel=False):
        obs = collections.OrderedDict()
        obs['qpos'] = self.get_qpos()
        obs['qvel'] = self.get_qvel()
        obs['effort'] = self.get_effort()
        obs['images'] = self.get_images()
        if get_base_vel:
            obs['base_vel'] = self.get_base_vel()

        R_T_curr = self.follower_bots[0].arm.get_ee_pose()
        cartesian_xyz = np.array([R_T_curr[0, 3], R_T_curr[1, 3], R_T_curr[2, 3]])
        cartesian_euler = ang.rotationMatrixToEulerAngles(R_T_curr[:3, :3])

        obs['robot_state'] = {}
        obs['robot_state']['cartesian_position'] = np.concatenate([cartesian_xyz, cartesian_euler])

        # Right arm hardcoded
        obs['robot_state']['gripper_position'] = obs['qpos'][-1]
        return obs
    
    def step(self, action, get_base_vel=False, get_obs=True):

        assert len(action) == 7
        dx = action[0]
        dy = action[1]
        dz = action[2]
        droll = action[3]
        dpitch = action[4]
        dyaw = action[5]

        breakpoint()

        # _, success = self.set_relative_ee(dx=dx, dy=dy, dz=dz, droll=droll, dpitch=dpitch, dyaw=dyaw, moving_time=10.0, accel_time=0.1, blocking=True)
        if get_obs:
            obs = self.get_observation(get_base_vel)
        else:
            obs = None
        return dm_env.TimeStep(
            step_type=dm_env.StepType.MID,
            reward=self.get_reward(),
            discount=None,
            observation=obs)


class AlohaEvalRunner(BaseEvalRunner):
    def __init__(self, args):
        super().__init__(args)
        
    def init_env(self):

        try:
            node = get_interbotix_global_node()
        except:
            node = create_interbotix_global_node('aloha')

        real_config = load_yaml_file(config_type='robot', name='aloha_stationary', base_path='/home/aloha/interbotix_ws/src/aloha/config').get('robot', {})

        env = make_cartesian_real_env(node=node, setup_robots=True, setup_base=False, config=real_config)
        try:
            robot_startup(node)
        except InterbotixException:
            pass

        self.env = env

    def _extract_observation(self, obs_dict, save_to_disk=False):
        image_observations = obs_dict["images"]

        right_image = image_observations["camera_wrist_right"]
        left_image = image_observations["camera_wrist_left"]

        # Cropping to the standard Aloha resolution due to TISL camera version incompatibility.
        start_col = 104
        end_col = 744
        right_image = right_image[:, start_col:end_col]
        left_image = left_image[:, start_col:end_col]

        left_image = image_tools.resize_with_pad(left_image, 224, 224)
        right_image = image_tools.resize_with_pad(right_image, 224, 224)

        #rotate wrist image to match the base frame. For example, if looking from the robot base, 
        # the object is on the left, while in the wrist image, the object is on the right, 
        # then you should rotate the wrist image by 180 degrees, i.e. wrist_image[::-1, ::-1, ::-1]

        left_image = left_image[:, :, ::-1]
        right_image = right_image[::-1, ::-1, ::-1]
        breakpoint()
 
        if save_to_disk:
            combined_image = np.concatenate([left_image, right_image], axis=1)
            combined_image = Image.fromarray(combined_image)
            combined_image.save("robot_camera_views.png")

        # wrist_image = wrist_image[..., ::-1]
        # In addition to image observations, also capture the proprioceptive state
        robot_state = obs_dict["robot_state"]
        cartesian_position = np.array(robot_state["cartesian_position"])
        euler = cartesian_position[3:6].copy()
        cartesian_position = np.concatenate([cartesian_position[:3], euler_to_rot6d(euler)])
        gripper_position = np.array([robot_state["gripper_position"]])
        gripper_position = binarize_gripper_actions_np(invert_gripper_actions_np(gripper_position), threshold=0.5)
        return {
            "right_image": left_image,
            "wrist_image": right_image,
            "cartesian_position": cartesian_position,
            "gripper_position": gripper_position,
            "state": np.concatenate([cartesian_position, gripper_position]),
            "euler": euler,
        }

if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    print(args)
    eval_runner = AlohaEvalRunner(args)
    eval_runner.run()
  
