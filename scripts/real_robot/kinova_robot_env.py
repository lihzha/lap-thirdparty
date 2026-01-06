import zmq
import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import time
import imageio

def quat_to_euler(quat):
    q_tare = np.array([0.7071, 0, 0, 0.7071])  # 90 deg about x-axis
    quat = np.concatenate([quat[1:], quat[:1]])  # Convert from (w, x, y, z) to (x, y, z, w)
    quat_new = R.from_quat(quat).as_quat()
    quat_new_new =  (R.from_quat(q_tare) * R.from_quat(quat_new)).as_quat()
    return R.from_quat(quat_new_new).as_euler("xyz", degrees=False)

def to_kinova(quat):
    q_tare = np.array([0.7071, 0, 0, 0.7071])
    quat = R.from_quat(quat).as_quat()
    return (R.from_quat(q_tare).inv() * R.from_quat(quat)).as_quat()
    
# def quat_to_euler(quat):
#     quat = np.concatenate([quat[1:], quat[:1]])
#     return R.from_quat(quat).as_euler("xyz", degrees=False)

def to_wxyz(quat_xyzw):
    return np.concatenate([ quat_xyzw[3:4], quat_xyzw[:3]])

def to_quat(euler):
    return R.from_euler("xyz", euler, degrees=False).as_quat()

def joint_delta_to_velocity(joint_delta):
    relative_max_joint_delta = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
    max_joint_delta = relative_max_joint_delta.max()
    if isinstance(joint_delta, list):
        joint_delta = np.array(joint_delta)
    return joint_delta / max_joint_delta

def joint_velocity_to_delta(joint_velocity):
    relative_max_joint_delta = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
    max_joint_delta = relative_max_joint_delta.max()
    if isinstance(joint_velocity, list):
        joint_velocity = np.array(joint_velocity)
    relative_max_joint_vel = joint_delta_to_velocity(relative_max_joint_delta)
    max_joint_vel_norm = (np.abs(joint_velocity) / relative_max_joint_vel).max()
    if max_joint_vel_norm > 1:
        joint_velocity = joint_velocity / max_joint_vel_norm
    joint_delta = joint_velocity * max_joint_delta
    return joint_delta

def euler_to_rot6d(euler_angles: np.ndarray) -> np.ndarray:
    rot_matrix = R.from_euler("xyz", euler_angles, degrees=False).as_matrix()
    rot6d = np.concatenate([rot_matrix[:, 0], rot_matrix[:, 1]], axis=0)
    return rot6d


class RobotEnv:
    def __init__(self):
        context = zmq.Context()
        socket = context.socket(zmq.REP)
        port = 5555
        socket.bind(f"tcp://*:{port}")
        self.env = socket
        self.base_pose = None
        self.arm_qpos = None
        
        for _ in range(30): 
            req = self.env.recv_pyobj()
            if "reset" in req:
                time.sleep(1 / 15)
                self.env.send_pyobj({})
                continue
            obs = req["obs"]
            pre_action = {"action": 
                {'base_pose': np.array([0,  0, 0]), 'arm_pos': np.array([0.35, 0.00, 0.30]), 'arm_quat': np.array([0.70666403, 0.70669471, 0.02453694, 0.02462681]), 'gripper_pos': np.array([0.00]),}
                }
            self.env.send_pyobj(pre_action)

    def reset(self):
        # self.env.send_pyobj({})
        pass

    def step(self, action):
        rep = {
            "action": {
                "base_pose": self.base_pose,
                "arm_pos": action[:3],
                "arm_quat": to_wxyz(to_kinova(to_quat(action[3:6]))),
                "gripper_pos": action[6:7],
            }
        }
        # assert (action.max() <= 1.5) and (action.min() >= -1.5)
        # rep = {
        #     "action": {
        #         "base_pose": self.base_pose,
        #         "arm_qpos": self.arm_qpos + joint_velocity_to_delta(action[:7]),
        #         "gripper_pos": action[7:8],
        #     }
        # }
        self.env.send_pyobj(rep)

    def get_observation(self):
        req = self.env.recv_pyobj()
        while "reset" in req:
            self.reset()
            time.sleep(0.5)
            req = self.env.recv_pyobj()
            
        obs_dict = req["obs"]
        base_image = cv2.imdecode(obs_dict["base_image"], cv2.IMREAD_COLOR)
        wrist_image = cv2.imdecode(obs_dict["wrist_image"], cv2.IMREAD_COLOR)
        obs_dict["gripper_pos"] = 1 - obs_dict["gripper_pos"]
        
        state = np.concatenate(
            [
                obs_dict["arm_pos"],
                euler_to_rot6d(quat_to_euler(obs_dict["arm_quat"])),
                np.array(obs_dict["gripper_pos"]),
            ]
        )
        self.base_pose = obs_dict["base_pose"]
        self.arm_qpos = obs_dict["arm_qpos"]
        
        
        return {
            "euler": quat_to_euler(obs_dict["arm_quat"]),
            "cartesian_position": state[:9],
            "gripper_position": state[9],
            "base_pose": obs_dict["base_pose"],
            "base_image": base_image,
            "wrist_image": wrist_image,
            "state": state,
            "joint_position": obs_dict["arm_qpos"],
        }