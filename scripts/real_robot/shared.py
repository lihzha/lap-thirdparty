import dataclasses
import datetime
import faulthandler
import os
import sys
import time

import cv2
from moviepy.editor import ImageSequenceClip
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from scipy.spatial.transform import Rotation as R
import tqdm

sys.path.append(".")
from helpers import interpolate_rpy
from helpers import prevent_keyboard_interrupt

faulthandler.enable()

DROID_CONTROL_FREQUENCY = 15
IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    # "right_wrist_0_rgb",
)


@dataclasses.dataclass
class Args:
    # Hardware parameters
    left_camera_id: str = "31177322"  # e.g., "24259877"
    right_camera_id: str = "38872458"  # e.g., "24514023"
    wrist_camera_id: str = "10501775"  # e.g., "13062452"
    # Policy parameters
    external_camera: str = "right"  # which external camera should be fed to the policy, choose from ["left", "right"]
    # Rollout parameters
    max_timesteps: int = 1200
    # How many actions to execute from a predicted action chunk before querying policy server again
    # 8 is usually a good default (equals 0.5 seconds of action execution).
    open_loop_horizon: int = 5
    # Remote server parameters
    remote_host: str = "0.0.0.0"  # point this to the IP address of the policy server, e.g., "192.168.1.100"
    remote_port: int = (
        8000  # point this to the port of the policy server, default server port for openpi servers is 8000
    )
    use_wrist_camera: bool = True  # whether to use the wrist camera image as input to the policy
    run_upstream: bool = False  # whether to run the upstream policy server
    right_image_encoding: str = "raw"  # choose from ["raw", "tf_jpeg"]
    wrist_image_encoding: str = "raw"  # choose from ["raw", "tf_jpeg"]
    delta_chunk: bool = True  # whether the action chunks are delta actions or absolute positions

class BaseEvalRunner:
    CHUNK_STEPS = 6

    def __init__(self, args):
        self.env = self.init_env()
        self.args: Args = args
        self.side_image_name = None

    def init_env(self):
        from droid.robot_env import RobotEnv

        return RobotEnv(
            action_space="cartesian_position",
            gripper_action_space="position",
        )

    def binarize_gripper(self, action):
        # Binarize gripper action
        if action[-1].item() > 0.5:
            print("closing gripper")
            action = np.concatenate([action[:-1], np.ones((1,))])
        else:
            action = np.concatenate([action[:-1], np.zeros((1,))])
        return action

    def _extract_observation(self, obs_dict, save_to_disk=False):
        raise NotImplementedError()

    def _record_frames(self, curr_obs, video, wrist_video):
        side_frame = curr_obs.get(self.side_image_name)
        if side_frame is None and self.args.external_camera is not None:
            side_frame = curr_obs.get(f"{self.args.external_camera}_image")
        if side_frame is not None:
            video.append(side_frame[0] if len(side_frame.shape) == 4 else side_frame)

        wrist_frame = curr_obs.get("wrist_image")
        if wrist_frame is not None:
            wrist_video.append(wrist_frame[0].copy() if len(wrist_frame.shape) == 4 else wrist_frame.copy())

    def obs_to_request(self, curr_obs, instruction):
        request = {
            "observation": {
                IMAGE_KEYS[0]: curr_obs[self.side_image_name],
                "cartesian_position": curr_obs["cartesian_position"],
                "gripper_position": curr_obs["gripper_position"],
                "joint_position": curr_obs["joint_position"],
                "state": curr_obs["state"],
            },
            "prompt": instruction,
            "batch_size": None,
        }
        if self.args.use_wrist_camera:
            request["observation"][IMAGE_KEYS[1]] = curr_obs["wrist_image"]
        return request
    
    def process_gripper_action(self, action, curr_obs):
        return 1- curr_obs["gripper_position"] if len(action) == 6 else 1 - action[..., -1]

    def get_action_from_response(self, response, curr_obs, use_quaternions=False):
        curr_pos = np.asarray(curr_obs["cartesian_position"][:3], dtype=float)
        curr_rpy = np.asarray(curr_obs["euler"], dtype=float)
        if "reasoning" in response and response["reasoning"] is not None:
            action = np.asarray(response["actions"])
            grip_action = self.process_gripper_action(action, curr_obs)
            print(grip_action)
            # Linearly interpolate to CHUNK_STEPS actions
            positions = np.linspace(curr_pos, curr_pos + action[:3], self.CHUNK_STEPS, endpoint=True)
            rpy_arr = interpolate_rpy(curr=curr_rpy, delta=action[3:6], steps=self.CHUNK_STEPS)
            grip_vals = np.full((self.CHUNK_STEPS, 1), grip_action)
            # grip_vals[: self.CHUNK_STEPS // 2] = 1 - curr_obs["gripper_position"]
            if use_quaternions:
                # Convert RPY to quaternions for action representation
                quat_arr = R.from_euler("xyz", rpy_arr, degrees=False).as_quat()  # (x,y,z,w)
                pred_action_chunk = np.concatenate([positions, quat_arr, grip_vals], axis=1)
            else:
                pred_action_chunk = np.concatenate([positions, rpy_arr, grip_vals], axis=1)

        else:
            pred_action_chunk = response["actions"].copy()
            if pred_action_chunk.shape[-1] > 7:
                return pred_action_chunk  # joint position or velocity
            if self.args.delta_chunk:
                pred_action_chunk[:, :3] += curr_pos
                rpy_arr = add_euler(curr=curr_rpy, delta=pred_action_chunk[:, 3:6])
                pred_action_chunk[:, 3:6] = rpy_arr
            
            grip_action = self.process_gripper_action(pred_action_chunk, curr_obs)
            pred_action_chunk[:, -1] = grip_action
            print(pred_action_chunk)
            if use_quaternions:
                quat_arr = R.from_euler("xyz", pred_action_chunk[:, 3:6], degrees=False).as_quat()  # (x,y,z,w)
                pred_action_chunk = np.concatenate(
                    [pred_action_chunk[:, :3], quat_arr, pred_action_chunk[:, 6:7]], axis=1
                )
        return pred_action_chunk

    def _concat_and_save_video(self, instruction, video, wrist_video):
        assert video is not None, "Side video stream should always be recorded."
        if not video:
            return
        video = np.stack(video)
        combined_video = video
        if wrist_video:
            wrist_video = np.stack(wrist_video)

            # Ensure both videos have the same width by resizing wrist view to match side view width
            _, side_width = video.shape[1:3]
            wrist_height, wrist_width = wrist_video.shape[1:3]

            if wrist_width != side_width:
                resized_wrist = []
                aspect_ratio = wrist_width / wrist_height
                new_height = int(side_width / aspect_ratio)
                for frame in wrist_video:
                    resized_frame = cv2.resize(frame, (side_width, new_height))
                    resized_wrist.append(resized_frame)
                wrist_video = np.stack(resized_wrist)

            # Concatenate side view and wrist view vertically (wrist below side view)
            combined_video = np.concatenate([video, wrist_video], axis=1)

        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
        save_filename = "video_" + instruction.replace(" ", "_") + "_" + timestamp
        ImageSequenceClip(list(combined_video), fps=10).write_videofile(save_filename + ".mp4", fps=10, codec="libx264")

    def _rollout_once(self, instruction, fetch_action_chunk, refresh_horizon, print_action=False):
        actions_from_chunk_completed = 0
        pred_action_chunk = None
        chunk_length = refresh_horizon
        video = []
        wrist_video = []
        for t in range(self.args.max_timesteps):
            start_time = time.time()
            try:
                curr_obs = self._extract_observation(
                    self.env.get_observation(),
                    save_to_disk=t == 0
                )
                self._record_frames(curr_obs, video, wrist_video)
                if pred_action_chunk is None or actions_from_chunk_completed >= chunk_length:
                    actions_from_chunk_completed = 0
                    with prevent_keyboard_interrupt():
                        pred_action_chunk = fetch_action_chunk(curr_obs)
                        chunk_length = min(refresh_horizon, len(pred_action_chunk))
                action = pred_action_chunk[actions_from_chunk_completed]
                if not self.args.delta_chunk and action.shape[-1] == 7:
                    action[:3] += curr_obs["cartesian_position"][:3]
                    rpy_arr = add_euler(curr=curr_obs["cartesian_position"][3:6], delta=action[3:6])
                    action[3:6] = rpy_arr
                action = self.binarize_gripper(action)
                actions_from_chunk_completed += 1
                if print_action:
                    print(action)
                self.env.step(action)
                elapsed_time = time.time() - start_time
                if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                    time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)
            except KeyboardInterrupt:
                break
        save_video = input("Save video? (enter y or n) ")
        if "y" in save_video.lower():
            self._concat_and_save_video(instruction, video, wrist_video)

    def _reset_until_confirmed(self):
        while True:
            self.env.reset()
            answer = input("Correctly reset (enter y or n)? ")
            if "n" not in answer.lower():
                break

    def _run_sessions(self, make_fetcher, refresh_horizon, print_action=False):
        while True:
            instruction = input("Enter instruction: ")
            print("Running rollout... press Ctrl+C to stop early.")
            self._rollout_once(instruction, make_fetcher(instruction), refresh_horizon, print_action=print_action)
            answer = input("Do one more eval? (enter y or n) ")
            if "n" in answer.lower():
                if hasattr(self.env, "close"):
                    self.env.close()
                    os._exit(0)
                break
            self._reset_until_confirmed()

    def run(self):
        # Connect to the policy server
        policy_client = websocket_client_policy.WebsocketClientPolicy(self.args.remote_host, self.args.remote_port)

        def make_fetcher(instruction):
            return lambda curr_obs: self.get_action_from_response(
                policy_client.infer(self.obs_to_request(curr_obs, instruction)), curr_obs
            )

        self._run_sessions(make_fetcher, refresh_horizon=self.CHUNK_STEPS)

    def run_upstream(self):
        # Connect to the policy server
        policy_client = websocket_client_policy.WebsocketClientPolicy(self.args.remote_host, self.args.remote_port)

        def make_fetcher(instruction):
            return lambda curr_obs: policy_client.infer(self.obs_to_request(curr_obs, instruction))["actions"]

        self._run_sessions(make_fetcher, refresh_horizon=self.args.open_loop_horizon, print_action=True)
    
def add_euler(curr: np.ndarray, delta: np.ndarray, seq: str = "xyz") -> np.ndarray:
    """
    Add Euler-angle delta to a current Euler rotation.
    Args:
        curr:  (3,) array for current Euler angles [roll, pitch, yaw]
        delta: (3,) array for Euler delta
        seq:   rotation sequence (default: "xyz", extrinsic)
    Returns:
        new_euler: (3,) array representing updated Euler angles
    """
    r_curr = R.from_euler(seq, curr)
    r_delta = R.from_euler(seq, delta)
    # Compose rotations: new = curr * delta
    r_new = r_curr * r_delta
    return r_new.as_euler(seq)
