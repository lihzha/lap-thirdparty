import dataclasses
import datetime
import faulthandler
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


class BaseEvalRunner:
    CHUNK_STEPS = 3

    def __init__(self, args):
        self.env = self.init_env()
        self.args: Args = args

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

    def _extract_observation(self, obs_dict):
        raise NotImplementedError()

    def obs_to_request(self, curr_obs, instruction):
        request = {
            "observation": {
                IMAGE_KEYS[0]: image_tools.resize_with_pad(curr_obs[self.side_image_name], 224, 224),
                "cartesian_position": curr_obs["cartesian_position"],
                "gripper_position": curr_obs["gripper_position"],
                "joint_position": curr_obs["joint_position"],
                "state": curr_obs["state"],
            },
            "prompt": instruction,
            "batch_size": None,
        }
        if self.args.use_wrist_camera:
            request["observation"][IMAGE_KEYS[1]] = image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224)
        return request

    def get_action_from_response(self, response, curr_obs, use_quaternions=False):
        curr_pos = np.asarray(curr_obs["cartesian_position"][:3], dtype=float)
        curr_rpy = np.asarray(curr_obs["euler"], dtype=float)
        if "reasoning" in response and response["reasoning"] is not None:
            action = np.asarray(response["actions"])
            grip_action = 1- curr_obs["gripper_position"] if len(action) == 6 else float(1 - action[-1])
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
            pred_action_chunk[:, :3] += curr_pos
            rpy_arr = interpolate_rpy(curr=curr_rpy, delta=pred_action_chunk[:, 3:6], steps=pred_action_chunk.shape[0])
            pred_action_chunk[:, 3:6] = rpy_arr
            pred_action_chunk[:, 6] = 1 - pred_action_chunk[:, 6]  # invert gripper action
            if use_quaternions:
                quat_arr = R.from_euler("xyz", pred_action_chunk[:, 3:6], degrees=False).as_quat()  # (x,y,z,w)
                pred_action_chunk = np.concatenate(
                    [pred_action_chunk[:, :3], quat_arr, pred_action_chunk[:, 6:7]], axis=1
                )
        return pred_action_chunk

    def run(self):
        # Connect to the policy server
        policy_client = websocket_client_policy.WebsocketClientPolicy(self.args.remote_host, self.args.remote_port)
        while True:
            instruction = input("Enter instruction: ")
            # Prepare to save video of rollout
            # bar = tqdm.tqdm(range(self.args.max_timesteps))
            bar = range(self.args.max_timesteps)
            print("Running rollout... press Ctrl+C to stop early.")
            # Maintain a small open-loop action chunk predicted from the latest policy call
            actions_from_chunk_completed = 0
            pred_action_chunk = None
            video = []
            wrist_video = []
            for t_step in bar:
                start_time = time.time()
                try:
                    # Get the current observation
                    curr_obs = self._extract_observation(
                        self.env.get_observation(),
                    )
                    if self.args.external_camera is not None:
                        if len(curr_obs[f"{self.args.external_camera}_image"].shape) == 4:
                            video.append(curr_obs[f"{self.args.external_camera}_image"][0])
                        else:
                            video.append(curr_obs[f"{self.args.external_camera}_image"])
                    else:
                        video.append(curr_obs["image"][0] if len(curr_obs["image"].shape) == 4 else curr_obs["image"])
                    wrist_video.append(
                        curr_obs["wrist_image"][0].copy()
                        if len(curr_obs["wrist_image"].shape) == 4
                        else curr_obs["wrist_image"].copy()
                    )
                    # Predict a new chunk if needed
                    if pred_action_chunk is None or actions_from_chunk_completed >= self.CHUNK_STEPS:
                        actions_from_chunk_completed = 0
                        # print("running inference again....*****")
                        request_data = self.obs_to_request(curr_obs, instruction)
                        # Wrap the server call in a context manager to prevent Ctrl+C from interrupting it
                        # Ctrl+C will be handled after the server call is complete
                        with prevent_keyboard_interrupt():
                            # Get response from policy server (may contain actions and/or reasoning)
                            st = time.time()
                            response = policy_client.infer(request_data)
                            pred_action_chunk = self.get_action_from_response(response, curr_obs)

                            et = time.time()
                            # print(f"Time taken for inference: {et - st}")
                    # Select current action to execute from chunk
                    action = pred_action_chunk[actions_from_chunk_completed]
                    action = self.binarize_gripper(action)
                    actions_from_chunk_completed += 1
                    self.env.step(action)
                    # Sleep to match DROID data collection frequency
                    elapsed_time = time.time() - start_time
                    if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                        time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)
                except KeyboardInterrupt:
                    break
            save_video = input("Save video? (enter y or n) ")
            if "y" in save_video.lower():
                video = np.stack(video)
                wrist_video = np.stack(wrist_video)

                # Ensure both videos have the same width by resizing wrist view to match side view width
                side_height, side_width = video.shape[1:3]
                wrist_height, wrist_width = wrist_video.shape[1:3]

                if wrist_width != side_width:
                    # Resize wrist video to match side view width while maintaining aspect ratio

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
                ImageSequenceClip(list(combined_video), fps=10).write_videofile(save_filename + ".mp4", codec="libx264")
            answer = input("Do one more eval? (enter y or n) ")
            if "n" in answer.lower():
                break
            while True:
                self.env.reset()
                answer = input("Correctly reset (enter y or n)? ")
                if "n" in answer.lower():
                    continue
                break

    def run_upstream(self):
        # Connect to the policy server
        policy_client = websocket_client_policy.WebsocketClientPolicy(self.args.remote_host, self.args.remote_port)
        while True:
            instruction = input("Enter instruction: ")
            # Prepare to save video of rollout
            # bar = tqdm.tqdm(range(self.args.max_timesteps))
            bar = range(self.args.max_timesteps)
            print("Running rollout... press Ctrl+C to stop early.")
            # Maintain a small open-loop action chunk predicted from the latest policy call
            actions_from_chunk_completed = 0
            pred_action_chunk = None
            video = []
            wrist_video = []
            for t_step in bar:
                start_time = time.time()
                try:
                    # Get the current observation
                    curr_obs = self._extract_observation(
                        self.env.get_observation(),
                    )
                    if self.args.external_camera is not None:
                        video.append(curr_obs[f"{self.args.external_camera}_image"])
                    else:
                        video.append(curr_obs["image"])
                    wrist_video.append(curr_obs["wrist_image"].copy())
                    # Predict a new chunk if needed
                    if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= self.args.open_loop_horizon:
                        actions_from_chunk_completed = 0
                        # print("running inference again....*****")
                        request_data = self.obs_to_request(curr_obs, instruction)
                        # Wrap the server call in a context manager to prevent Ctrl+C from interrupting it
                        # Ctrl+C will be handled after the server call is complete
                        with prevent_keyboard_interrupt():
                            # this returns natural language reasoning steps; convert to deltas then to absolute action
                            st = time.time()
                            pred_action_chunk = policy_client.infer(request_data)["actions"]
                            et = time.time()
                            # print(f"Time taken for inference: {et - st}")
                    # Select current action to execute from chunk
                    action = pred_action_chunk[actions_from_chunk_completed]
                    action = self.binarize_gripper(action)
                    actions_from_chunk_completed += 1
                    print(action)
                    self.env.step(action)
                    # Sleep to match DROID data collection frequency
                    elapsed_time = time.time() - start_time
                    if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                        time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)
                except KeyboardInterrupt:
                    break
            save_video = input("Save video? (enter y or n) ")
            if "y" in save_video.lower():
                video = np.stack(video)
                wrist_video = np.stack(wrist_video)

                # Ensure both videos have the same width by resizing wrist view to match side view width
                side_height, side_width = video.shape[1:3]
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
                ImageSequenceClip(list(combined_video), fps=10).write_videofile(save_filename + ".mp4", codec="libx264")
            answer = input("Do one more eval? (enter y or n) ")
            if "n" in answer.lower():
                break
            while True:
                self.env.reset()
                answer = input("Correctly reset (enter y or n)? ")
                if "n" in answer.lower():
                    continue
                break
