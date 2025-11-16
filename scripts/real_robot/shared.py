import contextlib
import dataclasses
import datetime
import signal
import time

from moviepy.editor import ImageSequenceClip
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import tqdm

AXIS_PERM = np.array([0, 2, 1])
AXIS_SIGN = np.array([1, 1, 1])
# DROID data collection frequency -- we slow down execution to match this frequency
DROID_CONTROL_FREQUENCY = 15


IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    # "right_wrist_0_rgb",
)


def interpolate_rpy(curr, delta, steps):
    """Interpolate roll-pitch-yaw angles using quaternion SLERP.

    This function uses spherical linear interpolation (SLERP) on quaternions
    to provide smooth rotation interpolation, avoiding gimbal lock and
    discontinuities that occur with naive linear interpolation of Euler angles.

    Args:
        curr: Current RPY angles as array of shape (3,) in radians
        delta: Change in RPY angles as array of shape (3,) or (n, 3) in radians
        steps: Number of interpolation steps

    Returns:
        Array of shape (steps, 3) with interpolated RPY values in radians
    """
    curr = np.asarray(curr, dtype=float)
    delta = np.asarray(delta, dtype=float)

    # Handle both 1D and 2D delta inputs
    if delta.ndim == 1:
        # Single delta vector - interpolate from curr to curr + delta
        target_rpy = curr + delta
    else:
        # Multiple deltas - use the first one
        target_rpy = curr + delta[0] if len(delta) > 0 else curr

    # Convert current and target RPY to rotation objects
    # RPY convention: rotate around x (roll), then y (pitch), then z (yaw)
    rot_curr = R.from_euler("xyz", curr, degrees=False)
    rot_target = R.from_euler("xyz", target_rpy, degrees=False)

    # Create SLERP interpolator
    key_times = np.array([0, 1])
    key_rots = R.concatenate([rot_curr, rot_target])
    slerp = Slerp(key_times, key_rots)

    # Generate interpolation times
    interp_times = np.linspace(0, 1, steps, endpoint=True)

    # Perform SLERP interpolation
    interpolated_rots = slerp(interp_times)

    # Convert back to RPY
    rpy_arr = interpolated_rots.as_euler("xyz", degrees=False)

    return rpy_arr


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
    open_loop_horizon: int = 8
    # Remote server parameters
    remote_host: str = "0.0.0.0"  # point this to the IP address of the policy server, e.g., "192.168.1.100"
    remote_port: int = (
        8000  # point this to the port of the policy server, default server port for openpi servers is 8000
    )
    in_camera_frame: bool = (
        False  # whether the predicted actions are in camera frame (True) or robot/base frame (False)
    )
    use_wrist_camera: bool = True  # whether to use the wrist camera image as input to the policy
    run_upstream: bool = False  # whether to run the upstream policy server
    predict_rotation: bool = False  # whether to use roll-pitch-yaw for orientation representation


# We are using Ctrl+C to optionally terminate rollouts early -- however, if we press Ctrl+C while the policy server is
# waiting for a new action chunk, it will raise an exception and the server connection dies.
# This context manager temporarily prevents Ctrl+C and delays it after the server call is complete.
@contextlib.contextmanager
def prevent_keyboard_interrupt():
    """Temporarily prevent keyboard interrupts by delaying them until after the protected code."""
    interrupted = False
    original_handler = signal.getsignal(signal.SIGINT)

    def handler(signum, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, handler)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, original_handler)
        if interrupted:
            raise KeyboardInterrupt


class BaseEvalRunner:
    CHUNK_STEPS = 15

    def __init__(self, args):
        self.env = self.init_env()
        self.args: Args = args
        self.cam_to_base_extrinsics_matrix = self.set_extrinsics()
        self.in_camera_frame = args.in_camera_frame
        assert self.in_camera_frame == (self.cam_to_base_extrinsics_matrix is not None), (
            "Must have extrinsics if using camera frame"
        )

    def init_env(self):
        raise NotImplementedError()

    def binarize_gripper(self, action):
        # Binarize gripper action
        if action[-1].item() > 0.5:
            print("closing gripper")
            action = np.concatenate([action[:-1], np.ones((1,))])
        else:
            action = np.concatenate([action[:-1], np.zeros((1,))])
        return action

    def _extract_observation(self, obs_dict, *, save_to_disk=False):
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

    def set_extrinsics(self):
        return None

    def get_action_from_response(self, response, curr_obs, use_quaternions=False, use_velocity=False):
        """Extract actions from server response, either directly or by parsing reasoning.

        Args:
            response: Server response dict containing 'actions' and/or 'reasoning'

        Returns:
            (delta_base, grip_actions) - translation delta and gripper actions
        """
        # If server already parsed actions, use them directly
        assert "actions" in response and response["actions"] is not None
        if "reasoning" in response and response["reasoning"] is not None:
            print(response["reasoning"])
            actions = np.asarray(response["actions"])
            # Extract translation delta (first 3 dims) and gripper (last dim)
            
            if not use_velocity:
                delta_base = actions[0, :3]  # Already in meters
                grip_actions = 1 - actions[0, -1]
                # grip_actions = actions[:, -1]

                # If in camera frame, transform to robot/base frame
                if self.in_camera_frame:
                    R_cb = self.cam_to_base_extrinsics_matrix[:3, :3]
                    delta_base = R_cb @ delta_base

                # Build absolute target from current state
                curr_pos = np.asarray(curr_obs["cartesian_position"][:3], dtype=float)
                curr_rpy = np.asarray(curr_obs["cartesian_position"][3:6], dtype=float)
                curr_grip = float(np.asarray(curr_obs["gripper_position"], dtype=float).reshape(-1)[0])
                next_pos = curr_pos + delta_base
                next_grip = float(grip_actions)
                # Linearly interpolate to CHUNK_STEPS actions
                positions = np.linspace(curr_pos, next_pos, self.CHUNK_STEPS, endpoint=True)
                if self.args.predict_rotation:
                    rpy_arr = interpolate_rpy(curr=curr_rpy, delta=actions[0, 3:6], steps=self.CHUNK_STEPS)
                else:
                    rpy_arr = np.tile(curr_rpy, (self.CHUNK_STEPS, 1))
                # grip_vals = np.linspace(curr_grip, next_grip, self.CHUNK_STEPS, endpoint=True).reshape(
                #     -1, 1
                # ) # no interpolation for gripper
                grip_vals = np.full((self.CHUNK_STEPS, 1), next_grip)
                if use_quaternions:
                    # Convert RPY to quaternions for action representation
                    quat_arr = R.from_euler("xyz", rpy_arr, degrees=False).as_quat()  # (x,y,z,w)
                    pred_action_chunk = np.concatenate([positions, quat_arr, grip_vals], axis=1)
                else:
                    pred_action_chunk = np.concatenate([positions, rpy_arr, grip_vals], axis=1)
            else:
                delta_base = actions[0, :-1]
                grip_actions = 1 - actions[0, -1]
                next_grip = float(grip_actions)

                pos_vel = np.tile(delta_base[:3], (self.CHUNK_STEPS, 1))
                rot_vel = np.tile(delta_base[3:6], (self.CHUNK_STEPS, 1))
                grip_vals = np.full((self.CHUNK_STEPS, 1), next_grip)

                if use_quaternions:
                    # Convert RPY to quaternions for action representation
                    quat_vel = R.from_euler("xyz", rot_vel, degrees=False).as_quat()  # (x,y,z,w)
                    pred_action_chunk = np.concatenate([pos_vel, quat_vel, grip_vals], axis=1)
                else:
                    pred_action_chunk = np.concatenate([pos_vel, rot_vel, grip_vals], axis=1)
                

        else:
            curr_pos = np.asarray(curr_obs["cartesian_position"][:3], dtype=float)
            curr_rpy = np.asarray(curr_obs["cartesian_position"][3:6], dtype=float)
            pred_action_chunk = response["actions"].copy()
            print(pred_action_chunk)
            pred_action_chunk[:, :3] += curr_pos
            if self.args.predict_rotation:
                rpy_arr = interpolate_rpy(
                    curr=curr_rpy, delta=pred_action_chunk[:, 3:6], steps=pred_action_chunk.shape[0]
                )
                pred_action_chunk[:, 3:6] = rpy_arr
            else:
                pred_action_chunk[:, 3:6] = curr_rpy
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
            bar = tqdm.tqdm(range(self.args.max_timesteps))
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
                        # Save the first observation to disk
                        save_to_disk=t_step == 0,
                    )
                    if self.args.external_camera is not None:
                        if len(curr_obs[f"{self.args.external_camera}_image"].shape) == 4:
                            video.append(curr_obs[f"{self.args.external_camera}_image"][0])
                        else:
                            video.append(curr_obs[f"{self.args.external_camera}_image"])
                    else:
                        video.append(curr_obs["image"][0] if len(curr_obs["image"].shape) == 4 else curr_obs["image"])
                    wrist_video.append(curr_obs["wrist_image"][0].copy() if len(curr_obs["wrist_image"].shape) == 4 else curr_obs["wrist_image"].copy())
                    # Predict a new chunk if needed
                    if pred_action_chunk is None or actions_from_chunk_completed >= self.args.open_loop_horizon:
                        actions_from_chunk_completed = 0
                        print("running inference again....*****")
                        request_data = self.obs_to_request(curr_obs, instruction)
                        # Wrap the server call in a context manager to prevent Ctrl+C from interrupting it
                        # Ctrl+C will be handled after the server call is complete
                        with prevent_keyboard_interrupt():
                            # Get response from policy server (may contain actions and/or reasoning)
                            st = time.time()
                            response = policy_client.infer(request_data)
                            pred_action_chunk = self.get_action_from_response(response, curr_obs)

                            et = time.time()
                            print(f"Time taken for inference: {et - st}")
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
            video = np.stack(video)
            wrist_video = np.stack(wrist_video)

            # Ensure both videos have the same width by resizing wrist view to match side view width
            side_height, side_width = video.shape[1:3]
            wrist_height, wrist_width = wrist_video.shape[1:3]

            if wrist_width != side_width:
                # Resize wrist video to match side view width while maintaining aspect ratio
                import cv2

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
            bar = tqdm.tqdm(range(self.args.max_timesteps))
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
                        # Save the first observation to disk
                        save_to_disk=t_step == 0,
                    )
                    if self.args.external_camera is not None:
                        video.append(curr_obs[f"{self.args.external_camera}_image"][0])
                    else:
                        video.append(curr_obs["image"][0])
                    wrist_video.append(curr_obs["wrist_image"][0].copy())
                    # Predict a new chunk if needed
                    if actions_from_chunk_completed == 0 or actions_from_chunk_completed >= self.args.open_loop_horizon:
                        actions_from_chunk_completed = 0
                        print("running inference again....*****")
                        request_data = self.obs_to_request(curr_obs, instruction)
                        # Wrap the server call in a context manager to prevent Ctrl+C from interrupting it
                        # Ctrl+C will be handled after the server call is complete
                        with prevent_keyboard_interrupt():
                            # this returns natural language reasoning steps; convert to deltas then to absolute action
                            st = time.time()
                            pred_action_chunk = policy_client.infer(request_data)["actions"]
                            et = time.time()
                            print(f"Time taken for inference: {et - st}")
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
            video = np.stack(video)
            wrist_video = np.stack(wrist_video)

            # Ensure both videos have the same width by resizing wrist view to match side view width
            side_height, side_width = video.shape[1:3]
            wrist_height, wrist_width = wrist_video.shape[1:3]

            if wrist_width != side_width:
                # Resize wrist video to match side view width while maintaining aspect ratio
                import cv2

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
