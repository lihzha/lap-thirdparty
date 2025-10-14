import contextlib
import datetime
import re
import signal
import time
import dataclasses
from openpi_client import image_tools

from moviepy.editor import ImageSequenceClip
import numpy as np
from openpi_client import websocket_client_policy
import tqdm

IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    # "right_wrist_0_rgb",
)

AXIS_PERM = np.array([0, 2, 1])
AXIS_SIGN = np.array([1, 1, 1])
# DROID data collection frequency -- we slow down execution to match this frequency
DROID_CONTROL_FREQUENCY = 15

@dataclasses.dataclass
class Args:
    # Hardware parameters
    left_camera_id: str = "31177322"  # e.g., "24259877"
    right_camera_id: str = "38872458"  # e.g., "24514023"
    wrist_camera_id: str = "10501775"  # e.g., "13062452"
    # Policy parameters
    external_camera: str = None  # which external camera should be fed to the policy, choose from ["left", "right"]
    # Rollout parameters
    max_timesteps: int = 600
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
    use_wrist_camera: bool = False  # whether to use the wrist camera image as input to the policy
    run_upstream: bool = False  # whether to run the upstream policy server

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


def _reasoning_to_action(
    sentences: list[str],
    start_pose: np.ndarray = None,
    in_camera_frame: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Invert `describe_movement`.
    Given a list of reasoning sentences (one per transition), reconstruct the
    sequence of gripper poses (4x4) and the per-step gripper actions.
    - Each sentence is expected to contain phrases like
      "move right 1.23 cm and move forward 4.56 cm and move down 7.89 cm and set gripper to 0.50".
    - Rotation is assumed constant (no rotation changes), matching describe_movement.
    - The first pose is `start_pose` (defaults to identity), then each sentence
      adds a translation delta to produce the next pose.
    Returns:
        (gripper_poses, gripper_actions)
        - gripper_poses: array of shape (len(sentences)+1, 4, 4)
        - gripper_actions: array of shape (len(sentences),)
    """
    if start_pose is None:
        start_pose = np.eye(4, dtype=float)
    else:
        start_pose = np.array(start_pose, dtype=float)
    if isinstance(sentences, str):
        sentences = [sentences]
    num_steps = len(sentences)
    poses = np.zeros((num_steps + 1, 4, 4), dtype=float)
    poses[0] = start_pose
    actions = np.zeros((num_steps,), dtype=float)
    # Regex patterns
    move_pat = re.compile(r"move\s+(right|left|forward|backward|up|down)\s+([\-\d\.]+)\s*cm", re.IGNORECASE)
    grip_pat = re.compile(r"set\s+gripper\s+to\s+([\-\d\.]+)", re.IGNORECASE)
    for i, sentence in enumerate(sentences):
        # Parse movements; accumulate in centimeters along (dx, dy, dz)
        dx_cm = dy_cm = dz_cm = 0.0
        print(sentence)
        for m in move_pat.finditer(sentence):
            direction = m.group(1).lower()
            value_cm = float(m.group(2))
            if direction == "right":
                dx_cm += value_cm
            elif direction == "left":
                dx_cm -= value_cm
            elif direction == "forward":
                dy_cm += value_cm
            elif direction == "backward":
                dy_cm -= value_cm
            elif direction == "down":
                dz_cm += value_cm
            elif direction == "up":
                dz_cm -= value_cm
            # if direction == "forward":
            #     dx_cm += value_cm
            # elif direction == "backward":
            #     dx_cm -= value_cm
            # elif direction == "left":
            #     dy_cm += value_cm
            # elif direction == "right":
            #     dy_cm -= value_cm
            # elif direction == "up":
            #     dz_cm += value_cm
            # elif direction == "down":
            #     dz_cm -= value_cm
        # Parse gripper action (defaults to previous if missing, else 0.0 for first)
        grip_match = grip_pat.search(sentence)
        if grip_match:
            actions[i] = float(grip_match.group(1))
        else:
            actions[i] = actions[i - 1] if i > 0 else 0.0
        # Convert from language (dx,dy,dz) in cm back to camera-frame translation (meters)
        v_cm = np.array([dx_cm, dy_cm, dz_cm], dtype=float)
        v_m = v_cm / 100.0
        # Recall: v = (AXIS_SIGN * t_cam[AXIS_PERM]) * 100
        # Therefore: t_cam[AXIS_PERM] = v_m / AXIS_SIGN
        if in_camera_frame:
            t_cam = np.zeros(3, dtype=float)
            # Avoid division-by-zero if AXIS_SIGN contains zeros (shouldn't, but safe)
            sign_safe = np.where(AXIS_SIGN == 0, 1.0, AXIS_SIGN.astype(float))
            t_mapped = v_m / sign_safe
            t_cam[AXIS_PERM] = t_mapped
        else:
            t_cam = v_m
        # Integrate translation to form next pose; keep rotation unchanged
        next_pose = poses[i].copy()
        next_pose[:3, 3] = poses[i][:3, 3] + t_cam
        poses[i + 1] = next_pose
    return poses, actions


class BaseEvalRunner:
    CHUNK_STEPS = 15

    def __init__(self, args):
        self.env = self.init_env()
        self.args = args
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
                "exterior_image_1_left": image_tools.resize_with_pad(
                    curr_obs[self.side_image_name], 224, 224
                ),
                "cartesian_position": curr_obs["cartesian_position"],
                "gripper_position": curr_obs["gripper_position"],
                "joint_position": curr_obs["joint_position"],
                "state": curr_obs["state"],
            },
            "prompt": instruction,
            "batch_size": None,
        }
        if self.args.use_wrist_camera:
            request["observation"]["wrist_image_left"] = image_tools.resize_with_pad(curr_obs["wrist_image"], 224, 224)
        return request

    def set_extrinsics(self):
        return None

    def get_action_from_reasoning(self, pred):
        poses_cam, grip_actions = _reasoning_to_action(pred, in_camera_frame=self.in_camera_frame)
        # Use the first delta translation in camera frame (meters)
        if self.in_camera_frame:
            t_cam = poses_cam[1][:3, 3] - poses_cam[0][:3, 3]
            # Map translation delta to robot/base frame using rotation only
            R_cb = self.cam_to_base_extrinsics_matrix[:3, :3]
            delta_base = R_cb @ t_cam
            breakpoint()
        else:
            delta_base = poses_cam[1][:3, 3] - poses_cam[0][:3, 3]
        return delta_base, grip_actions

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
                        video.append(curr_obs[f"{self.args.external_camera}_image"])
                    else:
                        video.append(curr_obs["image"])
                    # Predict a new chunk if needed
                    if pred_action_chunk is None or actions_from_chunk_completed >= self.args.open_loop_horizon:
                        actions_from_chunk_completed = 0
                        print("running inference again....*****")
                        request_data = self.obs_to_request(curr_obs, instruction)
                        # Wrap the server call in a context manager to prevent Ctrl+C from interrupting it
                        # Ctrl+C will be handled after the server call is complete
                        with prevent_keyboard_interrupt():
                            # this returns natural language reasoning steps; convert to deltas then to absolute action
                            st = time.time()
                            pred = policy_client.infer(request_data)["reasoning"]
                            delta_base, grip_actions = self.get_action_from_reasoning(pred)
                            # Build absolute target from current state
                            curr_pos = np.asarray(curr_obs["cartesian_position"][:3], dtype=float)
                            curr_rpy = np.asarray(curr_obs["cartesian_position"][3:6], dtype=float)
                            curr_grip = float(np.asarray(curr_obs["gripper_position"], dtype=float).reshape(-1)[0])
                            next_pos = curr_pos + delta_base
                            next_grip = float(grip_actions[0]) if grip_actions.size > 0 else curr_grip
                            # Linearly interpolate to CHUNK_STEPS actions
                            positions = np.linspace(curr_pos, next_pos, self.CHUNK_STEPS, endpoint=True)
                            rpy_arr = np.tile(curr_rpy, (self.CHUNK_STEPS, 1))
                            # grip_vals = np.linspace(curr_grip, next_grip, self.CHUNK_STEPS, endpoint=True).reshape(
                            #     -1, 1
                            # ) # no interpolation for gripper
                            grip_vals = np.full((self.CHUNK_STEPS, 1), next_grip)
                            pred_action_chunk = np.concatenate([positions, rpy_arr, grip_vals], axis=1)
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
            timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
            save_filename = "video_" + instruction.replace(" ", "_") + "_" + timestamp
            ImageSequenceClip(list(video), fps=10).write_videofile(save_filename + ".mp4", codec="libx264")
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
                        video.append(curr_obs[f"{self.args.external_camera}_image"])
                    else:
                        video.append(curr_obs["image"])
                    wrist_video.append(curr_obs["wrist_image"].copy())
                    # Predict a new chunk if needed
                    if actions_from_chunk_completed == 0  or actions_from_chunk_completed >= self.args.open_loop_horizon:
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
            timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
            save_filename = "video_" + instruction.replace(" ", "_") + "_" + timestamp
            ImageSequenceClip(list(video), fps=10).write_videofile(save_filename + ".mp4", codec="libx264")
            ImageSequenceClip(list(wrist_video), fps=10).write_videofile(save_filename + "_wrist.mp4", codec="libx264")
            answer = input("Do one more eval? (enter y or n) ")
            if "n" in answer.lower():
                break
            while True:
                self.env.reset()
                answer = input("Correctly reset (enter y or n)? ")
                if "n" in answer.lower():
                    continue
                break
