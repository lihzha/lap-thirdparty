import dataclasses
import datetime
import faulthandler
import os
import sys
import time

import cv2
import h5py
from moviepy.editor import ImageSequenceClip
import numpy as np
from openpi_client import websocket_client_policy
from scipy.spatial.transform import Rotation as R

sys.path.append(".")
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
    delta_chunk: bool = True  # whether the action chunks are delta actions or absolute positions
    # Rollout saving parameters
    save_rollout: bool = False  # whether to save rollout data to h5 file
    rollout_save_dir: str = "./rollouts"  # directory to save rollout h5 files
    rollout_subsample_rate: int = 10  # subsample rate when saving rollout (1 = no subsampling)

class BaseEvalRunner:

    def __init__(self, args):
        self.env = self.init_env()
        self.args: Args = args

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

    def _extract_observation(self, obs_dict, save_to_disk=False):
        raise NotImplementedError()

    def _record_frames(self, curr_obs, video, wrist_video):
        side_frame = curr_obs.get("right_image")
        if side_frame is not None:
            video.append(side_frame[0] if len(side_frame.shape) == 4 else side_frame)

        wrist_frame = curr_obs.get("wrist_image")
        if wrist_frame is not None:
            wrist_video.append(wrist_frame[0].copy() if len(wrist_frame.shape) == 4 else wrist_frame.copy())

    def obs_to_request(self, curr_obs, instruction):
        request = {
            "observation": {
                IMAGE_KEYS[0]: curr_obs["right_image"],
                "cartesian_position": curr_obs["cartesian_position"],
                "gripper_position": curr_obs["gripper_position"],
                "joint_position": curr_obs["joint_position"],
                "state": curr_obs["state"],
                IMAGE_KEYS[1]: curr_obs["wrist_image"]
            },
            "prompt": instruction,
            "batch_size": None,
        }
        return request
    
    def process_gripper_action(self, action, curr_obs):
        return 1- curr_obs["gripper_position"] if len(action) == 6 else 1 - action[..., -1]

    def get_action_from_response(self, response, curr_obs, use_quaternions=False):
        curr_pos = np.asarray(curr_obs["cartesian_position"][:3], dtype=float)
        curr_rpy = np.asarray(curr_obs["euler"], dtype=float)

        pred_action_chunk = response["actions"].copy()
        if pred_action_chunk.shape[-1] > 7:
            return pred_action_chunk  # joint position or velocity
        if self.args.delta_chunk:
            pred_action_chunk[:, :3] += curr_pos
            rpy_arr = add_euler(curr=curr_rpy, delta=pred_action_chunk[:, 3:6])
            pred_action_chunk[:, 3:6] = rpy_arr
        
        grip_action = self.process_gripper_action(pred_action_chunk, curr_obs)
        pred_action_chunk[:, -1] = grip_action
        if use_quaternions:
            quat_arr = R.from_euler("xyz", pred_action_chunk[:, 3:6], degrees=False).as_quat()  # (x,y,z,w)
            pred_action_chunk = np.concatenate(
                [pred_action_chunk[:, :3], quat_arr, pred_action_chunk[:, 6:7]], axis=1
            )
        return pred_action_chunk

    @staticmethod
    def _slugify(text: str) -> str:
        sanitized = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in text.strip())
        return sanitized or "unnamed"

    def _concat_and_save_video(self, instruction, video, wrist_video, policy_name=None, success_label=None):
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
        filename_parts = [
            "video",
            self._slugify(policy_name) if policy_name else None,
            self._slugify(instruction),
            self._slugify(success_label) if success_label else None,
            timestamp,
        ]
        save_filename = "_".join(part for part in filename_parts if part)
        ImageSequenceClip(list(combined_video), fps=10).write_videofile(save_filename + ".mp4", fps=10, codec="libx264")

    def _save_rollout_to_h5(self, rollout_data, instruction, policy_name=None, success_label=None):
        """Save rollout data to h5 file.
        
        Args:
            rollout_data: dict with keys:
                - 'cartesian_positions': list of (6,) arrays
                - 'gripper_positions': list of floats
                - 'actions': list of (7,) arrays (cartesian position + gripper)
                - 'side_images': list of image arrays
                - 'wrist_images': list of image arrays (optional)
            instruction: str
            policy_name: str or None
            success_label: str or None
        """
        if not rollout_data['actions']:
            print("No actions recorded, skipping h5 save.")
            return
        
        # Create save directory if it doesn't exist
        os.makedirs(self.args.rollout_save_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
        filename_parts = [
            "rollout",
            self._slugify(policy_name) if policy_name else None,
            self._slugify(instruction),
            self._slugify(success_label) if success_label else None,
            timestamp,
        ]
        save_filename = "_".join(part for part in filename_parts if part) + ".h5"
        save_path = os.path.join(self.args.rollout_save_dir, save_filename)
        
        # Subsample rate
        rate = self.args.rollout_subsample_rate
        
        # Convert lists to numpy arrays with subsampling
        cartesian_positions = np.stack(rollout_data['cartesian_positions'][::rate])  # (N, 6)
        gripper_positions = np.array(rollout_data['gripper_positions'][::rate])  # (N,)
        actions = np.stack(rollout_data['actions'][::rate])  # (N, 7)
        
        with h5py.File(save_path, 'w') as f:
            # Observation group
            obs_group = f.create_group('observation')
            
            # Robot state
            robot_state = obs_group.create_group('robot_state')
            robot_state.create_dataset('cartesian_position', data=cartesian_positions)
            robot_state.create_dataset('gripper_position', data=gripper_positions)
            
            # Images (subsampled)
            image_group = obs_group.create_group('image')
            if rollout_data['side_images']:
                side_images = np.stack(rollout_data['side_images'][::rate])  # (N, H, W, C)
                image_group.create_dataset('side', data=side_images, compression='gzip')
            if rollout_data.get('wrist_images'):
                wrist_images = np.stack(rollout_data['wrist_images'][::rate])  # (N, H, W, C)
                image_group.create_dataset('wrist', data=wrist_images, compression='gzip')
            
            # Action group
            action_group = f.create_group('action')
            action_group.create_dataset('cartesian_position', data=actions)
            
            # Metadata
            f.attrs['instruction'] = instruction
            f.attrs['subsample_rate'] = rate
            if policy_name:
                f.attrs['policy_name'] = policy_name
            if success_label:
                f.attrs['success'] = success_label == 'success'
        
        print(f"Rollout saved to {save_path} (subsampled by {rate}x, {len(actions)} frames)")

    def _rollout_once(self, instruction, fetch_action_chunk, refresh_horizon, print_action=False):
        actions_from_chunk_completed = 0
        pred_action_chunk = None
        chunk_length = refresh_horizon
        video = []
        wrist_video = []
        
        # Rollout data collection for h5 saving
        rollout_data = {
            'cartesian_positions': [],
            'gripper_positions': [],
            'actions': [],
            'side_images': [],
            'wrist_images': [],
        }
        
        for t in range(self.args.max_timesteps):
            start_time = time.time()
            try:
                curr_obs = self._extract_observation(
                    self.env.get_observation(),
                    save_to_disk=t == 0
                )
                self._record_frames(curr_obs, video, wrist_video)
                
                # Collect observation data for h5 saving
                if self.args.save_rollout:
                    rollout_data['cartesian_positions'].append(
                        np.asarray(curr_obs["cartesian_position"], dtype=np.float64)
                    )
                    rollout_data['gripper_positions'].append(
                        float(curr_obs["gripper_position"])
                    )
                    # Collect images
                    side_frame = curr_obs.get("right_image")
                    if side_frame is None and self.args.external_camera is not None:
                        side_frame = curr_obs.get(f"{self.args.external_camera}_image")
                    if side_frame is not None:
                        img = side_frame[0] if len(side_frame.shape) == 4 else side_frame
                        rollout_data['side_images'].append(img.copy())
                    wrist_frame = curr_obs.get("wrist_image")
                    if wrist_frame is not None:
                        img = wrist_frame[0] if len(wrist_frame.shape) == 4 else wrist_frame
                        rollout_data['wrist_images'].append(img.copy())
                
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
                
                # Collect action data for h5 saving (action sent to env.step)
                if self.args.save_rollout:
                    rollout_data['actions'].append(np.asarray(action, dtype=np.float64))
                
                self.env.step(action)
                elapsed_time = time.time() - start_time
                if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                    time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)
            except KeyboardInterrupt:
                break
        
        policy_name = None
        success_label = None
        save_video = input("Save video? (enter y or n) ")
        if "y" in save_video.lower():
            policy_name = input("Policy name for this rollout? ").strip()
            success_answer = input("Was the rollout successful? (enter y or n) ")
            success_label = "success" if "y" in success_answer.lower() else "fail"
            self._concat_and_save_video(instruction, video, wrist_video, policy_name=policy_name, success_label=success_label)
        
        # Save rollout to h5 if enabled
        if self.args.save_rollout:
            if policy_name is None:
                policy_name = input("Policy name for h5 rollout? ").strip()
            if success_label is None:
                success_answer = input("Was the rollout successful? (enter y or n) ")
                success_label = "success" if "y" in success_answer.lower() else "fail"
            self._save_rollout_to_h5(rollout_data, instruction, policy_name=policy_name, success_label=success_label)

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

        self._run_sessions(make_fetcher, refresh_horizon=self.args.open_loop_horizon)


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
