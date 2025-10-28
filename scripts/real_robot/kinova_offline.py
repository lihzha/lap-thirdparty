import dataclasses
import faulthandler
import logging
from pathlib import Path
import pickle
import sys

import cv2
import numpy as np
from openpi.policies import policy as _policy
from openpi_client import image_tools
import tyro

import openpi_cot.policies.adapters.policy_config_adapter as _policy_config
from openpi_cot.training import config as _config

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

IMAGE_KEYS = (
    "base_0_rgb",
    "left_wrist_0_rgb",
    # "right_wrist_0_rgb",
)

faulthandler.enable()

# DROID data collection frequency -- we slow down execution to match this frequency
DROID_CONTROL_FREQUENCY = 15


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy_config: str
    policy_dir: str
    default_prompt: str | None = None

    # Rollout parameters
    max_timesteps: int = 600
    # How many actions to execute from a predicted action chunk before querying policy server again
    # 8 is usually a good default (equals 0.5 seconds of action execution).
    open_loop_horizon: int = 8


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    if "cot" in args.policy_config or "eval" in args.policy_config:
        return _policy_config.create_trained_policy_cot(
            _config.get_config(args.policy_config), args.policy_dir, default_prompt=args.default_prompt
        )
    return _policy_config.create_trained_policy(
        _config.get_config(args.policy_config), args.policy_dir, default_prompt=args.default_prompt
    )


def _to_degrees(a, degrees: bool):
    return np.rad2deg(a) if degrees else a


def normalize_quat_xyzw(q, eps=1e-12):
    """
    Normalize quaternion(s) in xyzw ordering.
    q: array-like (..., 4)
    """
    q = np.asarray(q, dtype=np.float64)
    norm = np.linalg.norm(q, axis=-1, keepdims=True)
    return q / np.clip(norm, eps, None)


def quat_to_euler(q, degrees: bool = False):
    """
    Convert quaternion(s) to Euler angles.
    Convention: yaw-pitch-roll (Z-Y-X Tait-Bryan, extrinsic).
    Input:
        q: array-like (..., 4) in (x, y, z, w)
        degrees: if True, returned angles are in degrees.
    Output:
        angles: np.ndarray (..., 3) -> [roll (X), pitch (Y), yaw (Z)]
    """
    q = normalize_quat_xyzw(q)
    x, y, z, w = np.moveaxis(q, -1, 0)  # each (...,)

    # roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    # clamp to handle numerical edge cases
    sinp_clamped = np.clip(sinp, -1.0, 1.0)
    pitch = np.arcsin(sinp_clamped)

    # yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    angles = np.stack([roll, pitch, yaw], axis=-1)
    return _to_degrees(angles, degrees)


def read_video_frames(video_path: Path) -> list[np.ndarray]:
    """Read all frames from a video file."""
    cap = cv2.VideoCapture(str(video_path))
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
    cap.release()
    return frames


def load_tiger_demo_data(demo_name: str, timestep: int, data_dir: str = "all_data/data_tiger/demos"):
    """
    Load Tiger demo data for a specific timestep.

    Args:
        demo_name: Name of the demo folder
        timestep: Timestep index (t)
        data_dir: Base directory containing demo folders

    Returns:
        dict containing:
            - base_image: RGB image from base camera (H, W, 3)
            - wrist_image: RGB image from wrist camera (H, W, 3)
            - cartesian_position: arm_pos (3) + arm_quat (4) in xyzw format = (7,)
            - gripper_position: gripper position scalar
    """
    demo_path = Path(data_dir) / demo_name

    # Load pickle data
    pkl_path = demo_path / "data.pkl"
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    observations = data["observations"]

    # Verify timestep is valid
    if timestep >= len(observations):
        raise ValueError(f"Timestep {timestep} out of range (max: {len(observations) - 1})")

    # Load video frames
    base_video_path = demo_path / "base_image.mp4"
    wrist_video_path = demo_path / "wrist_image.mp4"

    base_frames = read_video_frames(base_video_path)
    wrist_frames = read_video_frames(wrist_video_path)

    # Get observation at timestep t
    obs = observations[timestep]

    # Extract data
    base_image = base_frames[timestep]  # (H, W, 3) RGB
    wrist_image = wrist_frames[timestep]  # (H, W, 3) RGB

    # Cartesian position: arm_pos (3) + arm_quat (4)
    cartesian_position = np.concatenate(
        [
            obs["arm_pos"],  # (3,)
            obs["arm_quat"],  # (4,) in xyzw format
        ]
    )

    gripper_position = obs["gripper_pos"][0]  # Scalar

    return {
        "base_image": base_image,
        "wrist_image": wrist_image,
        "cartesian_position": cartesian_position,
        "gripper_position": gripper_position,
    }


def main(args: Args):
    policy = create_policy(args)

    # Load demo data
    # demo_name = input("Enter demo name: ")
    # timestep = int(input("Enter timestep: "))
    demo_name = "20251013T175003694774"
    timestep = 0

    demo_data = load_tiger_demo_data(demo_name, timestep)
    base_image = demo_data["base_image"]
    wrist_image = demo_data["wrist_image"]
    cartesian_position = demo_data["cartesian_position"]
    gripper_position = demo_data["gripper_position"]

    instruction = input("Enter instruction: ")
    curr_obs = {
        "observation/image": base_image,
        "observation/wrist_image": wrist_image,
        "observation/cartesian_position": cartesian_position,
        "observation/gripper_position": np.array(gripper_position).reshape((-1, 1)),
        "observation/state": np.concatenate([cartesian_position, np.array(gripper_position)[None]]),
    }

    request_data = {
        "image": image_tools.resize_with_pad(curr_obs["observation/image"], 224, 224),
        "wrist_image": image_tools.resize_with_pad(curr_obs["observation/wrist_image"], 224, 224),
        "state": curr_obs["observation/state"],
        "prompt": instruction,
    }
    response = policy.infer(request_data)

    # Extract actions from response (either pre-parsed or parse from reasoning)
    if "actions" in response and response["actions"] is not None:
        pred_action_chunk = np.asarray(response["actions"])
    else:
        raise NotImplementedError
    print(pred_action_chunk)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)

    args: Args = tyro.cli(Args)
    print(args)
    main(args)
