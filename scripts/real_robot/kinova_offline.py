import dataclasses
import faulthandler
import logging
from pathlib import Path
import sys

import numpy as np
from openpi.policies import policy as _policy
from openpi_client import image_tools
from scipy.spatial.transform import Rotation as R
import tyro

import openpi_cot.policies.adapters.policy_config_adapter as _policy_config
from openpi_cot.training import config as _config

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared import IMAGE_KEYS

faulthandler.enable()

# DROID data collection frequency -- we slow down execution to match this frequency
DROID_CONTROL_FREQUENCY = 15


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy_config: str
    policy_dir: str
    # Environment to serve the policy for. This is only used when serving default policies.
    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None
    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False

    # Rollout parameters
    max_timesteps: int = 600
    # How many actions to execute from a predicted action chunk before querying policy server again
    # 8 is usually a good default (equals 0.5 seconds of action execution).
    open_loop_horizon: int = 8


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    if "cot" in args.policy.config or "eval" in args.policy.config:
        return _policy_config.create_trained_policy_cot(
            _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
        )
    return _policy_config.create_trained_policy(
        _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
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


def main(args: Args):
    CHUNK_STEPS = 8
    policy = create_policy(args)
    base_image = imageio.load()
    wrist_image = imageio.load()
    cartesian_position = np.zeros_like(0)
    gripper_position = 0
    instruction = input("Enter instruction: ")
    curr_obs = {
        "observation/image": base_image[None],
        "observation/wrist_image": wrist_image[None],
        "observation/cartesian_position": cartesian_position,
        "observation/gripper_position": np.array(gripper_position).reshape((-1, 1)),
        "observation/state": np.concatenate([cartesian_position, np.array(gripper_position)[None]]),
    }

    request_data = {
        "observation": {
            IMAGE_KEYS[0]: image_tools.resize_with_pad(curr_obs["observation/image"], 224, 224),
            IMAGE_KEYS[1]: image_tools.resize_with_pad(curr_obs["observation/wrist_image"], 224, 224),
            "cartesian_position": curr_obs["observation/cartesian_position"],
            "gripper_position": curr_obs["observation/gripper_position"],
            "state": curr_obs["observation/state"],
        },
        "prompt": instruction,
        "batch_size": None,
    }

    response = policy.infer(request_data)
    try:
        print(response["reasoning"])
    except:
        pass

    # Extract actions from response (either pre-parsed or parse from reasoning)
    if "actions" in response and response["actions"] is not None:
        actions = np.asarray(response["actions"])
        # Extract translation delta (first 3 dims) and gripper (last dim)
        delta_base = actions[0, :3]
        grip_actions = actions[:, -1]
    else:
        raise NotImplementedError

    # Build absolute target from current state
    curr_pos = np.asarray(curr_obs["observation/cartesian_position"], dtype=float)[:3]
    curr_rpy = np.asarray(curr_obs["observation/cartesian_position"], dtype=float)[3:6]
    curr_grip = float(np.asarray(curr_obs["observation/gripper_position"], dtype=float).reshape(-1)[0])
    next_pos = curr_pos + delta_base
    next_grip = float(grip_actions[0]) if grip_actions.size > 0 else curr_grip

    # Linearly interpolate to CHUNK_STEPS actions
    positions = np.linspace(curr_pos, next_pos, CHUNK_STEPS, endpoint=True)
    curr_quat = R.from_euler("xyz", curr_rpy, degrees=False).as_quat()  # (x,y,z,w)
    # grip_vals = np.linspace(curr_grip, next_grip, CHUNK_STEPS, endpoint=True).reshape(-1, 1)
    grip_vals = np.ones((CHUNK_STEPS, 1)) * curr_grip
    grip_vals[-1] = next_grip  # ensure last gripper value is the target
    # turn rpy_arr to quat_arr
    # convert (x,y,z,w) to (w,x,y,z)
    curr_quat = np.ones((CHUNK_STEPS, 4)) * curr_quat
    pred_action_chunk = np.concatenate([positions, curr_quat, grip_vals], axis=1)
    breakpoint()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)

    args: Args = tyro.cli(Args)
    print(args)
    main(args)
