import dataclasses
import enum
import logging
import socket

from openpi.policies import policy as _policy
from openpi.policies import policy_config as upstream_policy_config
from openpi.serving import websocket_policy_server
from openpi.training import config as upstream_config
import tyro

import openpi_cot.policies.adapters.policy_config_adapter as _policy_config
from openpi_cot.training import config as _config

# ruff: noqa

import contextlib
import dataclasses
import datetime
import faulthandler
import signal
import sys
import time
from pathlib import Path
import cv2
import numpy as np
import tqdm
import tyro
from moviepy.editor import ImageSequenceClip
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from scipy.spatial.transform import Rotation as R
import zmq

# Add parent directory to path to import from src
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared import Args, IMAGE_KEYS

faulthandler.enable()

# DROID data collection frequency -- we slow down execution to match this frequency
DROID_CONTROL_FREQUENCY = 15


class EnvMode(enum.Enum):
    """Supported environments."""

    ALOHA = "aloha"
    ALOHA_SIM = "aloha_sim"
    DROID = "droid"
    LIBERO = "libero"


@dataclasses.dataclass
class Checkpoint:
    """Load a policy from a trained checkpoint."""

    # Training config name (e.g., "pi0_aloha_sim").
    config: str
    # Checkpoint directory (e.g., "checkpoints/pi0_aloha_sim/exp/10000").
    dir: str


@dataclasses.dataclass
class Default:
    """Use the default policy for the given environment."""


@dataclasses.dataclass
class Args:
    """Arguments for the serve_policy script."""

    # Environment to serve the policy for. This is only used when serving default policies.
    env: EnvMode = EnvMode.ALOHA_SIM
    # If provided, will be used in case the "prompt" key is not present in the data, or if the model doesn't have a default
    # prompt.
    default_prompt: str | None = None
    # Port to serve the policy on.
    port: int = 8000
    # Record the policy's behavior for debugging.
    record: bool = False
    # Specifies how to load the policy. If not provided, the default policy for the environment will be used.
    policy: Checkpoint | Default = dataclasses.field(default_factory=Default)


# Default checkpoints that should be used for each environment.
DEFAULT_CHECKPOINT: dict[EnvMode, Checkpoint] = {
    EnvMode.ALOHA: Checkpoint(
        config="pi05_aloha",
        dir="gs://openpi-assets/checkpoints/pi05_base",
    ),
    EnvMode.ALOHA_SIM: Checkpoint(
        config="pi0_aloha_sim",
        dir="gs://openpi-assets/checkpoints/pi0_aloha_sim",
    ),
    EnvMode.DROID: Checkpoint(
        config="pi05_droid",
        dir="gs://openpi-assets/checkpoints/pi05_droid",
    ),
    EnvMode.LIBERO: Checkpoint(
        config="pi05_libero",
        dir="gs://openpi-assets/checkpoints/pi05_libero",
    ),
}


def create_default_policy(env: EnvMode, *, default_prompt: str | None = None) -> _policy.Policy:
    """Create a default policy for the given environment."""
    if checkpoint := DEFAULT_CHECKPOINT.get(env):
        return upstream_policy_config.create_trained_policy(
            upstream_config.get_config(checkpoint.config), checkpoint.dir, default_prompt=default_prompt
        )
    raise ValueError(f"Unsupported environment mode: {env}")


def create_policy(args: Args) -> _policy.Policy:
    """Create a policy from the given arguments."""
    match args.policy:
        case Checkpoint():
            if "cot" in args.policy.config or "eval" in args.policy.config:
                return _policy_config.create_trained_policy_cot(
                    _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
                )
            return _policy_config.create_trained_policy(
                _config.get_config(args.policy.config), args.policy.dir, default_prompt=args.default_prompt
            )
        case Default():
            return create_default_policy(args.env, default_prompt=args.default_prompt)


def main(args: Args) -> None:
    policy = create_policy(args)
    policy_metadata = policy.metadata
    # Record the policy's behavior.
    if args.record:
        policy = _policy.PolicyRecorder(policy, "policy_records")
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    logging.info("Creating server (host: %s, ip: %s)", hostname, local_ip)
    server = websocket_policy_server.WebsocketPolicyServer(
        policy=policy,
        host="0.0.0.0",
        port=args.port,
        metadata=policy_metadata,
    )
    server.serve_forever()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, force=True)
    main(tyro.cli(Args))


def _to_radians(a, degrees: bool):
    return np.deg2rad(a) if degrees else a


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


def euler_to_quat(angles, degrees: bool = False):
    """
    Convert Euler angles to quaternion(s).
    Convention: yaw-pitch-roll (Z-Y-X Tait-Bryan, extrinsic).
    Input:
        angles: array-like (..., 3) -> [roll (X), pitch (Y), yaw (Z)]
        degrees: if True, 'angles' are in degrees and converted to radians.
    Output:
        q: np.ndarray (..., 4) in (x, y, z, w)
    """
    angles = np.asarray(angles, dtype=np.float64)
    roll, pitch, yaw = np.moveaxis(angles, -1, 0)  # each (...,)

    roll = _to_radians(roll, degrees)
    pitch = _to_radians(pitch, degrees)
    yaw = _to_radians(yaw, degrees)

    cr = np.cos(roll * 0.5)
    sr = np.sin(roll * 0.5)
    cp = np.cos(pitch * 0.5)
    sp = np.sin(pitch * 0.5)
    cy = np.cos(yaw * 0.5)
    sy = np.sin(yaw * 0.5)

    # ZYX (yaw, pitch, roll) → quaternion (xyzw)
    w = cr * cp * cy + sr * sp * sy
    x = sr * cp * cy - cr * sp * sy
    y = cr * sp * cy + sr * cp * sy
    z = cr * cp * sy - sr * sp * cy

    q = np.stack([x, y, z, w], axis=-1)
    return normalize_quat_xyzw(q)


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


@dataclasses.dataclass
class Args:
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


def main(args: Args):
    # Make sure external camera is specified by user -- we only use one external camera for the policy

    # Initialize the Panda environment. Using joint velocity action space and gripper position action space is very important.
    # env = RobotEnv(action_space="joint_velocity", gripper_action_space="position")

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    port = 5555
    socket.bind(f"tcp://*:{port}")
    print(f"Server started on port {port}")

    extrinsics = [
        0.19344852600560863,
        -0.4704189157280809,
        0.968999457340307,
        -2.3815317980812005,
        0.1557806728117621,
        -0.6502647332046341,
    ]  # (x,y,z),(roll,pitch,yaw)
    # Turn (pos, quat_wxyz) into 4x4 cam->base extrinsics matrix
    pos = np.array(extrinsics[:3], dtype=float)
    roll, pitch, yaw = extrinsics[3:6]
    r = R.from_euler("xyz", [roll, pitch, yaw], degrees=False)
    quat_xyzw = r.as_quat()  # (x,y,z,w)
    rot_mat = R.from_quat(quat_xyzw).as_matrix()
    cam_to_base_extrinsics_matrix = np.eye(4, dtype=float)
    cam_to_base_extrinsics_matrix[:3, :3] = rot_mat
    cam_to_base_extrinsics_matrix[:3, 3] = pos

    # while True:
    #     env.reset()
    #     print()
    #     ans = input("Correctly reset (enter y or n)? ")
    #     if "n" in ans.lower():
    #         continue
    #     break

    # Connect to the policy server
    policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)

    while True:
        instruction = input("Enter instruction: ")
        # instruction = "pick up the tomato and put it into the metal plate"

        # Prepare to save video of rollout
        bar = tqdm.tqdm(range(args.max_timesteps))
        print("Running rollout... press Ctrl+C to stop early.")
        # Maintain a small open-loop action chunk predicted from the latest policy call
        actions_from_chunk_completed = 0
        pred_action_chunk = None
        CHUNK_STEPS = 8
        replay_images = []
        for t_step in bar:
            start_time = time.time()
            try:
                # Get the current observation
                req = socket.recv_pyobj()
                if "reset" in req:
                    time.sleep(1 / DROID_CONTROL_FREQUENCY)
                    socket.send_pyobj({})
                    continue

                obs = req["obs"]
                curr_obs = _extract_observation(
                    args,
                    obs,
                    # Save the first observation to disk
                    save_to_disk=t_step == 0,
                )

                # Predict a new chunk if needed
                if pred_action_chunk is None or actions_from_chunk_completed >= CHUNK_STEPS:
                    actions_from_chunk_completed = 0

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

                    # Wrap the server call in a context manager to prevent Ctrl+C from interrupting it
                    # Ctrl+C will be handled after the server call is complete
                    with prevent_keyboard_interrupt():
                        # Get response from policy server (may contain actions and/or reasoning)
                        st = time.time()
                        response = policy_client.infer(request_data)
                        try:
                            print(response["reasoning"])
                        except:
                            pass

                        breakpoint()

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
                        curr_grip = float(
                            np.asarray(curr_obs["observation/gripper_position"], dtype=float).reshape(-1)[0]
                        )
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

                        et = time.time()
                        print(f"Time taken for inference: {et - st}")

                replay_images.append(curr_obs["observation/image"][0])
                # Select current action to execute from chunk
                action = pred_action_chunk[actions_from_chunk_completed]
                actions_from_chunk_completed += 1

                # Binarize gripper action
                if action[-1].item() > 0.5:
                    action = np.concatenate([action[:-1], np.ones((1,))])
                    # break
                    # breakpoint()
                else:
                    action = np.concatenate([action[:-1], np.zeros((1,))])

                rep = {
                    "action": {
                        "base_pose": obs["base_pose"],
                        "arm_pos": action[:3],
                        "arm_quat": action[3:7],
                        "gripper_pos": action[-1:],
                    }
                }
                socket.send_pyobj(rep)

                # Sleep to match DROID data collection frequency
                elapsed_time = time.time() - start_time
                if elapsed_time < 1 / DROID_CONTROL_FREQUENCY:
                    time.sleep(1 / DROID_CONTROL_FREQUENCY - elapsed_time)
            except KeyboardInterrupt:
                break

        video = np.stack(replay_images)
        timestamp = datetime.datetime.now().strftime("%Y_%m_%d_%H:%M:%S")
        save_filename = "video_" + instruction.replace(" ", "_") + "_" + timestamp
        ImageSequenceClip(list(video), fps=10).write_videofile(save_filename + ".mp4", codec="libx264")

        answer = input("Do one more eval? (enter y or n) ")
        if "n" in answer.lower():
            break


def _extract_observation(args: Args, obs_dict, *, save_to_disk=False):
    base_image = cv2.imdecode(obs_dict["base_image"], cv2.IMREAD_COLOR)
    wrist_image = cv2.imdecode(obs_dict["wrist_image"], cv2.IMREAD_COLOR)
    # In addition to image observations, also capture the proprioceptive state
    cartesian_position = np.concatenate([obs_dict["arm_pos"], quat_to_euler(obs_dict["arm_quat"])])
    gripper_position = np.array(obs_dict["gripper_pos"]).item()

    if gripper_position > 0.5:
        gripper_position = 1.0
    else:
        gripper_position = 0.0

    return {
        "observation/image": base_image[None],
        "observation/wrist_image": wrist_image[None],
        "observation/cartesian_position": cartesian_position,
        "observation/gripper_position": np.array(gripper_position).reshape((-1, 1)),
        "observation/state": np.concatenate([cartesian_position, np.array(gripper_position)[None]]),
    }


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    print(args)
    main(args)
