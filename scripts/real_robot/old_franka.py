# ruff: noqa

import contextlib
import dataclasses
import datetime
import faulthandler
import signal
import sys
import time
from pathlib import Path

import numpy as np
import tqdm
import tyro
from moviepy.editor import ImageSequenceClip
from openpi_client import image_tools
from openpi_client import websocket_client_policy
from scipy.spatial.transform import Rotation as R
from droid.robot_env import RobotEnv

# Add parent directory to path to import from src

from shared import Args, IMAGE_KEYS

faulthandler.enable()

# DROID data collection frequency -- we slow down execution to match this frequency
DROID_CONTROL_FREQUENCY = 15


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
    in_camera_frame: bool = True  # whether the predicted movements are in camera frame or robot/base frame


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
    env = RobotEnv(
        robot_type="panda",
        action_space="cartesian_position",
        gripper_action_space="position",
        do_reset=True,
        default_resolution={"web": (640, 480), "rs": (640, 480)},
        resize_resolution={"web": (0, 0), "rs": (0, 0)},
    )
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
    print("Created the droid env!")

    while True:
        env.reset()
        print()
        ans = input("Correctly reset (enter y or n)? ")
        if "n" in ans.lower():
            continue
        break

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
                curr_obs = _extract_observation(
                    args,
                    env.get_observation(),
                    # Save the first observation to disk
                    save_to_disk=t_step == 0,
                )

                # Predict a new chunk if needed
                if pred_action_chunk is None or actions_from_chunk_completed >= CHUNK_STEPS:
                    actions_from_chunk_completed = 0

                    request_data = {
                        "observation": {
                            IMAGE_KEYS[0]: image_tools.resize_with_pad(curr_obs["observation/image"], 224, 224),
                            "cartesian_position": curr_obs["observation/cartesian_position"],
                            "gripper_position": curr_obs["observation/gripper_position"],
                            "state": curr_obs["observation/state"],
                        },
                        "prompt": instruction,
                        "batch_size": None,
                    }
                    if args.in_camera_frame:
                        request_data["observation"][IMAGE_KEYS[1]] = (
                            image_tools.resize_with_pad(curr_obs["observation/wrist_image"], 224, 224),
                        )

                    # Wrap the server call in a context manager to prevent Ctrl+C from interrupting it
                    # Ctrl+C will be handled after the server call is complete
                    with prevent_keyboard_interrupt():
                        # Get response from policy server (may contain actions and/or reasoning)
                        st = time.time()
                        response = policy_client.infer_reasoning(request_data)

                        # Extract actions from response (either pre-parsed or parse from reasoning)
                        if "actions" in response and response["actions"] is not None:
                            actions = np.asarray(response["actions"])
                            print(response["reasoning"])
                            # Extract translation delta (first 3 dims) and gripper (last dim)
                            delta_base = actions[0, :3]
                            grip_actions = 1 - actions[:, -1]
                        else:
                            raise NotImplementedError

                        # Map translation delta to robot/base frame if in camera frame
                        if args.in_camera_frame:
                            R_cb = cam_to_base_extrinsics_matrix[:3, :3]
                            delta_base = R_cb @ delta_base

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

                env.step(action)

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
        while True:
            env.reset()
            answer = input("Correctly reset (enter y or n)? ")
            if "n" in answer.lower():
                continue
            break


def _extract_observation(args: Args, obs_dict, *, save_to_disk=False):
    image_observations = obs_dict["image"]
    # In addition to image observations, also capture the proprioceptive state
    robot_state = obs_dict["robot_state"]
    cartesian_position = np.array(robot_state["cartesian_position"])
    gripper_position = np.array([robot_state["gripper_position"]])

    if gripper_position > 0.2:
        gripper_position = 1.0
    else:
        gripper_position = 0.0

    return {
        "observation/image": image_observations["0"][..., ::-1][None],
        "observation/wrist_image": image_observations["1"][None],
        # "observation/wrist_image": np.rot90(image_observations["1"][..., ::-1], k=1), # rotate 90 degrees
        # "observation/wrist_image": np.rot90(image_observations["1"][..., ::-1], k=2), # rotate 180 degrees
        # "observation/wrist_image": image_observations["1"][..., ::-1][::-1],  # flip vertically, up -> dowm
        #  "observation/wrist_image": image_observations["14846828_left"][..., :3][..., ::-1],  # drop alpha channel and convert BGR to RGB
        "observation/cartesian_position": cartesian_position,
        "observation/gripper_position": np.array([gripper_position]),
        "observation/state": np.concatenate([cartesian_position, [gripper_position]]),
    }


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    print(args)
    main(args)
