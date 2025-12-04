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

    # Connect to the policy server
    policy_client = websocket_client_policy.WebsocketClientPolicy(args.remote_host, args.remote_port)

    while True:
        # instruction = input("Enter instruction: ")
        instruction = "pick the green block and place it forward by 0.5m."

        # Prepare to save video of rollout
        bar = tqdm.tqdm(range(args.max_timesteps))
        print("Running rollout... press Ctrl+C to stop early.")
        # Maintain a small open-loop action chunk predicted from the latest policy call
        actions_from_chunk_completed = 0
        pred_action_chunk = None
        CHUNK_STEPS = 6
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
                        # ()

                        # Extract actions from response (either pre-parsed or parse from reasoning)
                        if "actions" in response and response["actions"] is not None:
                            pred_action_chunk = np.asarray(response["actions"])
                        else:
                            raise NotImplementedError

                        et = time.time()
                        print(f"Time taken for inference: {et - st}")

                replay_images.append(curr_obs["observation/image"])
                # Select current action to execute from chunk
                action = pred_action_chunk[actions_from_chunk_completed]
                actions_from_chunk_completed += 1

                # # Binarize gripper action
                # if action[-1].item() > 0.5:
                #     action = np.concatenate([action[:-1], np.ones((1,))])
                # else:
                #     action = np.concatenate([action[:-1], np.zeros((1,))])

                assert len(action) == 11, f"Expected action of length 11, got {len(action)}"
                rep = {
                    "action": {
                        "base_pose": action[:3],
                        "arm_pos": action[3:6],
                        "arm_quat": action[6:10],
                        "gripper_pos": action[10:11],
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


def quat_to_r6(q: np.ndarray) -> np.ndarray:
    """
    Convert quaternion (w, x, y, z) to the first 6 elements of rotation matrix:
    [r11, r12, r13, r21, r22, r23].
    """
    assert q.shape[-1] == 4, "Input must be quaternion (w, x, y, z)"

    q = q / np.linalg.norm(q)
    quat_xyzw = np.roll(q, -1)  # scipy expects [x, y, z, w]
    # print(quat_xyzw)
    R_mat = R.from_quat(quat_xyzw).as_matrix()

    r6 = np.concatenate([R_mat[0, :], R_mat[1, :]])  # first 2 rows
    # r6 = np.random.rand(*r6.shape)
    return r6


def _extract_observation(args: Args, obs_dict, *, save_to_disk=False):
    base_image = cv2.imdecode(obs_dict["base_image"], cv2.IMREAD_COLOR)
    wrist_image = cv2.imdecode(obs_dict["wrist_image"], cv2.IMREAD_COLOR)
    # In addition to image observations, also capture the proprioceptive state
    # obs_dict["base_pose"][2] = np.random.rand(*obs_dict["base_pose"].shape)[2]
    # print(obs_dict["base_pose"] )
    state = np.concatenate(
        [
            obs_dict["base_pose"],
            obs_dict["arm_pos"],
            quat_to_r6(obs_dict["arm_quat"]),
            np.array(obs_dict["gripper_pos"]),
        ]
    )
    # if gripper_position > 0.5:
    #     gripper_position = 1.0
    # else:
    #     gripper_position = 0.0

    return {
        "observation/image": base_image,
        "observation/wrist_image": wrist_image,
        "observation/state": state,
    }


if __name__ == "__main__":
    args: Args = tyro.cli(Args)
    print(args)
    main(args)
