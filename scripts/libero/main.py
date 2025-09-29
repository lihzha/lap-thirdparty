import collections
import dataclasses
import logging
import math
import pathlib

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import matplotlib.pyplot as plt
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name]()
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "libero_spatial":
        max_steps = 220  # longest training demo has 193 steps
    elif args.task_suite_name == "libero_object":
        max_steps = 280  # longest training demo has 254 steps
    elif args.task_suite_name == "libero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "libero_10":
        max_steps = 520  # longest training demo has 505 steps
    elif args.task_suite_name == "libero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # Start evaluation
    total_episodes, total_successes = 0, 0
    for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
        # Get task
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed)

        # Start episodes
        task_episodes, task_successes = 0, 0
        for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
            logging.info(f"\nTask: {task_description}")

            # Reset environment
            env.reset()
            action_plan = collections.deque()

            # Set initial states
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            actions_xyz = []  # Store first 3 action dimensions

            logging.info(f"Starting episode {task_episodes + 1}...")
            while t < max_steps + args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        t += 1
                        continue

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][:, :])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][:, :])
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )

                    # Save preprocessed image for replay video
                    replay_images.append(img)

                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    # _quat2axisangle(obs["robot0_eef_quat"]),
                                    _quat2euler(obs["robot0_eef_quat"]),
                                    # obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        # Query model to get action
                        action_chunk = client.infer(element)["actions"]
                        assert len(action_chunk) >= args.replan_steps, (
                            f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        )
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()
                    print(action)

                    # Store first 3 action dimensions (XYZ position deltas)
                    # actions_xyz.append(action[:3].copy())
                    actions_xyz.append(obs["robot0_eef_pos"].copy())

                    # Execute action in environment
                    obs, reward, done, info = env.step(action.tolist())
                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")

            # Create combined video with action visualization
            if actions_xyz and replay_images:
                combined_frames = _create_action_visualization(actions_xyz, replay_images, fps=10)
                if combined_frames:
                    imageio.mimwrite(
                        pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}_with_actions.mp4",
                        combined_frames,
                        fps=10,
                    )

            # Also save the original replay video
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")


def _get_libero_env(task, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}

    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _create_action_visualization(actions_xyz, replay_images, fps=10):
    """
    Create a visualization showing the first 3 action dimensions (XYZ) alongside the replay images.

    Args:
        actions_xyz: List of arrays containing the first 3 action dimensions for each timestep
        replay_images: List of images from the replay
        fps: Frames per second for the output video

    Returns:
        List of combined frames (image + action plot) for video creation
    """
    if not actions_xyz or not replay_images:
        return []

    # Convert actions to numpy array
    actions_array = np.array(actions_xyz)
    timesteps = len(actions_array)

    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.tight_layout(pad=2.0)

    # Set up the action plot
    ax2.set_xlabel("Timestep")
    ax2.set_ylabel("Action Value")
    ax2.set_title("First 3 Action Dimensions (XYZ)")
    ax2.grid(alpha=0.3)

    # Set consistent y-axis limits based on all data
    all_actions = actions_array.flatten()
    y_min, y_max = np.min(all_actions), np.max(all_actions)
    y_range = y_max - y_min
    y_margin = y_range * 0.1
    ax2.set_ylim(y_min - y_margin, y_max + y_margin)
    ax2.set_xlim(0, timesteps - 1)

    # Create line objects for each dimension
    (line_x,) = ax2.plot([], [], "r-", label="X", linewidth=2)
    (line_y,) = ax2.plot([], [], "g-", label="Y", linewidth=2)
    (line_z,) = ax2.plot([], [], "b-", label="Z", linewidth=2)
    ax2.legend()

    # Remove axis labels from image subplot
    ax1.axis("off")
    ax1.set_title("Robot View")

    combined_frames = []

    for t in range(timesteps):
        # Update image
        ax1.clear()
        ax1.axis("off")
        ax1.set_title("Robot View")
        ax1.imshow(replay_images[t])

        # Update action plot
        ax2.clear()
        ax2.set_xlabel("Timestep")
        ax2.set_ylabel("Action Value")
        ax2.set_title("First 3 Action Dimensions (XYZ)")
        ax2.grid(alpha=0.3)
        ax2.set_ylim(y_min - y_margin, y_max + y_margin)
        ax2.set_xlim(0, timesteps - 1)

        # Plot action history up to current timestep
        if t > 0:
            ax2.plot(range(t + 1), actions_array[: t + 1, 0], "r-", label="X", linewidth=2)
            ax2.plot(range(t + 1), actions_array[: t + 1, 1], "g-", label="Y", linewidth=2)
            ax2.plot(range(t + 1), actions_array[: t + 1, 2], "b-", label="Z", linewidth=2)
        else:
            ax2.plot([0], [actions_array[0, 0]], "ro", label="X", markersize=8)
            ax2.plot([0], [actions_array[0, 1]], "go", label="Y", markersize=8)
            ax2.plot([0], [actions_array[0, 2]], "bo", label="Z", markersize=8)

        ax2.legend()

        # Convert plot to image
        fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        buf = buf.reshape((*fig.canvas.get_width_height()[::-1], 3))

        combined_frames.append(buf)

    plt.close(fig)
    return combined_frames


def _quat2euler(quat):
    q = np.asarray(quat, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError("quat must be shape (4,), ordered as [x, y, z, w]")

    # Normalize to guard against drift / scaling
    norm = np.linalg.norm(q)
    if norm == 0.0:
        return np.zeros(3, dtype=np.float64)
    x, y, z, w = q / norm

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    # Clamp to handle numerical issues outside [-1, 1]
    if sinp >= 1.0:
        pitch = math.pi / 2.0
    elif sinp <= -1.0:
        pitch = -math.pi / 2.0
    else:
        pitch = math.asin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw], dtype=np.float64)


# def _quat2axisangle(quat):
#     """
#     Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
#     """
#     # clip quaternion
#     if quat[3] > 1.0:
#         quat[3] = 1.0
#     elif quat[3] < -1.0:
#         quat[3] = -1.0

#     den = np.sqrt(1.0 - quat[3] * quat[3])
#     if math.isclose(den, 0.0):
#         # This is (close to) a zero degree rotation, immediately return
#         return np.zeros(3)

#     return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)
