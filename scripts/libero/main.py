import collections
import dataclasses
import datetime
import enum
import json
import logging
import math
import pathlib

import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # resolution used to render training data


class PolicyType(str, enum.Enum):
    """Supported policy serving modes for LIBERO eval."""

    PI05 = "PI05"
    COT = "COT"
    FT = "FT"


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5
    policy_type: PolicyType = PolicyType.PI05

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "libero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    num_steps_wait: int = 10  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task
    control_mode: str = "OSC_POSE"  # Controller type. Options: OSC_POSE, IK_POSE, OSC_POSITION, JOINT_POSITION, etc.

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "data/libero/videos"  # Path to save videos
    results_out_path: str = "data/libero/results"  # Path to save evaluation results

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
    pathlib.Path(args.results_out_path).mkdir(parents=True, exist_ok=True)

    # Initialize results tracking
    all_results = {
        "metadata": {
            "timestamp": datetime.datetime.now().isoformat(),
            "task_suite": args.task_suite_name,
            "policy_type": args.policy_type.value,
            "control_mode": args.control_mode,
            "seed": args.seed,
            "num_trials_per_task": args.num_trials_per_task,
            "replan_steps": args.replan_steps,
        },
        "episodes": [],
        "per_task_results": [],
        "summary": {},
    }

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
        env, task_description = _get_libero_env(task, LIBERO_ENV_RESOLUTION, args.seed, args.control_mode)

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
            episode_start_time = datetime.datetime.now()

            logging.info(f"Starting episode {task_episodes+1}...")
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
                    if args.policy_type == PolicyType.PI05:
                        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    else:
                        img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                        wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
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
                        if args.policy_type == PolicyType.COT or args.policy_type == PolicyType.FT:
                            eef_pos = np.asarray(obs["robot0_eef_pos"], dtype=np.float32)
                            eef_euler = _quat2euler(obs["robot0_eef_quat"]).astype(np.float32, copy=False)
                            gripper_qpos = np.asarray(obs["robot0_gripper_qpos"], dtype=np.float32)
                            gripper_state = np.array([float(np.mean(gripper_qpos))], dtype=np.float32)
                            state = np.concatenate((eef_pos, eef_euler, gripper_state)).astype(np.float32, copy=False)
                            element = {
                                "observation/image": img,
                                "observation/wrist_image": wrist_img,
                                "observation/state": state,
                                "prompt": str(task_description),
                            }
                        elif args.policy_type == PolicyType.PI05:
                            # PI05 expects DROID-style observation keys
                            joint_pos = obs.get("robot0_joint_pos")
                            if joint_pos is None:
                                joint_pos = obs.get("joint_pos")
                            if joint_pos is None:
                                raise KeyError("Observation missing joint positions for PI05 policy.")
                            joint_pos = np.asarray(joint_pos, dtype=np.float32).reshape(-1)
                            if joint_pos.size > 7:
                                joint_pos = joint_pos[:7]
                            gripper_qpos = obs.get("robot0_gripper_qpos")
                            if gripper_qpos is None:
                                gripper_qpos = obs.get("gripper_qpos")
                            if gripper_qpos is None:
                                raise KeyError("Observation missing gripper position for PI05 policy.")
                            gripper_pos = float(np.mean(np.asarray(gripper_qpos, dtype=np.float32)))
                            element = {
                                "observation/exterior_image_1_left": img,
                                "observation/wrist_image_left": wrist_img,
                                "observation/joint_position": joint_pos,
                                "observation/gripper_position": np.array([gripper_pos], dtype=np.float32),
                                "prompt": str(task_description),
                            }

                        # Query model to get action
                        response = client.infer(element)
                        if "actions" not in response:
                            raise KeyError("Policy response missing 'actions' field")
                        action_chunk = np.asarray(response["actions"], dtype=np.float32)
                        if action_chunk.ndim == 1:
                            action_chunk = action_chunk[None, ...]
                        if action_chunk.ndim != 2:
                            raise ValueError(
                                f"Expected action chunk with shape (T, D), got {action_chunk.shape}"
                            )
                        if args.policy_type == PolicyType.COT and "reasoning" in response:
                            logging.debug("Policy reasoning: %s", response["reasoning"])
                        # if args.policy_type == PolicyType.PI05:
                            # PI05 droid outputs are typically 8-D (7 joints + gripper). Env expects 7.
                            # if action_chunk.shape[1] > 7:
                            #     action_chunk = action_chunk[:, :7]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action = action_plan.popleft()

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

            # Calculate episode duration
            episode_duration = (datetime.datetime.now() - episode_start_time).total_seconds()

            # Record episode results
            episode_result = {
                "task_id": task_id,
                "task_description": task_description,
                "episode_id": episode_idx,
                "global_episode_id": total_episodes - 1,
                "success": bool(done),
                "num_steps": t - args.num_steps_wait,
                "total_steps_with_wait": t,
                "duration_seconds": episode_duration,
            }
            all_results["episodes"].append(episode_result)

            # Save a replay video of the episode
            suffix = "success" if done else "failure"
            task_segment = task_description.replace(" ", "_")
            imageio.mimwrite(
                pathlib.Path(args.video_out_path) / f"rollout_{task_segment}_{suffix}.mp4",
                [np.asarray(x) for x in replay_images],
                fps=10,
            )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")

        # Record per-task results
        task_success_rate = float(task_successes) / float(task_episodes) if task_episodes > 0 else 0.0
        task_result = {
            "task_id": task_id,
            "task_description": task_description,
            "num_episodes": task_episodes,
            "num_successes": task_successes,
            "success_rate": task_success_rate,
        }
        all_results["per_task_results"].append(task_result)

        # Log final results
        logging.info(f"Current task success rate: {task_success_rate}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    # Calculate and save final summary
    overall_success_rate = float(total_successes) / float(total_episodes) if total_episodes > 0 else 0.0
    all_results["summary"] = {
        "total_episodes": total_episodes,
        "total_successes": total_successes,
        "overall_success_rate": overall_success_rate,
        "num_tasks": num_tasks_in_suite,
    }

    # Save results to JSON file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    results_filename = f"results_{args.task_suite_name}_{args.policy_type.value}_{timestamp}.json"
    results_path = pathlib.Path(args.results_out_path) / results_filename
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logging.info(f"Total success rate: {overall_success_rate}")
    logging.info(f"Total episodes: {total_episodes}")
    logging.info(f"Results saved to: {results_path}")


def _get_libero_env(task, resolution, seed, controller="OSC_POSE"):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "controller": controller,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def _quat2euler(quat):
    q = np.asarray(quat, dtype=np.float64)
    if q.shape != (4,):
        raise ValueError("quat must be shape (4,), ordered as [x, y, z, w]")

    # Normalize quaternion to guard against numerical drift
    norm = np.linalg.norm(q)
    if norm == 0.0:
        return np.zeros(3, dtype=np.float64)
    x, y, z, w = q / norm

    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if sinp >= 1.0:
        pitch = math.pi / 2.0
    elif sinp <= -1.0:
        pitch = -math.pi / 2.0
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw], dtype=np.float64)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    tyro.cli(eval_libero)