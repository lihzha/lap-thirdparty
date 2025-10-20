"""
Convert Tiger demo dataset to LeRobot format.

The Tiger dataset contains demonstrations with:
- data.pkl: Contains timestamps, observations (arm_pos, arm_quat, gripper_pos), and actions
- base_image.mp4: Base camera video
- wrist_image.mp4: Wrist camera video

Usage:
    python scripts/tiger/convert_tiger_data_to_lerobot.py --data_dir all_data/data_tiger/demos

Optional:
    python scripts/tiger/convert_tiger_data_to_lerobot.py --data_dir all_data/data_tiger/demos --push_to_hub
"""

from pathlib import Path
import pickle
import shutil

import cv2
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from tqdm import tqdm
import tyro

REPO_NAME = "your_hf_username/tiger_demos"  # Name of the output dataset


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


def main(data_dir: str, *, push_to_hub: bool = False):
    data_path = Path(data_dir)
    if not data_path.exists():
        raise FileNotFoundError(f"Data directory {data_path} does not exist")

    # Get all demo directories
    demo_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])
    print(f"Found {len(demo_dirs)} demonstrations")

    # Clean up any existing dataset in the output directory
    output_path = HF_LEROBOT_HOME / REPO_NAME
    if output_path.exists():
        shutil.rmtree(output_path)

    # Create LeRobot dataset
    # State: arm_pos (3) + arm_quat (4) + gripper_pos (1) = 8 dimensions
    # Actions: arm_pos (3) + arm_quat (4) + gripper_pos (1) = 8 dimensions
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="tiger",  # Custom robot type
        fps=10,  # Tiger demos are recorded at 10 FPS
        features={
            "image": {
                "dtype": "image",
                "shape": (360, 640, 3),  # Native video resolution (H, W, C)
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (480, 640, 3),  # Native video resolution (H, W, C)
                "names": ["height", "width", "channel"],
            },
            "state": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["state"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Process each demonstration
    for demo_dir in tqdm(demo_dirs, desc="Converting demonstrations"):
        pkl_path = demo_dir / "data.pkl"
        base_video_path = demo_dir / "base_image.mp4"
        wrist_video_path = demo_dir / "wrist_image.mp4"

        # Check if all required files exist
        if not pkl_path.exists():
            print(f"Warning: {pkl_path} not found, skipping")
            continue
        if not base_video_path.exists():
            print(f"Warning: {base_video_path} not found, skipping")
            continue
        if not wrist_video_path.exists():
            print(f"Warning: {wrist_video_path} not found, skipping")
            continue

        # Load pickle data
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        timestamps = data["timestamps"]
        observations = data["observations"]
        actions = data["actions"]

        # Read video frames
        base_frames = read_video_frames(base_video_path)
        wrist_frames = read_video_frames(wrist_video_path)

        # Verify alignment
        num_timesteps = len(timestamps)
        num_base_frames = len(base_frames)
        num_wrist_frames = len(wrist_frames)

        if not (num_timesteps == num_base_frames == num_wrist_frames):
            print(
                f"Warning: Misalignment in {demo_dir.name}: "
                f"timesteps={num_timesteps}, base_frames={num_base_frames}, "
                f"wrist_frames={num_wrist_frames}, skipping"
            )
            continue

        # Add frames to dataset
        for i in range(num_timesteps):
            obs = observations[i]
            action = actions[i]

            # Construct state: arm_pos (3) + arm_quat (4) + gripper_pos (1)
            state = np.concatenate(
                [
                    obs["arm_pos"],  # (3,)
                    obs["arm_quat"],  # (4,)
                    obs["gripper_pos"],  # (1,)
                ]
            ).astype(np.float32)

            # Construct action: arm_pos (3) + arm_quat (4) + gripper_pos (1)
            action_vector = np.concatenate(
                [
                    action["arm_pos"],  # (3,)
                    action["arm_quat"],  # (4,)
                    action["gripper_pos"],  # (1,)
                ]
            ).astype(np.float32)

            # Add frame to dataset
            dataset.add_frame(
                {
                    "image": base_frames[i],
                    "wrist_image": wrist_frames[i],
                    "state": state,
                    "actions": action_vector,
                    "task": f"tiger_demo_{demo_dir.name}",  # Use demo folder name as task
                }
            )

        # Save episode
        dataset.save_episode()

    print(f"\nConversion complete! Dataset saved to {output_path}")
    print(f"Total episodes: {len(demo_dirs)}")

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["tiger", "manipulation"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )
        print(f"Dataset pushed to Hugging Face Hub: {REPO_NAME}")


if __name__ == "__main__":
    tyro.cli(main)
