"""Test script to verify Tiger data conversion on a single demo."""

import pickle
import shutil
from pathlib import Path

import cv2
import numpy as np
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset

# Use a test repo name
TEST_REPO_NAME = "test/tiger_single_demo"


def main():
    # Test on a single demo
    demo_dir = Path("all_data/data_tiger/demos/20251013T175916701388")

    if not demo_dir.exists():
        raise FileNotFoundError(f"Demo directory {demo_dir} does not exist")

    print(f"Testing conversion on: {demo_dir}")

    # Clean up any existing test dataset
    output_path = HF_LEROBOT_HOME / TEST_REPO_NAME
    if output_path.exists():
        print(f"Removing existing test dataset at {output_path}")
        shutil.rmtree(output_path)

    # Load pickle data
    pkl_path = demo_dir / "data.pkl"
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)

    timestamps = data["timestamps"]
    observations = data["observations"]
    actions = data["actions"]

    print(f"Loaded pickle data:")
    print(f"  Timestamps: {len(timestamps)}")
    print(f"  Observations: {len(observations)}")
    print(f"  Actions: {len(actions)}")

    # Check first observation structure
    obs_0 = observations[0]
    print(f"\nFirst observation keys: {obs_0.keys()}")
    print(f"  arm_pos shape: {obs_0['arm_pos'].shape}")
    print(f"  arm_quat shape: {obs_0['arm_quat'].shape}")
    print(f"  gripper_pos shape: {obs_0['gripper_pos'].shape}")

    # Read video frames
    base_video_path = demo_dir / "base_image.mp4"
    wrist_video_path = demo_dir / "wrist_image.mp4"

    print(f"\nReading videos...")
    cap = cv2.VideoCapture(str(base_video_path))
    base_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, base_frame = cap.read()
    if ret:
        print(f"  Base video: {base_frame_count} frames, shape: {base_frame.shape}")
    cap.release()

    cap = cv2.VideoCapture(str(wrist_video_path))
    wrist_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ret, wrist_frame = cap.read()
    if ret:
        print(f"  Wrist video: {wrist_frame_count} frames, shape: {wrist_frame.shape}")
    cap.release()

    # Verify alignment
    print(f"\nVerifying alignment:")
    print(f"  Timestamps: {len(timestamps)}")
    print(f"  Base frames: {base_frame_count}")
    print(f"  Wrist frames: {wrist_frame_count}")
    if len(timestamps) == base_frame_count == wrist_frame_count:
        print("  ✓ All aligned!")
    else:
        print("  ✗ Misalignment detected!")
        return

    # Create LeRobot dataset
    print(f"\nCreating LeRobot dataset at {output_path}...")
    dataset = LeRobotDataset.create(
        repo_id=TEST_REPO_NAME,
        robot_type="tiger",
        fps=10,
        features={
            "image": {
                "dtype": "image",
                "shape": (360, 640, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image": {
                "dtype": "image",
                "shape": (360, 640, 3),
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
        image_writer_threads=4,
        image_writer_processes=2,
    )

    # Read all frames
    print("Reading all frames...")

    def read_video_frames(video_path):
        cap = cv2.VideoCapture(str(video_path))
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        cap.release()
        return frames

    base_frames = read_video_frames(base_video_path)
    wrist_frames = read_video_frames(wrist_video_path)

    print(f"Read {len(base_frames)} base frames and {len(wrist_frames)} wrist frames")

    # Add frames to dataset
    print("Adding frames to dataset...")
    for i in range(len(timestamps)):
        obs = observations[i]
        action = actions[i]

        # Construct state and action vectors
        state = np.concatenate(
            [obs["arm_pos"], obs["arm_quat"], obs["gripper_pos"]]
        ).astype(np.float32)

        action_vector = np.concatenate(
            [action["arm_pos"], action["arm_quat"], action["gripper_pos"]]
        ).astype(np.float32)

        dataset.add_frame(
            {
                "image": base_frames[i],
                "wrist_image": wrist_frames[i],
                "state": state,
                "actions": action_vector,
                "task": f"tiger_demo_{demo_dir.name}",
            }
        )

    # Save episode
    print("Saving episode...")
    dataset.save_episode()

    print(f"\n✓ Conversion successful!")
    print(f"Dataset saved to: {output_path}")
    print(f"Total frames: {len(timestamps)}")

    # Try loading the dataset back
    print("\nTesting dataset loading...")
    loaded_dataset = LeRobotDataset(TEST_REPO_NAME)
    print(f"Loaded dataset: {len(loaded_dataset)} frames")

    # Get first sample
    sample = loaded_dataset[0]
    print(f"\nFirst sample keys: {sample.keys()}")
    print(f"  State shape: {sample['observation.state'].shape}")
    print(f"  Action shape: {sample['action'].shape}")
    print(f"  Image shape: {sample['observation.image'].shape}")

    print("\n✓ All tests passed!")


if __name__ == "__main__":
    main()
