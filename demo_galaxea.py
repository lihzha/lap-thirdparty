"""Complete demo: Load TFDS, get EEF poses, and visualize."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tensorflow_datasets as tfds
import tyro


def load_eef_poses(npz_path: str = "partial/rlds/eef_poses_all.npz"):
    """Load EEF poses from NPZ file."""
    data = np.load(npz_path, allow_pickle=True)
    return {
        "episode_ids": data["episode_ids"],
        "episode_starts": data["episode_starts"],
        "episode_lengths": data["episode_lengths"],
        "left_eef_pose": data["left_eef_pose"],
        "right_eef_pose": data["right_eef_pose"],
        "left_eef_action": data["left_eef_action"],
        "right_eef_action": data["right_eef_action"],
    }


def create_lookup(eef_data):
    """Create episode ID to index lookup."""
    return {str(eid): idx for idx, eid in enumerate(eef_data["episode_ids"])}


def describe_translation(delta_pos: np.ndarray) -> str:
    """Convert 3D translation to human-readable description."""
    dx, dy, dz = delta_pos * 100  # Convert to cm

    descriptions = []
    if abs(dx) > 0.5:
        descriptions.append(f"{'forward' if dx > 0 else 'backward'} {abs(dx):.1f}cm")
    if abs(dy) > 0.5:
        descriptions.append(f"{'left' if dy > 0 else 'right'} {abs(dy):.1f}cm")
    if abs(dz) > 0.5:
        descriptions.append(f"{'up' if dz > 0 else 'down'} {abs(dz):.1f}cm")

    return " and ".join(descriptions) if descriptions else "stay in place (< 0.5cm)"


def demo_full_pipeline(
    dataset_name: str = "sample_r1_lite",
    data_dir: str = "partial/rlds",
    eef_npz_path: str = "partial/rlds/eef_poses_all.npz",
    episode_idx: int = 0,
    frame_t: int = 10,
    horizon: int = 15,
    output_dir: str = "visualizations",
):
    """
    Complete demo: Load TFDS, query EEF poses, and visualize.

    Args:
        dataset_name: Name of TFDS dataset
        data_dir: TFDS data directory
        eef_npz_path: Path to EEF poses NPZ file
        episode_idx: Episode index to visualize
        frame_t: Starting frame
        horizon: Number of frames ahead
        output_dir: Output directory for visualization
    """
    print("=" * 80)
    print("Complete Pipeline Demo: TFDS + EEF Poses + Visualization")
    print("=" * 80)

    # Step 1: Load TFDS dataset
    print("\n[1/5] Loading TFDS dataset...")
    ds = tfds.load(dataset_name, split="train", data_dir=data_dir)
    print(f"✓ Loaded dataset: {dataset_name}")

    # Step 2: Load EEF poses
    print("\n[2/5] Loading EEF poses from NPZ...")
    eef_data = load_eef_poses(eef_npz_path)
    print(f"✓ Loaded EEF poses for {len(eef_data['episode_ids'])} episodes")

    # Create lookup dictionary
    eef_lookup = create_lookup(eef_data)
    print("✓ Created lookup dictionary")

    # Step 3: Get specific episode from TFDS
    print(f"\n[3/5] Extracting episode {episode_idx} from TFDS...")
    episode = None
    for idx, ep in enumerate(ds):
        if idx == episode_idx:
            episode = ep
            break

    if episode is None:
        print(f"✗ Episode {episode_idx} not found")
        return

    episode_id = episode["episode_metadata"]["file_path"].numpy().decode("utf-8")
    print(f"✓ Episode ID: {episode_id[:80]}...")

    # Get episode steps
    steps_list = list(episode["steps"])
    num_steps = len(steps_list)
    print(f"✓ Episode has {num_steps} steps")

    # Step 4: Query EEF poses for this episode
    print("\n[4/5] Querying EEF poses for episode...")
    if episode_id not in eef_lookup:
        print("✗ No EEF poses found for this episode")
        print(f"   Available episodes: {len(eef_lookup)}")
        return

    ep_idx = eef_lookup[episode_id]
    start = eef_data["episode_starts"][ep_idx]
    length = eef_data["episode_lengths"][ep_idx]

    # Get EEF poses for this episode
    left_poses = eef_data["left_eef_pose"][start : start + length]
    right_poses = eef_data["right_eef_pose"][start : start + length]
    left_actions = eef_data["left_eef_action"][start : start + length]
    right_actions = eef_data["right_eef_action"][start : start + length]

    print("✓ Retrieved EEF poses:")
    print(f"  Left poses: {left_poses.shape}")
    print(f"  Right poses: {right_poses.shape}")
    print(f"  Left actions: {left_actions.shape}")
    print(f"  Right actions: {right_actions.shape}")

    # Step 5: Visualize with delta information
    print("\n[5/5] Creating visualization...")

    # Check frame bounds
    frame_t_h = min(frame_t + horizon, num_steps - 1)
    actual_h = frame_t_h - frame_t

    if frame_t >= num_steps:
        print(f"✗ frame_t={frame_t} exceeds episode length {num_steps}")
        return

    # Get images
    img_t = steps_list[frame_t]["observation"]["image_camera_head"].numpy()
    img_t_h = steps_list[frame_t_h]["observation"]["image_camera_head"].numpy()

    # Calculate EEF deltas from poses
    left_delta = left_poses[frame_t_h, :3] - left_poses[frame_t, :3]
    right_delta = right_poses[frame_t_h, :3] - right_poses[frame_t, :3]

    # Get descriptions
    left_desc = describe_translation(left_delta)
    right_desc = describe_translation(right_delta)

    # Display detailed information
    print(f"\n  Visualizing frames {frame_t} → {frame_t_h} (horizon={actual_h})")
    print(f"\n  Frame {frame_t}:")
    print(f"    Left EE:  pos={left_poses[frame_t, :3]}, euler={left_poses[frame_t, 3:]}")
    print(f"    Right EE: pos={right_poses[frame_t, :3]}, euler={right_poses[frame_t, 3:]}")

    print(f"\n  Frame {frame_t_h}:")
    print(f"    Left EE:  pos={left_poses[frame_t_h, :3]}, euler={left_poses[frame_t_h, 3:]}")
    print(f"    Right EE: pos={right_poses[frame_t_h, :3]}, euler={right_poses[frame_t_h, 3:]}")

    print("\n  Movement:")
    print(f"    Left arm:  {left_desc}")
    print(f"    Right arm: {right_desc}")

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Plot frame t
    axes[0].imshow(img_t)
    axes[0].set_title(f"Frame t = {frame_t}", fontsize=16, fontweight="bold")
    axes[0].axis("off")

    # Plot frame t+h
    axes[1].imshow(img_t_h)
    axes[1].set_title(f"Frame t+h = {frame_t_h} (h={actual_h})", fontsize=16, fontweight="bold")
    axes[1].axis("off")

    # Add text descriptions
    fig.text(0.5, 0.15, "End Effector Movement:", ha="center", fontsize=14, fontweight="bold")

    fig.text(
        0.5,
        0.10,
        f"Left arm: {left_desc}",
        ha="center",
        fontsize=12,
        color="blue",
        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.5),
    )

    fig.text(
        0.5,
        0.05,
        f"Right arm: {right_desc}",
        ha="center",
        fontsize=12,
        color="red",
        bbox=dict(boxstyle="round", facecolor="lightcoral", alpha=0.5),
    )

    # Add precise values
    precise_text = (
        f"Precise deltas (meters): "
        f"Left=[{left_delta[0]:.4f}, {left_delta[1]:.4f}, {left_delta[2]:.4f}]  "
        f"Right=[{right_delta[0]:.4f}, {right_delta[1]:.4f}, {right_delta[2]:.4f}]"
    )
    fig.text(0.5, 0.01, precise_text, ha="center", fontsize=8, color="gray")

    plt.tight_layout(rect=[0, 0.18, 1, 1])

    # Save
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    output_file = output_path / f"demo_ep{episode_idx}_t{frame_t}_h{actual_h}.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    print(f"\n✓ Saved visualization to: {output_file}")

    # Show statistics
    print("\n" + "=" * 80)
    print("Pipeline Statistics:")
    print("=" * 80)
    print("  TFDS episodes loaded: 1")
    print(f"  EEF poses queried: {length} steps")
    print(f"  Visualization created: {output_file.name}")
    print("\nData format:")
    print("  Each pose: [x, y, z, roll, pitch, yaw]")
    print("  Position: meters, Rotation: radians")
    print("=" * 80)

    # Show sample data access pattern
    print("\n" + "=" * 80)
    print("Example: Access Pattern for Training")
    print("=" * 80)
    print("""
# Load EEF poses once
eef_data = np.load('eef_poses_all.npz', allow_pickle=True)
lookup = {str(eid): i for i, eid in enumerate(eef_data['episode_ids'])}

# During training loop
for episode in tfds_dataset:
    episode_id = episode['episode_metadata']['file_path'].numpy().decode('utf-8')

    # Query EEF poses
    idx = lookup[episode_id]
    start = eef_data['episode_starts'][idx]
    length = eef_data['episode_lengths'][idx]

    # Get poses
    left_poses = eef_data['left_eef_pose'][start:start+length]

    # Use with images/observations from TFDS
    for step_idx, step in enumerate(episode['steps']):
        image = step['observation']['image_camera_head']
        eef_pose = left_poses[step_idx]  # [x, y, z, roll, pitch, yaw]

        # Your training logic here
        ...
    """)
    print("=" * 80)

    plt.close()


if __name__ == "__main__":
    tyro.cli(demo_full_pipeline)
