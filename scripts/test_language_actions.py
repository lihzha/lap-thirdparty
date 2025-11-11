#!/usr/bin/env python3
"""
Test language action generation in EEF frame vs base frame and visualize on video.

This script:
1. Loads trajectory data from trajectory.h5
2. Computes delta actions between consecutive frames
3. Generates language actions in both EEF frame and base frame
4. Visualizes the language actions on the video
"""

from pathlib import Path
import re
import sys

import cv2
import h5py
import numpy as np
from scipy.spatial.transform import Rotation as R

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# Import utilities directly to avoid module dependencies
def _format_numeric(val: float, sum_decimal: str) -> str:
    """Match droid policy formatting for numbers"""
    decimals = 0
    if isinstance(sum_decimal, str):
        if sum_decimal == "no_number":
            return ""
        m = re.fullmatch(r"(\d+)f", sum_decimal)
        if m:
            decimals = int(m.group(1))
    return f"{val:.{decimals}f}"


def summarize_numeric_actions(arr_like, sum_decimal: str, include_rotation: bool = False) -> str | None:
    """Convert numeric delta EE actions ([..., 7]) into a language string."""
    arr = np.asarray(arr_like, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[-1] < 7:
        return None

    # Handle compact format
    if sum_decimal == "compact":
        # Sum translations over the window and convert to cm
        dx_cm = int(round(float(arr[..., 0].sum()) * 100.0))
        dy_cm = int(round(float(arr[..., 1].sum()) * 100.0))
        dz_cm = int(round(float(arr[..., 2].sum()) * 100.0))

        parts = [f"{dx_cm:+03d}", f"{dy_cm:+03d}", f"{dz_cm:+03d}"]

        # Gripper: threshold at 0.5 to convert to binary
        g_last = float(arr[-1, 6])
        grip_binary = 1 if g_last >= 0.5 else 0
        parts.append(str(grip_binary))

        return "<" + " ".join(parts) + ">"

    # Convert to centimeters
    if sum_decimal == "no_number":
        decimals = 0
    else:
        decimals = int(re.fullmatch(r"(\d+)f", sum_decimal).group(1))

    # Sum translations over the window
    dx_m = float(arr[..., 0].sum())
    dy_m = float(arr[..., 1].sum())
    dz_m = float(arr[..., 2].sum())
    dx = round(abs(dx_m * 100.0), decimals)
    dy = round(abs(dy_m * 100.0), decimals)
    dz = round(abs(dz_m * 100.0), decimals)

    parts: list[str] = []

    if sum_decimal == "no_number":
        if dx_m > 0:
            parts.append("move forward")
        elif dx_m < 0:
            parts.append("move back")
        if dy_m > 0:
            parts.append("move left")
        elif dy_m < 0:
            parts.append("move right")
        if dz_m > 0:
            parts.append("move up")
        elif dz_m < 0:
            parts.append("move down")
    else:
        fmt_dx = _format_numeric(dx, sum_decimal)
        fmt_dy = _format_numeric(dy, sum_decimal)
        fmt_dz = _format_numeric(dz, sum_decimal)
        if dx_m > 0 and dx != 0:
            parts.append(f"move forward {fmt_dx} cm")
        elif dx_m < 0 and dx != 0:
            parts.append(f"move back {fmt_dx} cm")
        if dz_m > 0 and dz != 0:
            parts.append(f"move down {fmt_dz} cm")
        elif dz_m < 0 and dz != 0:
            parts.append(f"move up {fmt_dz} cm")
        if dy_m > 0 and dy != 0:
            parts.append(f"move right {fmt_dy} cm")
        elif dy_m < 0 and dy != 0:
            parts.append(f"move left {fmt_dy} cm")

    # Final gripper value from last step
    g_last = float(arr[-1, 6])
    if g_last > 0.5:
        parts.append("open gripper")
    else:
        parts.append("close gripper")
    return ", ".join(parts)


def transform_actions_to_eef_frame(actions: np.ndarray, initial_state: np.ndarray) -> np.ndarray:
    """Transform actions from base frame to end effector frame."""
    actions = np.asarray(actions, dtype=float)
    initial_state = np.asarray(initial_state, dtype=float)

    assert actions.ndim == 1
    transformed_actions = actions.copy()

    assert len(initial_state.shape) == 1 and initial_state[7] == 0, "Only supporting euler angle now"
    euler = initial_state[3:6]
    initial_rotation = R.from_euler("xyz", euler)

    # Get rotation matrix: base -> EEF
    R_base_to_eef = initial_rotation.as_matrix().T  # Transpose to get EEF <- base

    delta_pos_base = actions[:3]
    delta_pos_eef = R_base_to_eef @ delta_pos_base
    transformed_actions[:3] = delta_pos_eef

    return transformed_actions


def load_trajectory(trajectory_path: str):
    """Load trajectory data from h5 file."""
    with h5py.File(trajectory_path, "r") as f:
        # Extract cartesian position (x, y, z, roll, pitch, yaw in extrinsic euler XYZ)
        cartesian_position = f["observation"]["robot_state"]["cartesian_position"][:]

        # Also check if there's gripper data
        if "gripper_position" in f["observation"]["robot_state"]:
            gripper_position = f["observation"]["robot_state"]["gripper_position"][:]
        else:
            # Use default closed gripper if not available
            gripper_position = np.zeros((len(cartesian_position), 1))

    return cartesian_position, gripper_position


def compute_delta_actions(cartesian_position, gripper_position):
    """
    Compute delta actions between consecutive frames.

    Args:
        cartesian_position: (T, 6) array with [x, y, z, roll, pitch, yaw]
        gripper_position: (T,) or (T, 1) array with gripper state

    Returns:
        delta_actions: (T-1, 7) array with [dx, dy, dz, droll, dpitch, dyaw, gripper]
    """
    T = len(cartesian_position)
    delta_actions = np.zeros((T - 1, 7))

    # Ensure gripper_position is 1D
    if gripper_position.ndim == 2:
        gripper_position = gripper_position.squeeze()

    for i in range(T - 1):
        # Translation deltas (in meters)
        delta_actions[i, :3] = cartesian_position[i + 1, :3] - cartesian_position[i, :3]

        # Rotation deltas (in radians)
        # For extrinsic euler angles, we can compute the delta directly
        delta_actions[i, 3:6] = cartesian_position[i + 1, 3:6] - cartesian_position[i, 3:6]

        # Gripper state (use next state)
        delta_actions[i, 6] = gripper_position[i + 1]

    return delta_actions


def generate_language_actions(delta_actions, cartesian_trajectory, format_type="verbose", decimal_places=1, subsample_rate=1):
    """
    Generate language actions in both base frame and EEF frame.

    Args:
        delta_actions: (T, 7) array of delta actions in base frame
        cartesian_trajectory: (T+1, 6) array with EEF poses [x, y, z, roll, pitch, yaw] for each timestep
        format_type: "verbose" or "compact"
        decimal_places: Number of decimal places for verbose format
        subsample_rate: Number of frames to aggregate together (e.g., 10 means sum every 10 frames)

    Returns:
        base_frame_actions: list of language action strings in base frame
        eef_frame_actions: list of language action strings in EEF frame
        aggregated_actions: (T//subsample_rate, 7) array of aggregated delta actions
    """
    sum_decimal = "compact" if format_type == "compact" else f"{decimal_places}f"

    base_frame_actions = []
    eef_frame_actions = []

    # Aggregate actions by subsampling
    T = len(delta_actions)
    num_windows = T // subsample_rate
    aggregated_actions = []

    for window_idx in range(num_windows):
        start_idx = window_idx * subsample_rate
        end_idx = min(start_idx + subsample_rate, T)

        # Sum translations and rotations over the window
        window_actions = delta_actions[start_idx:end_idx]
        aggregated_action = np.zeros(7)
        aggregated_action[:6] = window_actions[:, :6].sum(axis=0)  # Sum translations and rotations
        aggregated_action[6] = window_actions[-1, 6]  # Use last gripper state in window

        aggregated_actions.append(aggregated_action)

        # Base frame language action
        base_lang = summarize_numeric_actions(aggregated_action, sum_decimal, include_rotation=False)
        base_frame_actions.append(base_lang)

        # Transform to EEF frame using the current chunk's starting state
        # The EEF frame changes with each chunk based on robot's current pose
        current_state = cartesian_trajectory[start_idx]
        current_state_with_marker = np.concatenate([current_state, [0.0, 0.0]])
        eef_action = transform_actions_to_eef_frame(aggregated_action, current_state_with_marker)
        eef_lang = summarize_numeric_actions(eef_action, sum_decimal, include_rotation=False)
        eef_frame_actions.append(eef_lang)

    aggregated_actions = np.array(aggregated_actions)
    return base_frame_actions, eef_frame_actions, aggregated_actions


def add_text_to_frame(frame, text, position, font_scale=0.5, color=(255, 255, 255), thickness=1):
    """Add text to frame with black background for readability."""
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Get text size
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)

    # Draw black background
    x, y = position
    cv2.rectangle(frame, (x - 5, y - text_height - 5), (x + text_width + 5, y + baseline + 5), (0, 0, 0), -1)

    # Draw text
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

    return text_height + baseline + 10


def create_video_with_language_actions(
    video_path: str,
    base_frame_actions: list,
    eef_frame_actions: list,
    output_path: str,
    subsample_rate: int = 1,
):
    """
    Create a video with language actions displayed as a bar below the video.

    Args:
        video_path: Path to input video
        base_frame_actions: List of language action strings in base frame
        eef_frame_actions: List of language action strings in EEF frame
        output_path: Path to output video
        subsample_rate: Subsample rate for video frames (1 = no subsampling)
    """
    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video info: {width}x{height}, {fps} fps, {total_frames} frames")
    print(f"Number of language actions: {len(base_frame_actions)}")
    print(f"Subsample rate: {subsample_rate} (output fps will be {fps / subsample_rate:.1f})")

    # Add space for text below video
    text_bar_height = 150
    new_height = height + text_bar_height

    # Create video writer with adjusted fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_fps = fps / subsample_rate
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, new_height))

    frame_idx = 0
    action_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Only process frames that align with subsample rate
        if frame_idx % subsample_rate == 0:
            # Create expanded frame with black bar below
            expanded_frame = np.zeros((new_height, width, 3), dtype=np.uint8)
            expanded_frame[:height, :, :] = frame

            # Add text to black bar
            y_offset = height + 20

            # Frame and action info
            info_text = f"Original Frame: {frame_idx}/{total_frames} | Action Window: {action_idx}/{len(base_frame_actions)} (frames {action_idx * subsample_rate}-{(action_idx + 1) * subsample_rate})"
            y_offset += add_text_to_frame(
                expanded_frame, info_text, (10, y_offset), font_scale=0.4, color=(200, 200, 200)
            )

            # Base frame action
            base_text = f"Base Frame: {base_frame_actions[action_idx]}"
            y_offset += add_text_to_frame(
                expanded_frame, base_text, (10, y_offset), font_scale=0.6, color=(100, 255, 100), thickness=2
            )

            # EEF frame action
            eef_text = f"EEF Frame:  {eef_frame_actions[action_idx]}"
            y_offset += add_text_to_frame(
                expanded_frame, eef_text, (10, y_offset), font_scale=0.6, color=(255, 100, 100), thickness=2
            )

            out.write(expanded_frame)
            action_idx += 1

            if action_idx >= len(base_frame_actions):
                break

            if action_idx % 10 == 0:
                print(f"Processed action {action_idx}/{len(base_frame_actions)} (frame {frame_idx}/{total_frames})")

        frame_idx += 1

    cap.release()
    out.release()
    print(f"Video saved to: {output_path}")


def print_comparison(base_frame_actions, eef_frame_actions, delta_actions, num_samples=10):
    """Print a comparison of base frame vs EEF frame language actions."""
    print("\n" + "=" * 80)
    print("LANGUAGE ACTION COMPARISON (Base Frame vs EEF Frame)")
    print("=" * 80)

    # Find steps with largest movements
    delta_norms = np.linalg.norm(delta_actions[:, :3], axis=1)
    largest_indices = np.argsort(delta_norms)[-num_samples:][::-1]

    print(f"\nShowing {num_samples} steps with largest movements:")
    for idx in largest_indices:
        dx, dy, dz = delta_actions[idx, :3] * 100  # Convert to cm
        print(f"\nStep {idx} (delta: [{dx:.2f}, {dy:.2f}, {dz:.2f}] cm):")
        print(f"  Base Frame: {base_frame_actions[idx]}")
        print(f"  EEF Frame:  {eef_frame_actions[idx]}")

    print("\n" + "=" * 80)


def main():
    # Configuration
    SUBSAMPLE_RATE = 10  # Aggregate actions every 10 frames for better visualization
    DECIMAL_PLACES = 1  # Number of decimal places for verbose format

    # Paths
    base_dir = Path(__file__).parent.parent
    traj_path = base_dir / "Fri_Apr_21_21:03:49_2023" / "trajectory.h5"
    video_path = base_dir / "Fri_Apr_21_21:03:49_2023" / "recordings" / "MP4" / "28221883.mp4"
    output_path = base_dir / "language_actions_comparison.mp4"

    print(f"Loading trajectory from: {traj_path}")
    cartesian_position, gripper_position = load_trajectory(str(traj_path))
    print(f"Loaded {len(cartesian_position)} timesteps")
    print(f"Cartesian position shape: {cartesian_position.shape}")
    print(f"Gripper position shape: {gripper_position.shape}")

    print("\nComputing delta actions...")
    delta_actions = compute_delta_actions(cartesian_position, gripper_position)
    print(f"Delta actions shape: {delta_actions.shape}")

    # Print initial state for reference
    initial_state = cartesian_position[0]
    print("\nInitial state (x, y, z, roll, pitch, yaw):")
    print(f"  Position: {initial_state[:3]}")
    print(f"  Rotation: {initial_state[3:6]} rad = {initial_state[3:6] * 180 / np.pi} deg")

    print(f"\nGenerating language actions (subsampling every {SUBSAMPLE_RATE} frames)...")
    print("  Note: EEF frame is updated for each chunk based on current robot pose")
    base_frame_actions, eef_frame_actions, aggregated_actions = generate_language_actions(
        delta_actions,
        cartesian_position,  # Pass full trajectory so EEF frame can be updated per chunk
        format_type="verbose",
        decimal_places=DECIMAL_PLACES,
        subsample_rate=SUBSAMPLE_RATE,
    )
    print(f"Generated {len(base_frame_actions)} aggregated actions (from {len(delta_actions)} original actions)")

    # Print comparison
    print_comparison(base_frame_actions, eef_frame_actions, aggregated_actions, num_samples=10)

    print(f"\nCreating video with language actions (subsampled at rate {SUBSAMPLE_RATE})...")
    create_video_with_language_actions(
        str(video_path),
        base_frame_actions,
        eef_frame_actions,
        str(output_path),
        subsample_rate=SUBSAMPLE_RATE,
    )

    print(f"\n✓ Done! Output saved to: {output_path}")
    print(f"   - Aggregated {SUBSAMPLE_RATE} frames per action")
    print(f"   - Video duration: {len(base_frame_actions) / 60 * SUBSAMPLE_RATE:.1f} seconds at original speed")


if __name__ == "__main__":
    main()
