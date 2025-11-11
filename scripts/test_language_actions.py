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


def _round_to_nearest_n(value: float, n: int = 5) -> int:
    """Round a value to the nearest multiple of n."""
    return int(round(value / n) * n)


def summarize_numeric_actions(arr_like, sum_decimal: str, include_rotation: bool = False) -> str | None:
    """Convert numeric delta EE actions ([..., 7]) into a language string.

    Expects translation in indices [0,1,2] (meters) and gripper at index 6.
    Sums over time, converts meters→cm, emits signed directional commands and final gripper setting.
    """
    arr = np.asarray(arr_like, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.shape[-1] < 7:
        return None

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

    if include_rotation:
        droll_rad = float(arr[..., 3].sum())
        dpitch_rad = float(arr[..., 4].sum())
        dyaw_rad = float(arr[..., 5].sum())
        droll = _round_to_nearest_n(abs(droll_rad * 180.0 / np.pi), 5)
        dpitch = _round_to_nearest_n(abs(dpitch_rad * 180.0 / np.pi), 5)
        dyaw = _round_to_nearest_n(abs(dyaw_rad * 180.0 / np.pi), 5)

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
        if include_rotation:
            if droll_rad > 0:
                parts.append("tilt left")
            elif droll_rad < 0:
                parts.append("tilt right")
            if dpitch_rad > 0:
                parts.append("tilt back")
            elif dpitch_rad < 0:
                parts.append("tilt forward")
            if dyaw_rad > 0:
                parts.append("rotate counterclockwise")
            elif dyaw_rad < 0:
                parts.append("rotate clockwise")
    else:
        fmt_dx = _format_numeric(dx, sum_decimal)
        fmt_dy = _format_numeric(dy, sum_decimal)
        fmt_dz = _format_numeric(dz, sum_decimal)
        if dx_m > 0 and dx != 0:
            parts.append(f"move forward {fmt_dx} cm")
        elif dx_m < 0 and dx != 0:
            parts.append(f"move back {fmt_dx} cm")
        if dz_m > 0 and dz != 0:
            parts.append(f"move up {fmt_dz} cm")
        elif dz_m < 0 and dz != 0:
            parts.append(f"move down {fmt_dz} cm")
        if dy_m > 0 and dy != 0:
            parts.append(f"move left {fmt_dy} cm")
        elif dy_m < 0 and dy != 0:
            parts.append(f"move right {fmt_dy} cm")
        if include_rotation:
            if droll_rad > 0 and droll != 0:
                parts.append(f"tilt left {droll} degrees")
            elif droll_rad < 0 and droll != 0:
                parts.append(f"tilt right {droll} degrees")
            if dpitch_rad > 0 and dpitch != 0:
                parts.append(f"tilt back {dpitch} degrees")
            elif dpitch_rad < 0 and dpitch != 0:
                parts.append(f"tilt forward {dpitch} degrees")
            if dyaw_rad > 0 and dyaw != 0:
                parts.append(f"rotate counterclockwise {dyaw} degrees")
            elif dyaw_rad < 0 and dyaw != 0:
                parts.append(f"rotate clockwise {dyaw} degrees")

    # Final gripper value from last step
    g_last = float(arr[-1, 6])
    if g_last > 0.5:
        parts.append("open gripper")
    else:
        parts.append("close gripper")
    # parts.append(f"set gripper to {g_last:.0f}")
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

    # Transform position deltas
    delta_pos_base = actions[:3]
    delta_pos_eef = R_base_to_eef @ delta_pos_base
    transformed_actions[:3] = delta_pos_eef

    # Transform rotation deltas
    # For small rotation deltas represented as Euler angles, we can approximate
    # by transforming the rotation vector
    delta_rot_base = actions[3:6]
    delta_rot_eef = R_base_to_eef @ delta_rot_base
    transformed_actions[3:6] = delta_rot_eef

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

    # Ensure gripper is 1D
    if gripper_position.ndim == 2:
        gripper_position = gripper_position.squeeze()

    return cartesian_position, gripper_position


def _wrap_angle_delta(delta_rad):
    """Wrap angle delta to [-pi, pi] range."""
    return np.arctan2(np.sin(delta_rad), np.cos(delta_rad))


def summarize_absolute_state(state, gripper_value, frame_type="Base"):
    """
    Format absolute robot state as readable text.

    Args:
        state: array with [x, y, z, roll, pitch, yaw] in meters and radians
        gripper_value: gripper state (0-1, where 1 is open)
        frame_type: "Base" or "EEF" for display label

    Returns:
        Formatted string with position, rotation, and gripper state
    """
    x, y, z, roll, pitch, yaw = state[:6]

    # Convert to cm and degrees for readability
    x_cm = x * 100
    y_cm = y * 100
    z_cm = z * 100
    roll_deg = roll * 180 / np.pi
    pitch_deg = pitch * 180 / np.pi
    yaw_deg = yaw * 180 / np.pi

    # Format position
    pos_str = f"Pos: ({x_cm:6.1f}, {y_cm:6.1f}, {z_cm:6.1f}) cm"

    # Format rotation
    rot_str = f"Rot: ({roll_deg:6.1f}, {pitch_deg:6.1f}, {yaw_deg:6.1f}) deg"

    # Format gripper
    gripper_str = "Gripper: OPEN" if gripper_value > 0.5 else "Gripper: CLOSED"

    return f"{pos_str}  |  {rot_str}  |  {gripper_str}"


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
        # Wrap angle differences to [-pi, pi] to avoid jumps at 360°/0° boundary
        delta_roll = cartesian_position[i + 1, 3] - cartesian_position[i, 3]
        delta_pitch = cartesian_position[i + 1, 4] - cartesian_position[i, 4]
        delta_yaw = cartesian_position[i + 1, 5] - cartesian_position[i, 5]

        delta_actions[i, 3] = _wrap_angle_delta(delta_roll)
        delta_actions[i, 4] = _wrap_angle_delta(delta_pitch)
        delta_actions[i, 5] = _wrap_angle_delta(delta_yaw)

        # Gripper state (use next state)
        delta_actions[i, 6] = gripper_position[i + 1]

    return delta_actions


def compute_relative_state(current_state, initial_state):
    """
    Compute relative state in the initial EEF frame.

    Args:
        current_state: [x, y, z, roll, pitch, yaw] current pose
        initial_state: [x, y, z, roll, pitch, yaw] initial pose

    Returns:
        relative_state: [dx, dy, dz, droll, dpitch, dyaw] relative to initial pose in initial EEF frame
    """
    # Get position difference in base frame
    delta_pos_base = current_state[:3] - initial_state[:3]

    # Transform position difference to initial EEF frame
    initial_euler = initial_state[3:6]
    R_initial = R.from_euler("xyz", initial_euler)
    R_base_to_eef = R_initial.as_matrix().T  # Transpose to get EEF <- base
    delta_pos_eef = R_base_to_eef @ delta_pos_base

    # Compute rotation difference
    # For Euler angles, we can approximate the difference for small rotations
    delta_rot_base = current_state[3:6] - initial_state[3:6]
    # Wrap to [-pi, pi]
    delta_rot_base = np.array([_wrap_angle_delta(d) for d in delta_rot_base])
    # Transform to EEF frame
    delta_rot_eef = R_base_to_eef @ delta_rot_base

    relative_state = np.concatenate([delta_pos_eef, delta_rot_eef])
    return relative_state


def generate_language_actions(
    delta_actions, cartesian_trajectory, gripper_trajectory, format_type="verbose", decimal_places=1, subsample_rate=1
):
    """
    Generate absolute state descriptions in base frame and relative states in EEF frame.

    Args:
        delta_actions: (T, 7) array of delta actions in base frame (unused, kept for compatibility)
        cartesian_trajectory: (T+1, 6) array with EEF poses [x, y, z, roll, pitch, yaw] for each timestep
        gripper_trajectory: (T+1,) array with gripper states
        format_type: "verbose" or "compact"
        decimal_places: Number of decimal places for verbose format
        subsample_rate: Number of frames to subsample (e.g., 10 means show every 10th frame)

    Returns:
        base_frame_states: list of absolute state strings in base frame
        eef_frame_states: list of relative state strings in initial EEF frame
        subsampled_indices: array of frame indices that were used
    """
    base_frame_states = []
    eef_frame_states = []

    # Get initial state for EEF frame reference
    initial_state = cartesian_trajectory[0]

    # Subsample trajectory by taking every Nth frame
    T = len(cartesian_trajectory)
    num_windows = T // subsample_rate
    subsampled_indices = []

    for window_idx in range(num_windows):
        frame_idx = window_idx * subsample_rate

        # Get absolute state at this frame
        state = cartesian_trajectory[frame_idx]
        gripper_value = gripper_trajectory[frame_idx]

        # Format absolute state for base frame
        state_str = summarize_absolute_state(state, gripper_value, frame_type="Base")
        base_frame_states.append(state_str)

        # Compute and format relative state for EEF frame
        relative_state = compute_relative_state(state, initial_state)
        relative_state_str = summarize_absolute_state(relative_state, gripper_value, frame_type="EEF")
        eef_frame_states.append(relative_state_str)

        subsampled_indices.append(frame_idx)

    subsampled_indices = np.array(subsampled_indices)

    return base_frame_states, eef_frame_states, subsampled_indices


def add_text_to_frame(frame, text, position, font_scale=0.25, color=(255, 255, 255), thickness=1):
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
    base_frame_states: list,
    eef_frame_states: list,
    output_path: str,
    subsample_rate: int = 1,
):
    """
    Create a video with absolute robot states displayed as a bar below the video.

    Args:
        video_path: Path to input video
        base_frame_states: List of absolute state strings in base frame
        eef_frame_states: List of relative state strings in EEF frame
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
    print(f"Number of states to display: {len(base_frame_states)}")
    print(f"Subsample rate: {subsample_rate} (output fps will be {fps / subsample_rate:.1f})")

    # Add space for text below video - need room for 2 lines
    text_bar_height = 150
    new_height = height + text_bar_height

    # Create video writer with adjusted fps
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    output_fps = fps / subsample_rate
    out = cv2.VideoWriter(output_path, fourcc, output_fps, (width, new_height))

    frame_idx = 0
    state_idx = 0

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

            # Frame info
            info_text = f"Frame: {frame_idx}/{total_frames} | State #{state_idx}/{len(base_frame_states)}"
            y_offset += add_text_to_frame(
                expanded_frame, info_text, (10, y_offset), font_scale=0.8, color=(200, 200, 200)
            )

            # Base frame absolute state
            base_text = f"Base: {base_frame_states[state_idx]}"
            y_offset += add_text_to_frame(
                expanded_frame, base_text, (10, y_offset), font_scale=0.8, color=(100, 255, 100), thickness=2
            )

            # EEF frame relative state
            eef_text = f"EEF:  {eef_frame_states[state_idx]}"
            y_offset += add_text_to_frame(
                expanded_frame, eef_text, (10, y_offset), font_scale=0.8, color=(255, 100, 100), thickness=2
            )

            out.write(expanded_frame)
            state_idx += 1

            if state_idx >= len(base_frame_states):
                break

            if state_idx % 10 == 0:
                print(f"Processed state {state_idx}/{len(base_frame_states)} (frame {frame_idx}/{total_frames})")

        frame_idx += 1

    cap.release()
    out.release()
    print(f"Video saved to: {output_path}")


def print_comparison(base_frame_states, eef_frame_states, subsampled_indices, num_samples=10):
    """Print sample absolute states from the trajectory."""
    print("\n" + "=" * 80)
    print("STATE SAMPLES (Base Frame vs EEF Frame)")
    print("=" * 80)

    # Sample evenly across the trajectory
    sample_step = max(1, len(base_frame_states) // num_samples)
    sample_indices = list(range(0, len(base_frame_states), sample_step))[:num_samples]

    print(f"\nShowing {len(sample_indices)} sampled states from trajectory:")
    print("Base Frame = Absolute position in world frame")
    print("EEF Frame  = Relative position from initial EEF pose\n")

    for state_idx in sample_indices:
        frame_idx = subsampled_indices[state_idx]
        print(f"\nState #{state_idx} (Frame {frame_idx}):")
        print(f"  Base: {base_frame_states[state_idx]}")
        print(f"  EEF:  {eef_frame_states[state_idx]}")

    print("\n" + "=" * 80)


def main():
    # Configuration
    SUBSAMPLE_RATE = 1  # Aggregate actions every 10 frames for better visualization
    DECIMAL_PLACES = 1  # Number of decimal places for verbose format

    traj_folder_name = "Thu_Nov__9_10:01:09_2023"  # "Fri_Apr_21_21:03:49_2023"
    # cam_serial = "25947356"
    cam_serial = "14549178"

    # Paths
    base_dir = Path(__file__).parent.parent
    traj_path = base_dir / traj_folder_name / "trajectory.h5"
    video_path = base_dir / traj_folder_name / "recordings" / "MP4" / f"{cam_serial}.mp4"
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

    print(f"\nGenerating absolute state descriptions (subsampling every {SUBSAMPLE_RATE} frames)...")
    base_frame_states, eef_frame_states, subsampled_indices = generate_language_actions(
        delta_actions,
        cartesian_position,
        gripper_position,  # Pass gripper trajectory
        format_type="verbose",
        decimal_places=DECIMAL_PLACES,
        subsample_rate=SUBSAMPLE_RATE,
    )
    print(f"Generated {len(base_frame_states)} state descriptions (from {len(cartesian_position)} total frames)")

    # Print sample states
    print_comparison(base_frame_states, eef_frame_states, subsampled_indices, num_samples=10)

    print(f"\nCreating video with absolute states (subsampled at rate {SUBSAMPLE_RATE})...")
    create_video_with_language_actions(
        str(video_path),
        base_frame_states,
        eef_frame_states,
        str(output_path),
        subsample_rate=SUBSAMPLE_RATE,
    )

    print(f"\n✓ Done! Output saved to: {output_path}")
    print(f"   - Showing every {SUBSAMPLE_RATE}th frame")
    print(f"   - Total states shown: {len(base_frame_states)}")
    print(f"   - Video duration: {len(base_frame_states) / 60 * SUBSAMPLE_RATE:.1f} seconds at original speed")


if __name__ == "__main__":
    main()
