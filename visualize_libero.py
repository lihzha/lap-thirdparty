import cv2
import numpy as np
import tensorflow_datasets as tfds

ds = tfds.load("libero_10_no_noops", split="train", shuffle_files=False)


def save_trajectory_video(trajectory, output_path="trajectory.mp4", img_key="image", max_frames=None, fps=10):
    """
    trajectory: a dict from RLDS where trajectory['steps'] contains lists of elements.
    output_path: path to save the video file.
    img_key: which observation image key to display (e.g., 'image', 'front', 'rgb').
    max_frames: maximum number of frames to include (None = all frames).
    fps: frames per second for the output video.
    """
    # Collect images and actions from steps
    frames = []
    for step in trajectory["steps"]:
        img = step["observation"][img_key].numpy()
        action = step["action"].numpy()

        # Convert image to BGR for OpenCV
        if img.dtype != np.uint8:
            img = (img * 255).astype(np.uint8)
        if len(img.shape) == 3 and img.shape[2] == 3:
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        else:
            img_bgr = img

        # Add action text to the image
        img_with_text = img_bgr.copy()

        # Format action for display
        action_str = "Action: [" + ", ".join([f"{a:.3f}" for a in action[-1:]]) + "]"

        # Add background rectangle for better text visibility
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        text_size = cv2.getTextSize(action_str, font, font_scale, thickness)[0]

        # Position text at top of image
        text_x = 5
        text_y = 15

        # Draw black background rectangle
        cv2.rectangle(
            img_with_text,
            (text_x - 2, text_y - text_size[1] - 2),
            (text_x + text_size[0] + 2, text_y + 2),
            (0, 0, 0),
            -1,
        )

        # Draw text in white
        cv2.putText(
            img_with_text, action_str, (text_x, text_y), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA
        )

        frames.append(img_with_text)

        if max_frames is not None and len(frames) >= max_frames:
            break

    if len(frames) == 0:
        print("No frames found in trajectory")
        return

    # Get frame dimensions
    height, width = frames[0].shape[:2]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Write frames
    for frame in frames:
        out.write(frame)

    out.release()
    print(f"Video saved to {output_path} with {len(frames)} frames")


def _render_state_frame(state, size=(640, 480), max_abs=None, last_series=None, last_index=None, last_max_abs=None):
    state = np.asarray(state).flatten()
    if state.size == 0:
        return np.zeros((size[1], size[0], 3), dtype=np.uint8)

    height, width = size[1], size[0]
    margin = 10
    plot_h = height // 3 if last_series is not None and last_index is not None else 0
    bars_h = height - plot_h
    frame = np.full((height, width, 3), 20, dtype=np.uint8)

    if max_abs is None:
        max_abs = float(np.max(np.abs(state)))
        max_abs = max(max_abs, 1e-6)

    mid_y = bars_h // 2
    cv2.line(frame, (margin, mid_y), (width - margin, mid_y), (80, 80, 80), 1)

    bar_space = max(1, width - 2 * margin)
    bar_width = max(1, bar_space // state.size)
    usable_height = max(1, (bars_h // 2) - margin)

    for i, val in enumerate(state):
        x0 = margin + i * bar_width
        x1 = min(width - margin - 1, x0 + bar_width - 1)
        if x0 >= width - margin:
            break
        bar_h = int((abs(float(val)) / max_abs) * usable_height)
        if val >= 0:
            y0, y1 = max(margin, mid_y - bar_h), mid_y - 1
        else:
            y0, y1 = mid_y + 1, min(bars_h - margin - 1, mid_y + bar_h)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 180, 255), -1)

    label = f"state dims={state.size} max_abs={max_abs:.3f}"
    cv2.putText(frame, label, (margin, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

    if plot_h > 0:
        y_start = bars_h
        y_end = height - 1
        cv2.rectangle(frame, (0, y_start), (width - 1, y_end), (15, 15, 15), -1)

        if last_max_abs is None:
            last_max_abs = float(np.max(np.abs(last_series)))
            last_max_abs = max(last_max_abs, 1e-6)

        mid_plot_y = y_start + plot_h // 2
        cv2.line(frame, (margin, mid_plot_y), (width - margin, mid_plot_y), (80, 80, 80), 1)

        usable_plot_h = max(1, (plot_h // 2) - margin)
        plot_w = max(1, width - 2 * margin)
        count = last_index + 1
        for i in range(1, count):
            x0 = margin + int((i - 1) * (plot_w - 1) / max(1, count - 1))
            x1 = margin + int(i * (plot_w - 1) / max(1, count - 1))
            v0 = float(last_series[i - 1])
            v1 = float(last_series[i])
            y0 = mid_plot_y - int((v0 / last_max_abs) * usable_plot_h)
            y1 = mid_plot_y - int((v1 / last_max_abs) * usable_plot_h)
            cv2.line(frame, (x0, y0), (x1, y1), (255, 180, 0), 1)

        last_val = float(last_series[last_index])
        plot_label = f"last dim={last_val:.3f}"
        cv2.putText(
            frame,
            plot_label,
            (margin, y_start + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )
    return frame


def save_state_video(trajectory, output_path="state.mp4", state_key="state", max_frames=None, fps=10, size=(640, 480)):
    """
    trajectory: a dict from RLDS where trajectory['steps'] contains lists of elements.
    output_path: path to save the video file.
    state_key: which observation state key to visualize (e.g., 'state').
    max_frames: maximum number of frames to include (None = all frames).
    fps: frames per second for the output video.
    size: (width, height) of the state visualization frames.
    """
    states = []
    for step in trajectory["steps"]:
        state = step["observation"][state_key].numpy()
        print(state)
        states.append(state)
        if max_frames is not None and len(states) >= max_frames:
            break

    if len(states) == 0:
        print("No states found in trajectory")
        return

    max_abs = float(np.max([np.max(np.abs(s)) for s in states]))
    max_abs = max(max_abs, 1e-6)
    last_series = [float(np.asarray(s).flatten()[-2]) for s in states]
    print("Last series:", last_series)
    last_max_abs = float(np.max(np.abs(last_series)))
    last_max_abs = max(last_max_abs, 1e-6)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, size)

    for i, state in enumerate(states):
        frame = _render_state_frame(
            state,
            size=size,
            max_abs=max_abs,
            last_series=last_series,
            last_index=i,
            last_max_abs=last_max_abs,
        )
        out.write(frame)

    out.release()
    print(f"State video saved to {output_path} with {len(states)} frames")


# --- Example usage ---
traj = next(iter(ds))  # get one trajectory
save_trajectory_video(traj, output_path="trajectory.mp4", img_key="image")
save_state_video(traj, output_path="state.mp4", state_key="state")
