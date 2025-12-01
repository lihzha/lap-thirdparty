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
        action_str = "Action: [" + ", ".join([f"{a:.3f}" for a in action]) + "]"

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


# --- Example usage ---
traj = next(iter(ds))  # get one trajectory
save_trajectory_video(traj, output_path="trajectory.mp4", img_key="image")
