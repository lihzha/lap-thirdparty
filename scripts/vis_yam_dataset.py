#!/usr/bin/env python3
"""Visualize YAM TFDS dataset with image stream and gripper position over time."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np
import tensorflow_datasets as tfds
import cv2


def load_yam_dataset(dataset_path: str, split: str = "train"):
    """Load YAM dataset from the given path.
    
    Args:
        dataset_path: Path to the TFDS dataset directory
        split: Dataset split to load (default: train)
        
    Returns:
        TensorFlow dataset
    """
    # Load the dataset using builder_from_directory
    builder = tfds.builder_from_directory(dataset_path)
    ds = builder.as_dataset(split=split)
    return ds


def extract_episode_data(episode) -> dict:
    """Extract all data from a single episode into numpy arrays.
    
    Args:
        episode: TFDS episode dictionary
        
    Returns:
        Dictionary with extracted episode data
    """
    images = []
    states = []
    actions = []
    rewards = []
    language_instruction = None
    
    for step in episode['steps']:
        # Extract image (224, 224, 3) uint8 RGB
        image = step['observation']['image'].numpy()
        images.append(image)
        
        # Extract state (7,) float32: [ee_pos(3), ee_euler(3), gripper_pos(1)])
        state = step['observation']['state'].numpy()
        states.append(state)
        
        # Extract action (7,) float32
        action = step['action'].numpy()
        actions.append(action)
        
        # Extract reward
        reward = step['reward'].numpy()
        rewards.append(reward)
        
        # Extract language instruction (same for all steps)
        if language_instruction is None:
            language_instruction = step['language_instruction'].numpy().decode('utf-8')
    
    # Extract gripper positions (last element of state, index 6)
    gripper_positions = np.array([state[6] for state in states])
    
    return {
        'images': np.array(images),
        'states': np.array(states),
        'actions': np.array(actions),
        'rewards': np.array(rewards),
        'gripper_positions': gripper_positions,
        'language_instruction': language_instruction,
        'file_path': episode['episode_metadata']['file_path'].numpy().decode('utf-8'),
    }


def create_interactive_viewer(episodes: list, dataset_path: str):
    """Create an interactive matplotlib viewer for the dataset.
    
    Args:
        episodes: List of episode dictionaries
        dataset_path: Path to the dataset (for display)
    """
    n_episodes = len(episodes)
    if n_episodes == 0:
        print("No episodes to visualize")
        return
    
    # Extract first episode to get dimensions
    episode_data = extract_episode_data(episodes[0])
    n_frames = len(episode_data['images'])
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    
    # Image subplot (left side)
    ax_image = plt.subplot(2, 2, (1, 3))
    ax_image.set_title("Video Frame", fontsize=12, fontweight='bold')
    ax_image.axis('off')
    im_img = ax_image.imshow(episode_data['images'][0])
    
    # Gripper position plot (top right)
    ax_gripper = plt.subplot(2, 2, 2)
    ax_gripper.set_title("Gripper Position Over Time", fontsize=12, fontweight='bold')
    ax_gripper.set_xlabel("Frame")
    ax_gripper.set_ylabel("Gripper Position")
    ax_gripper.grid(True, alpha=0.3)
    
    # Plot gripper position
    frames = np.arange(n_frames)
    gripper_line, = ax_gripper.plot(frames, episode_data['gripper_positions'], 'b-', linewidth=2, label='Gripper Position')
    gripper_vline = ax_gripper.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Current Frame')
    ax_gripper.legend()
    ax_gripper.set_xlim(0, n_frames - 1)
    
    # Info text (bottom right)
    ax_info = plt.subplot(2, 2, 4)
    ax_info.axis('off')
    info_text = ax_info.text(0.1, 0.9, '', transform=ax_info.transAxes, 
                            fontsize=10, verticalalignment='top',
                            family='monospace', wrap=True)
    
    # State variables
    current_frame = [0]
    current_episode_idx = [0]
    episode_data = [extract_episode_data(episodes[0])]
    is_playing = [False]
    
    def update_frame_info(frame):
        """Update the info text with current frame information."""
        data = episode_data[0]
        state = data['states'][frame]
        action = data['actions'][frame]
        reward = data['rewards'][frame]
        
        info_str = f"""Episode: {current_episode_idx[0] + 1}/{n_episodes}
Frame: {frame}/{n_frames - 1}

State:
  EE Position: [{state[0]:.3f}, {state[1]:.3f}, {state[2]:.3f}]
  EE Euler: [{state[3]:.3f}, {state[4]:.3f}, {state[5]:.3f}]
  Gripper: {state[6]:.3f}

Action:
  Joint Pos: [{action[0]:.3f}, {action[1]:.3f}, {action[2]:.3f}, {action[3]:.3f}, {action[4]:.3f}, {action[5]:.3f}, {action[6]:.3f}]

Reward: {reward:.3f}

Task: {data['language_instruction']}

File: {Path(data['file_path']).name}"""
        info_text.set_text(info_str)
    
    def update_frame(frame):
        """Update the display with a new frame."""
        frame = int(frame)
        current_frame[0] = frame
        data = episode_data[0]
        
        # Update image
        im_img.set_data(data['images'][frame])
        
        # Update vertical line in gripper plot
        gripper_vline.set_xdata([frame, frame])
        
        # Update info
        update_frame_info(frame)
        
        fig.canvas.draw_idle()
    
    def load_episode(idx):
        """Load a new episode."""
        current_episode_idx[0] = idx
        episode_data[0] = extract_episode_data(episodes[idx])
        
        # Update plots
        n_frames_new = len(episode_data[0]['images'])
        frames_new = np.arange(n_frames_new)
        
        # Update gripper plot
        gripper_line.set_xdata(frames_new)
        gripper_line.set_ydata(episode_data[0]['gripper_positions'])
        ax_gripper.set_xlim(0, n_frames_new - 1)
        ax_gripper.set_ylim(
            episode_data[0]['gripper_positions'].min() - 0.1,
            episode_data[0]['gripper_positions'].max() + 0.1
        )
        
        # Update slider range
        frame_slider.valmax = n_frames_new - 1
        frame_slider.ax.set_xlim(0, n_frames_new - 1)
        frame_slider.set_val(0)
        
        # Update episode slider
        episode_slider.set_val(idx)
        
        update_frame(0)
    
    # Add frame slider
    ax_frame_slider = fig.add_axes([0.15, 0.02, 0.5, 0.02])
    frame_slider = Slider(ax_frame_slider, 'Frame', 0, n_frames - 1, valinit=0, valstep=1)
    frame_slider.on_changed(update_frame)
    
    # Add episode slider
    ax_episode_slider = fig.add_axes([0.15, 0.05, 0.5, 0.02])
    episode_slider = Slider(ax_episode_slider, 'Episode', 0, n_episodes - 1, valinit=0, valstep=1)
    
    def on_episode_change(val):
        idx = int(val)
        if idx != current_episode_idx[0]:
            load_episode(idx)
    
    episode_slider.on_changed(on_episode_change)
    
    # Add play/pause button
    ax_play = fig.add_axes([0.7, 0.02, 0.08, 0.03])
    play_button = Button(ax_play, 'Play/Pause')
    
    def toggle_play(event):
        is_playing[0] = not is_playing[0]
        if is_playing[0]:
            play_animation()
    
    play_button.on_clicked(toggle_play)
    
    def play_animation():
        """Play animation automatically."""
        import time
        frame_delay = 0.1  # 10 FPS
        
        while is_playing[0] and current_frame[0] < len(episode_data[0]['images']) - 1:
            current_frame[0] += 1
            frame_slider.set_val(current_frame[0])
            plt.pause(frame_delay)
        
        is_playing[0] = False
    
    # Add next/prev episode buttons
    ax_next = fig.add_axes([0.8, 0.02, 0.08, 0.03])
    next_button = Button(ax_next, 'Next')
    
    def next_episode(event):
        if current_episode_idx[0] < n_episodes - 1:
            load_episode(current_episode_idx[0] + 1)
    
    next_button.on_clicked(next_episode)
    
    ax_prev = fig.add_axes([0.7, 0.05, 0.08, 0.03])
    prev_button = Button(ax_prev, 'Prev')
    
    def prev_episode(event):
        if current_episode_idx[0] > 0:
            load_episode(current_episode_idx[0] - 1)
    
    prev_button.on_clicked(prev_episode)
    
    # Initialize display
    update_frame_info(0)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.show()


def save_video(episodes: list, output_path: str, fps: int = 10, max_episodes: int = None):
    """Save episodes as a video file with image and gripper plot.
    
    Args:
        episodes: List of episode dictionaries
        output_path: Path to save the video
        fps: Frames per second for the video
        max_episodes: Maximum number of episodes to include (None for all)
    """
    if max_episodes is not None:
        episodes = episodes[:max_episodes]
    
    # Get dimensions from first episode
    sample = extract_episode_data(episodes[0])
    img_h, img_w = sample['images'][0].shape[:2]
    
    # Video dimensions: image on left, plot on right
    plot_width = 600
    plot_height = 400
    total_width = img_w + plot_width
    total_height = max(img_h, plot_height) + 60  # Extra space for text
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (total_width, total_height))
    
    print(f"Saving video to {output_path}...")
    
    for ep_idx, episode in enumerate(episodes):
        data = extract_episode_data(episode)
        n_frames = len(data['images'])
        
        print(f"  Episode {ep_idx + 1}/{len(episodes)}: {n_frames} frames")
        
        # Create matplotlib figure for plotting
        fig, ax = plt.subplots(figsize=(plot_width/100, plot_height/100), dpi=100)
        ax.set_title("Gripper Position Over Time", fontsize=10, fontweight='bold')
        ax.set_xlabel("Frame")
        ax.set_ylabel("Gripper Position")
        ax.grid(True, alpha=0.3)
        
        frames = np.arange(n_frames)
        ax.plot(frames, data['gripper_positions'], 'b-', linewidth=2)
        ax.set_xlim(0, n_frames - 1)
        ax.set_ylim(
            data['gripper_positions'].min() - 0.1,
            data['gripper_positions'].max() + 0.1
        )
        
        for i in range(n_frames):
            # Create frame
            frame = np.zeros((total_height, total_width, 3), dtype=np.uint8)
            
            # Add image on left
            frame[60:60+img_h, :img_w] = data['images'][i]
            
            # Update plot with current frame marker
            ax.clear()
            ax.set_title("Gripper Position Over Time", fontsize=10, fontweight='bold')
            ax.set_xlabel("Frame")
            ax.set_ylabel("Gripper Position")
            ax.grid(True, alpha=0.3)
            ax.plot(frames, data['gripper_positions'], 'b-', linewidth=2)
            ax.axvline(x=i, color='r', linestyle='--', linewidth=2)
            ax.set_xlim(0, n_frames - 1)
            ax.set_ylim(
                data['gripper_positions'].min() - 0.1,
                data['gripper_positions'].max() + 0.1
            )
            
            # Render plot to image
            fig.canvas.draw()
            plot_img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
            plot_img = plot_img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            
            # Resize plot to match desired dimensions
            plot_img_resized = cv2.resize(plot_img, (plot_width, plot_height))
            
            # Add plot on right
            frame[60:60+plot_height, img_w:img_w+plot_width] = plot_img_resized
            
            # Convert RGB to BGR for OpenCV
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            # Add text
            cv2.putText(frame_bgr, f'Ep {ep_idx+1}/{len(episodes)} Frame {i}/{n_frames-1}',
                       (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.putText(frame_bgr, f'Gripper: {data["gripper_positions"][i]:.3f}',
                       (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            cv2.putText(frame_bgr, f'Task: {data["language_instruction"][:50]}',
                       (img_w + 10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            
            out.write(frame_bgr)
        
        plt.close(fig)
    
    out.release()
    print(f"Video saved to {output_path}")


def print_dataset_info(dataset_path: str):
    """Print information about the dataset."""
    builder = tfds.builder_from_directory(dataset_path)
    info = builder.info
    
    print(f"Dataset: {info.name}")
    print(f"Version: {info.version}")
    print(f"Description: {info.description}")
    print(f"\nSplits:")
    for split_name, split_info in info.splits.items():
        print(f"  {split_name}: {split_info.num_examples} episodes")


def main():
    parser = argparse.ArgumentParser(
        description='Visualize YAM TFDS dataset with image stream and gripper position',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Interactive viewer
  python scripts/vis_yam_dataset.py ~/tensorflow_datasets/yam_dataset/1.0.0/
  
  # Save as video
  python scripts/vis_yam_dataset.py ~/tensorflow_datasets/yam_dataset/1.0.0/ --save-video output.mp4
  
  # Print dataset info
  python scripts/vis_yam_dataset.py ~/tensorflow_datasets/yam_dataset/1.0.0/ --info
        """
    )
    parser.add_argument('dataset_path', type=str,
                        help='Path to the TFDS dataset directory (e.g., ~/tensorflow_datasets/yam_dataset/1.0.0/)')
    parser.add_argument('--split', type=str, default='train',
                        help='Dataset split to visualize (default: train)')
    parser.add_argument('--save-video', type=str, default=None,
                        help='Save as video to the specified path')
    parser.add_argument('--fps', type=int, default=10,
                        help='FPS for saved video (default: 10)')
    parser.add_argument('--max-episodes', type=int, default=None,
                        help='Maximum number of episodes for video (default: all)')
    parser.add_argument('--max-episodes-viewer', type=int, default=50,
                        help='Maximum number of episodes to load for interactive viewer (default: 50)')
    parser.add_argument('--info', action='store_true',
                        help='Print dataset info and exit')
    args = parser.parse_args()
    
    dataset_path = str(Path(args.dataset_path).expanduser())
    
    if not Path(dataset_path).exists():
        print(f"Error: Dataset path not found: {dataset_path}")
        return
    
    if args.info:
        print_dataset_info(dataset_path)
        return
    
    print(f"Loading dataset from {dataset_path}...")
    ds = load_yam_dataset(dataset_path, args.split)
    
    # Convert to list for random access
    print("Loading episodes into memory...")
    if args.save_video:
        episodes = list(ds)
        if args.max_episodes:
            episodes = episodes[:args.max_episodes]
    else:
        # Limit episodes for interactive viewer to avoid memory issues
        episodes = list(ds.take(args.max_episodes_viewer))
    
    print(f"Loaded {len(episodes)} episodes")
    
    if args.save_video:
        save_video(episodes, args.save_video, args.fps, args.max_episodes)
    else:
        create_interactive_viewer(episodes, dataset_path)


if __name__ == '__main__':
    main()
