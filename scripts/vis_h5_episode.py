#!/usr/bin/env python3
"""Visualize items from an HDF5 robot episode file."""

import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from pathlib import Path


def load_h5_data(filepath: str) -> dict:
    """Load all relevant data from the HDF5 file."""
    data = {}
    with h5py.File(filepath, 'r') as f:
        # Images
        data['image_1'] = f['observation/image/1'][:]
        data['image_left'] = f['observation/image/31425515_left'][:]
        data['image_right'] = f['observation/image/31425515_right'][:]
        
        # Robot state
        data['cartesian_position'] = f['observation/robot_state/cartesian_position'][:]
        data['gripper_position'] = f['observation/robot_state/gripper_position'][:]
        data['joint_positions'] = f['observation/robot_state/joint_positions'][:]
        
        print(data['cartesian_position'].shape)
        
        # Actions
        data['action_cartesian'] = f['action/cartesian_position'][:]
        
        print(data['action_cartesian'].shape)
        
        data['action_gripper'] = f['action/gripper_action'][:]
        data['target_cartesian'] = f['action/target_cartesian_position'][:]
        data['target_gripper'] = f['action/target_gripper_position'][:]
        
        # Controller info
        data['stage'] = f['observation/stage'][:]
        data['success'] = f['observation/controller_info/success'][:]
        data['failure'] = f['observation/controller_info/failure'][:]
        
    return data


def create_interactive_viewer(data: dict, filepath: str):
    """Create an interactive matplotlib viewer for the episode."""
    n_frames = len(data['image_1'])
    
    # Create figure with subplots
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'Episode: {Path(filepath).name} ({n_frames} frames)', fontsize=14)
    
    # Image axes - 3 cameras on top row
    ax_img1 = fig.add_subplot(2, 4, 1)
    ax_img_left = fig.add_subplot(2, 4, 2)
    ax_img_right = fig.add_subplot(2, 4, 3)
    
    # Robot state plots
    ax_cart = fig.add_subplot(2, 4, 5)
    ax_joints = fig.add_subplot(2, 4, 6)
    ax_gripper = fig.add_subplot(2, 4, 7)
    ax_action = fig.add_subplot(2, 4, 8)
    
    # Info text area
    ax_info = fig.add_subplot(2, 4, 4)
    ax_info.axis('off')
    
    # Initial frame
    current_frame = [0]
    
    # Display images
    im1 = ax_img1.imshow(data['image_1'][0])
    ax_img1.set_title('Camera 1 (wrist)')
    ax_img1.axis('off')
    
    # Convert RGBA to RGB for display
    im_left = ax_img_left.imshow(data['image_left'][0][:, :, :3])
    ax_img_left.set_title('Left Camera')
    ax_img_left.axis('off')
    
    im_right = ax_img_right.imshow(data['image_right'][0][:, :, :3])
    ax_img_right.set_title('Right Camera')
    ax_img_right.axis('off')
    
    # Plot cartesian position
    time = np.arange(n_frames)
    cart_labels = ['x', 'y', 'z', 'rx', 'ry', 'rz']
    for i in range(6):
        ax_cart.plot(time, data['cartesian_position'][:, i], label=cart_labels[i], alpha=0.7)
    cart_vline = ax_cart.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax_cart.set_xlabel('Frame')
    ax_cart.set_ylabel('Position')
    ax_cart.set_title('Cartesian Position')
    ax_cart.legend(loc='upper right', fontsize=8)
    ax_cart.grid(True, alpha=0.3)
    
    # Plot joint positions
    for i in range(7):
        ax_joints.plot(time, data['joint_positions'][:, i], label=f'j{i+1}', alpha=0.7)
    joints_vline = ax_joints.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax_joints.set_xlabel('Frame')
    ax_joints.set_ylabel('Angle (rad)')
    ax_joints.set_title('Joint Positions')
    ax_joints.legend(loc='upper right', fontsize=8)
    ax_joints.grid(True, alpha=0.3)
    
    # Plot gripper
    ax_gripper.plot(time, data['gripper_position'], label='state', linewidth=2)
    ax_gripper.plot(time, data['action_gripper'], label='action', linestyle='--', alpha=0.7)
    ax_gripper.plot(time, data['target_gripper'], label='target', linestyle=':', alpha=0.7)
    gripper_vline = ax_gripper.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax_gripper.set_xlabel('Frame')
    ax_gripper.set_ylabel('Gripper')
    ax_gripper.set_title('Gripper State')
    ax_gripper.legend(loc='upper right', fontsize=8)
    ax_gripper.grid(True, alpha=0.3)
    
    # Plot action cartesian (first 3 dims = xyz)
    action_labels = ['ax', 'ay', 'az', 'arx', 'ary', 'arz', 'ag']
    for i in range(min(7, data['action_cartesian'].shape[1])):
        ax_action.plot(time, data['action_cartesian'][:, i], label=action_labels[i], alpha=0.7)
    action_vline = ax_action.axvline(x=0, color='r', linestyle='--', linewidth=2)
    ax_action.set_xlabel('Frame')
    ax_action.set_ylabel('Action')
    ax_action.set_title('Cartesian Action')
    ax_action.legend(loc='upper right', fontsize=8)
    ax_action.grid(True, alpha=0.3)
    
    # Info text
    info_text = ax_info.text(0.1, 0.5, '', transform=ax_info.transAxes, 
                             fontsize=11, verticalalignment='center',
                             fontfamily='monospace')
    
    def update_info(frame):
        info_str = f"""Frame: {frame} / {n_frames-1}
        
Stage: {data['stage'][frame]}
Success: {data['success'][frame]}
Failure: {data['failure'][frame]}

Cartesian Pos:
  x: {data['cartesian_position'][frame, 0]:.4f}
  y: {data['cartesian_position'][frame, 1]:.4f}
  z: {data['cartesian_position'][frame, 2]:.4f}

Gripper: {data['gripper_position'][frame]:.4f}
"""
        info_text.set_text(info_str)
    
    def update(frame):
        frame = int(frame)
        current_frame[0] = frame
        
        # Update images
        im1.set_data(data['image_1'][frame])
        im_left.set_data(data['image_left'][frame][:, :, :3])
        im_right.set_data(data['image_right'][frame][:, :, :3])
        
        # Update vertical lines
        cart_vline.set_xdata([frame, frame])
        joints_vline.set_xdata([frame, frame])
        gripper_vline.set_xdata([frame, frame])
        action_vline.set_xdata([frame, frame])
        
        # Update info
        update_info(frame)
        
        fig.canvas.draw_idle()
    
    # Add slider
    ax_slider = fig.add_axes([0.15, 0.02, 0.7, 0.02])
    slider = Slider(ax_slider, 'Frame', 0, n_frames - 1, valinit=0, valstep=1)
    slider.on_changed(update)
    
    # Add play/pause button
    ax_play = fig.add_axes([0.05, 0.02, 0.08, 0.03])
    play_button = Button(ax_play, 'Play')
    
    playing = [False]
    
    def toggle_play(event):
        playing[0] = not playing[0]
        play_button.label.set_text('Pause' if playing[0] else 'Play')
    
    play_button.on_clicked(toggle_play)
    
    # Animation timer
    def animate(event):
        if playing[0]:
            next_frame = (current_frame[0] + 1) % n_frames
            slider.set_val(next_frame)
    
    timer = fig.canvas.new_timer(interval=50)  # 20 fps
    timer.add_callback(animate, None)
    timer.start()
    
    # Initial update
    update_info(0)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)
    plt.show()


def save_video(data: dict, output_path: str, fps: int = 20):
    """Save the episode as a video file."""
    try:
        import cv2
    except ImportError:
        print("OpenCV not installed. Install with: pip install opencv-python")
        return
    
    n_frames = len(data['image_1'])
    
    # Get dimensions for combined frame
    h1, w1 = data['image_1'].shape[1:3]
    h2, w2 = data['image_left'].shape[1:3]
    
    # Scale stereo images to match height of camera 1
    scale = h1 / h2
    new_w2 = int(w2 * scale)
    
    # Total width
    total_w = w1 + new_w2 * 2
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (total_w, h1))
    
    print(f"Saving video to {output_path}...")
    for i in range(n_frames):
        # Get frames
        img1 = data['image_1'][i]
        img_left = data['image_left'][i][:, :, :3]
        img_right = data['image_right'][i][:, :, :3]
        
        # Resize stereo images
        img_left_resized = cv2.resize(img_left, (new_w2, h1))
        img_right_resized = cv2.resize(img_right, (new_w2, h1))
        
        # Combine horizontally
        combined = np.hstack([img_left_resized, img1, img_right_resized])
        
        # Convert RGB to BGR for OpenCV
        combined_bgr = cv2.cvtColor(combined, cv2.COLOR_RGB2BGR)
        
        # Add frame number text
        cv2.putText(combined_bgr, f'Frame: {i}/{n_frames-1}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        out.write(combined_bgr)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{n_frames} frames")
    
    out.release()
    print(f"Video saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Visualize HDF5 robot episode')
    parser.add_argument('filepath', type=str, help='Path to the HDF5 file')
    parser.add_argument('--save-video', type=str, default=None,
                        help='Save as video to the specified path')
    parser.add_argument('--fps', type=int, default=20,
                        help='FPS for saved video (default: 20)')
    args = parser.parse_args()
    
    filepath = Path(args.filepath).expanduser()
    if not filepath.exists():
        print(f"Error: File not found: {filepath}")
        return
    
    print(f"Loading data from {filepath}...")
    data = load_h5_data(str(filepath))
    print(f"Loaded {len(data['image_1'])} frames")
    
    if args.save_video:
        save_video(data, args.save_video, args.fps)
    else:
        create_interactive_viewer(data, str(filepath))


if __name__ == '__main__':
    main()
