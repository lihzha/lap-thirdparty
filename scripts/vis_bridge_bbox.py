#!/usr/bin/env python3
"""Visualize Bridge dataset with bounding box annotations from JSONL file.

This script loads bridge data from gs://pi0-cot/OXE/bridge_v2_oxe and visualizes
the bounding box annotations from the corresponding JSONL file.

Usage:
    # Interactive viewer (local)
    python scripts/vis_bridge_bbox.py
    
    # Log to wandb
    python scripts/vis_bridge_bbox.py --wandb --wandb-project bridge-bbox-vis
    
    # Save frames locally
    python scripts/vis_bridge_bbox.py --save-dir ./output_frames
"""

import argparse
import json
import os
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
import numpy as np
import tensorflow_datasets as tfds

try:
    import wandb
except ImportError:
    wandb = None


def load_bbox_annotations(jsonl_path: str) -> tuple[dict[str, dict], dict[str, dict]]:
    """Load bounding box annotations from JSONL file.
    
    Args:
        jsonl_path: Path to the JSONL file with bbox annotations
        
    Returns:
        Tuple of:
        - Dictionary mapping UUID to annotation data
        - Dictionary mapping file_path to annotation data (for matching)
    """
    annotations = {}
    filepath_to_annot = {}
    
    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                uuid = data["uuid"]
                annotations[uuid] = data
                
                # Also index by file_path for matching
                episode_metadata = data.get("episode_metadata", {})
                file_path = episode_metadata.get("file_path", "")
                if file_path:
                    filepath_to_annot[file_path] = data
    
    print(f"Loaded {len(annotations)} bbox annotations from {jsonl_path}")
    print(f"  - {len(filepath_to_annot)} unique file_paths for matching")
    return annotations, filepath_to_annot


def parse_uuid(uuid: str) -> tuple[str, str, int]:
    """Parse UUID to extract tfrecord path and episode index.
    
    UUID format: gs://pi0-cot/OXE/bridge_v2_oxe/1.0.0/bridge_dataset-train.tfrecord-00000-of-01024::episode_0
    
    Returns:
        (base_path, tfrecord_file, episode_index)
    """
    # Split by :: to separate tfrecord path from episode
    tfrecord_path, episode_part = uuid.rsplit("::", 1)
    episode_idx = int(episode_part.replace("episode_", ""))
    
    # Extract the tfrecord file name and base path
    parts = tfrecord_path.rsplit("/", 1)
    if len(parts) == 2:
        base_path, tfrecord_file = parts
    else:
        base_path = ""
        tfrecord_file = parts[0]
    
    return base_path, tfrecord_file, episode_idx


def load_bridge_dataset(data_dir: str, split: str = "train", max_episodes: int | None = None):
    """Load bridge dataset from the given directory.
    
    Args:
        data_dir: Path to bridge_v2_oxe dataset (GCS or local)
        split: Dataset split (train/val)
        max_episodes: Maximum number of episodes to load
        
    Returns:
        List of episode data dictionaries
    """
    print(f"Loading dataset from {data_dir}...")
    
    # Try loading as TFDS builder
    try:
        builder = tfds.builder_from_directory(data_dir)
        ds = builder.as_dataset(split=split)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Trying alternative loading method...")
        # Alternative: use RLDS loader directly
        import tensorflow as tf
        ds = tf.data.Dataset.list_files(f"{data_dir}/*.tfrecord*")
        raise NotImplementedError("Direct tfrecord loading not implemented yet")
    
    episodes = []
    for i, episode in enumerate(ds):
        if max_episodes is not None and i >= max_episodes:
            break
        episodes.append(episode)
        if (i + 1) % 100 == 0:
            print(f"  Loaded {i + 1} episodes...")
    
    print(f"Loaded {len(episodes)} episodes")
    return episodes


def extract_episode_data(episode, episode_idx: int) -> dict:
    """Extract all data from a single episode into numpy arrays.
    
    Args:
        episode: TFDS episode object
        episode_idx: Index of the episode
        
    Returns:
        Dictionary with images, states, actions, etc.
    """
    images = []
    wrist_images = []
    states = []
    actions = []
    language_instruction = None
    
    for step in episode['steps']:
        obs = step['observation']
        
        # Get primary image (try different key names)
        if 'image' in obs:
            images.append(obs['image'].numpy())
        elif 'image_0' in obs:
            images.append(obs['image_0'].numpy())
        else:
            # Find any image key
            for key in obs:
                if 'image' in key.lower() and 'wrist' not in key.lower():
                    images.append(obs[key].numpy())
                    break
        
        # Get wrist image if available
        if 'wrist_image' in obs:
            wrist_images.append(obs['wrist_image'].numpy())
        elif 'wrist_image_left' in obs:
            wrist_images.append(obs['wrist_image_left'].numpy())
        
        # Get state if available
        if 'state' in obs:
            states.append(obs['state'].numpy())
        
        # Get action
        if 'action' in step:
            actions.append(step['action'].numpy())
        
        # Get language instruction
        if language_instruction is None:
            if 'language_instruction' in step:
                instr = step['language_instruction'].numpy()
                if isinstance(instr, bytes):
                    language_instruction = instr.decode('utf-8')
                else:
                    language_instruction = str(instr)
    
    # Get episode metadata
    file_path = ""
    if 'episode_metadata' in episode:
        metadata = episode['episode_metadata']
        if 'file_path' in metadata:
            fp = metadata['file_path'].numpy()
            if isinstance(fp, bytes):
                file_path = fp.decode('utf-8')
            else:
                file_path = str(fp)
    
    return {
        'images': np.array(images) if images else np.array([]),
        'wrist_images': np.array(wrist_images) if wrist_images else np.array([]),
        'states': np.array(states) if states else np.array([]),
        'actions': np.array(actions) if actions else np.array([]),
        'language_instruction': language_instruction or "",
        'file_path': file_path,
        'episode_idx': episode_idx,
    }


def draw_bbox_on_image(
    img: np.ndarray,
    bbox: list[int],
    label: str,
    is_target: bool | None,
    color: tuple[int, int, int] | None = None,
    thickness: int = 2,
) -> np.ndarray:
    """Draw a bounding box on an image.
    
    Args:
        img: Image array (H, W, 3)
        bbox: [x1, y1, x2, y2] in pixel coordinates (1000x1000 scale)
        label: Object label
        is_target: Whether this is the target object
        color: RGB color tuple (if None, auto-assign based on is_target)
        thickness: Line thickness
        
    Returns:
        Image with bbox drawn
    """
    img = img.copy()
    h, w = img.shape[:2]
    
    # The bbox coords in JSONL are in 1000x1000 scale, normalize to image size
    x1, y1, x2, y2 = bbox
    scale_x = w / 1000.0
    scale_y = h / 1000.0
    
    x1 = int(x1 * scale_x)
    y1 = int(y1 * scale_y)
    x2 = int(x2 * scale_x)
    y2 = int(y2 * scale_y)
    
    # Clamp to image bounds
    x1 = max(0, min(w - 1, x1))
    y1 = max(0, min(h - 1, y1))
    x2 = max(0, min(w - 1, x2))
    y2 = max(0, min(h - 1, y2))
    
    # Set color based on is_target
    if color is None:
        if is_target is True:
            color = (0, 255, 0)  # Green for target
        elif is_target is False:
            color = (255, 165, 0)  # Orange for non-target
        else:
            color = (128, 128, 128)  # Gray for unknown
    
    # Draw rectangle
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    
    # Draw label
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    text_thickness = 1
    
    # Add target indicator to label
    display_label = label
    if is_target is True:
        display_label = f"[T] {label}"
    
    # Get text size for background
    (text_w, text_h), baseline = cv2.getTextSize(display_label, font, font_scale, text_thickness)
    
    # Draw text background
    cv2.rectangle(img, (x1, y1 - text_h - 4), (x1 + text_w + 4, y1), color, -1)
    
    # Draw text
    cv2.putText(img, display_label, (x1 + 2, y1 - 2), font, font_scale, (255, 255, 255), text_thickness)
    
    return img


def visualize_episode_with_bbox(
    episode_data: dict,
    annotations: dict,
    uuid: str,
    output_path: str | None = None,
) -> None:
    """Visualize a single episode with bounding box annotations.
    
    Args:
        episode_data: Episode data dictionary
        annotations: All bbox annotations
        uuid: UUID for this episode
        output_path: Optional path to save visualization
    """
    if uuid not in annotations:
        print(f"No annotations found for UUID: {uuid}")
        return
    
    annot = annotations[uuid]
    labels = annot.get("labels", [])
    task = annot.get("task", "")
    
    print(f"\nEpisode: {uuid}")
    print(f"Task: {task}")
    print(f"Total frames: {annot.get('total_frames', len(episode_data['images']))}")
    print(f"Annotated frames: {len(labels)}")
    
    images = episode_data["images"]
    if len(images) == 0:
        print("No images in episode")
        return
    
    # Create figure with subplots for annotated frames
    n_annotated = len(labels)
    if n_annotated == 0:
        print("No labeled frames")
        return
    
    cols = min(3, n_annotated)
    rows = (n_annotated + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 6 * rows))
    if n_annotated == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    fig.suptitle(f"Task: {task}\nUUID: {uuid}", fontsize=12)
    
    for i, label_data in enumerate(labels):
        frame_idx = label_data["frame"]
        target_object = label_data.get("target_object", "")
        all_objects = label_data.get("all_objects", [])
        
        if frame_idx >= len(images):
            print(f"Frame {frame_idx} out of range (max: {len(images) - 1})")
            continue
        
        img = images[frame_idx].copy()
        
        # Draw all bboxes
        for obj in all_objects:
            obj_label = obj.get("label", "")
            bbox = obj.get("bbox", [])
            is_target = obj.get("is_target", None)
            
            if len(bbox) == 4:
                img = draw_bbox_on_image(img, bbox, obj_label, is_target)
        
        axes[i].imshow(img)
        axes[i].set_title(f"Frame {frame_idx}\nTarget: {target_object}")
        axes[i].axis("off")
    
    # Hide unused subplots
    for i in range(n_annotated, len(axes)):
        axes[i].axis("off")
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def create_interactive_viewer(
    episodes: list,
    filepath_to_annot: dict,
    data_dir: str,
):
    """Create an interactive matplotlib viewer for the dataset with bboxes.
    
    Args:
        episodes: List of episode data
        filepath_to_annot: Bbox annotations indexed by file_path
        data_dir: Dataset directory path
    """
    current_episode_idx = [0]
    current_frame = [0]
    
    # Load first episode
    n_episodes = len(episodes)
    episode_data = [extract_episode_data(episodes[0], 0)]
    
    n_frames = len(episode_data[0]['images'])
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f'Bridge Dataset with BBox Annotations\n{data_dir}', fontsize=12)
    
    # Image axes
    ax_img = fig.add_subplot(2, 3, 1)
    ax_wrist = fig.add_subplot(2, 3, 2)
    ax_info = fig.add_subplot(2, 3, 3)
    ax_info.axis('off')
    
    # Annotation info panel
    ax_bbox_info = fig.add_subplot(2, 3, (4, 6))
    ax_bbox_info.axis('off')
    
    # Initial display
    if len(episode_data[0]['images']) > 0:
        im_img = ax_img.imshow(episode_data[0]['images'][0])
    else:
        im_img = ax_img.imshow(np.zeros((256, 256, 3), dtype=np.uint8))
    ax_img.set_title('Primary Camera')
    ax_img.axis('off')
    
    if len(episode_data[0]['wrist_images']) > 0:
        im_wrist = ax_wrist.imshow(episode_data[0]['wrist_images'][0])
    else:
        im_wrist = ax_wrist.imshow(np.zeros((256, 256, 3), dtype=np.uint8))
    ax_wrist.set_title('Wrist Camera')
    ax_wrist.axis('off')
    
    # Info text
    info_text = ax_info.text(0.05, 0.95, '', transform=ax_info.transAxes,
                             fontsize=10, verticalalignment='top',
                             fontfamily='monospace')
    
    bbox_info_text = ax_bbox_info.text(0.05, 0.95, '', transform=ax_bbox_info.transAxes,
                                        fontsize=9, verticalalignment='top',
                                        fontfamily='monospace')
    
    def get_current_annotation():
        """Get annotation for current episode using file_path matching."""
        data = episode_data[0]
        file_path = data.get('file_path', '')
        return filepath_to_annot.get(file_path)
    
    def get_frame_annotation(annot: dict | None, frame_idx: int) -> dict | None:
        if annot is None:
            return None
        for label in annot.get("labels", []):
            if label.get("frame") == frame_idx:
                return label
        return None
    
    def update_display():
        data = episode_data[0]
        frame = current_frame[0]
        annot = get_current_annotation()
        file_path = data.get('file_path', '')
        
        # Get image for this frame
        if frame < len(data['images']):
            img = data['images'][frame].copy()
            
            # Draw bboxes if we have annotations for this frame
            if annot:
                frame_annot = get_frame_annotation(annot, frame)
                if frame_annot:
                    for obj in frame_annot.get("all_objects", []):
                        bbox = obj.get("bbox", [])
                        label = obj.get("label", "")
                        is_target = obj.get("is_target", None)
                        if len(bbox) == 4:
                            img = draw_bbox_on_image(img, bbox, label, is_target)
            
            im_img.set_data(img)
        
        if len(data['wrist_images']) > 0 and frame < len(data['wrist_images']):
            im_wrist.set_data(data['wrist_images'][frame])
        
        # Update info text
        matched_str = "MATCHED" if annot else "NO MATCH"
        info_str = f"""Episode: {current_episode_idx[0] + 1}/{n_episodes}
Frame: {frame}/{len(data['images']) - 1}

Task: {data['language_instruction'][:50]}...

file_path: ...{file_path[-40:] if len(file_path) > 40 else file_path}
Status: {matched_str}"""
        info_text.set_text(info_str)
        
        # Update bbox info
        bbox_str = ""
        if annot:
            frame_annot = get_frame_annotation(annot, frame)
            
            bbox_str = f"Annotations for episode:\n"
            bbox_str += f"  Task: {annot.get('task', 'N/A')}\n"
            bbox_str += f"  Total frames: {annot.get('total_frames', 'N/A')}\n"
            bbox_str += f"  Labeled frames: {len(annot.get('labels', []))}\n\n"
            
            if frame_annot:
                bbox_str += f"Frame {frame} annotations:\n"
                bbox_str += f"  Target object: {frame_annot.get('target_object', 'N/A')}\n"
                bbox_str += f"  Objects:\n"
                for obj in frame_annot.get("all_objects", []):
                    label = obj.get("label", "")
                    bbox = obj.get("bbox", [])
                    is_target = obj.get("is_target", None)
                    target_marker = "[T]" if is_target else "[_]"
                    bbox_str += f"    {target_marker} {label}: {bbox}\n"
            else:
                bbox_str += f"Frame {frame}: No annotations"
        else:
            bbox_str = f"No annotations found for this episode\n\nfile_path: {file_path}"
        
        bbox_info_text.set_text(bbox_str)
        fig.canvas.draw_idle()
    
    def update_frame(val):
        frame = int(val)
        current_frame[0] = frame
        update_display()
    
    def load_episode(idx):
        current_episode_idx[0] = idx
        episode_data[0] = extract_episode_data(episodes[idx], idx)
        
        # Update slider range
        n_frames = max(1, len(episode_data[0]['images']))
        frame_slider.valmax = n_frames - 1
        frame_slider.ax.set_xlim(0, n_frames - 1)
        frame_slider.set_val(0)
        current_frame[0] = 0
        
        # Update episode slider
        episode_slider.set_val(idx)
        
        update_display()
    
    # Add frame slider
    ax_frame_slider = fig.add_axes([0.15, 0.02, 0.5, 0.02])
    frame_slider = Slider(ax_frame_slider, 'Frame', 0, max(1, n_frames - 1), valinit=0, valstep=1)
    frame_slider.on_changed(update_frame)
    
    # Add episode slider
    ax_episode_slider = fig.add_axes([0.15, 0.05, 0.5, 0.02])
    episode_slider = Slider(ax_episode_slider, 'Episode', 0, max(1, n_episodes - 1), valinit=0, valstep=1)
    
    def on_episode_change(val):
        idx = int(val)
        if idx != current_episode_idx[0]:
            load_episode(idx)
    
    episode_slider.on_changed(on_episode_change)
    
    # Add next/prev annotated frame buttons
    ax_prev = fig.add_axes([0.7, 0.02, 0.08, 0.03])
    ax_next = fig.add_axes([0.8, 0.02, 0.08, 0.03])
    prev_button = Button(ax_prev, 'Prev Ann')
    next_button = Button(ax_next, 'Next Ann')
    
    def goto_prev_annotated(event):
        annot = get_current_annotation()
        if annot is None:
            return
        labels = annot.get("labels", [])
        annotated_frames = sorted([l["frame"] for l in labels])
        current = current_frame[0]
        
        for f in reversed(annotated_frames):
            if f < current:
                frame_slider.set_val(f)
                return
    
    def goto_next_annotated(event):
        annot = get_current_annotation()
        if annot is None:
            return
        labels = annot.get("labels", [])
        annotated_frames = sorted([l["frame"] for l in labels])
        current = current_frame[0]
        
        for f in annotated_frames:
            if f > current:
                frame_slider.set_val(f)
                return
    
    prev_button.on_clicked(goto_prev_annotated)
    next_button.on_clicked(goto_next_annotated)
    
    # Initial display
    update_display()
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)
    plt.show()


def save_annotated_frames(
    episodes: list,
    filepath_to_annot: dict,
    output_dir: str,
    max_episodes: int | None = None,
):
    """Save annotated frames as images.
    
    Args:
        episodes: List of episodes
        filepath_to_annot: Bbox annotations indexed by file_path
        output_dir: Directory to save images
        max_episodes: Maximum episodes to process
    """
    os.makedirs(output_dir, exist_ok=True)
    
    processed = 0
    matched = 0
    
    for ep_idx, episode in enumerate(episodes):
        if max_episodes and processed >= max_episodes:
            break
        
        episode_data = extract_episode_data(episode, ep_idx)
        file_path = episode_data.get('file_path', '')
        
        # Match by file_path
        annot = filepath_to_annot.get(file_path)
        if annot is None:
            continue
        
        matched += 1
        
        if len(episode_data['images']) == 0:
            continue
        
        labels = annot.get("labels", [])
        task = annot.get("task", "unknown")
        
        for label_data in labels:
            frame_idx = label_data["frame"]
            if frame_idx >= len(episode_data['images']):
                continue
            
            img = episode_data['images'][frame_idx].copy()
            
            # Draw all bboxes
            for obj in label_data.get("all_objects", []):
                bbox = obj.get("bbox", [])
                obj_label = obj.get("label", "")
                is_target = obj.get("is_target", None)
                if len(bbox) == 4:
                    img = draw_bbox_on_image(img, bbox, obj_label, is_target)
            
            # Save image
            filename = f"ep{ep_idx:04d}_frame{frame_idx:03d}.png"
            cv2.imwrite(os.path.join(output_dir, filename), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        
        processed += 1
        if processed % 10 == 0:
            print(f"Processed {processed} episodes (matched {matched})...")
    
    print(f"Saved annotated frames to {output_dir} ({matched} episodes matched)")


def create_grid_image(images: list[np.ndarray], cols: int = 4) -> np.ndarray:
    """Create a grid image from a list of images.
    
    Args:
        images: List of images (H, W, 3)
        cols: Number of columns in the grid
        
    Returns:
        Grid image as numpy array
    """
    if not images:
        return np.zeros((256, 256, 3), dtype=np.uint8)
    
    n = len(images)
    rows = (n + cols - 1) // cols
    
    # Get consistent size from first image
    h, w = images[0].shape[:2]
    
    # Create empty grid
    grid = np.zeros((rows * h, cols * w, 3), dtype=np.uint8)
    
    for i, img in enumerate(images):
        row = i // cols
        col = i % cols
        # Resize if needed
        if img.shape[:2] != (h, w):
            img = cv2.resize(img, (w, h))
        grid[row * h:(row + 1) * h, col * w:(col + 1) * w] = img
    
    return grid


def log_to_wandb(
    episodes: list,
    filepath_to_annot: dict,
    max_episodes: int = 50,
    images_per_log: int = 16,
):
    """Log annotated frames to wandb.
    
    Args:
        episodes: List of episodes
        filepath_to_annot: Bbox annotations indexed by file_path
        max_episodes: Maximum episodes to process
        images_per_log: Number of images per wandb log entry
    """
    if wandb is None:
        print("wandb not installed. Install with: pip install wandb")
        return
    
    # Collect all annotated images
    all_images = []
    all_captions = []
    
    processed = 0
    matched = 0
    
    for ep_idx, episode in enumerate(episodes):
        if processed >= max_episodes:
            break
        
        episode_data = extract_episode_data(episode, ep_idx)
        file_path = episode_data.get('file_path', '')
        
        # Match by file_path
        annot = filepath_to_annot.get(file_path)
        if annot is None:
            continue
        
        matched += 1
        
        if len(episode_data['images']) == 0:
            continue
        
        labels = annot.get("labels", [])
        task = annot.get("task", "")
        
        for label_data in labels:
            frame_idx = label_data["frame"]
            if frame_idx >= len(episode_data['images']):
                continue
            
            img = episode_data['images'][frame_idx].copy()
            target_object = label_data.get("target_object", "")
            
            # Draw all bboxes
            for obj in label_data.get("all_objects", []):
                bbox = obj.get("bbox", [])
                obj_label = obj.get("label", "")
                is_target = obj.get("is_target", None)
                if len(bbox) == 4:
                    img = draw_bbox_on_image(img, bbox, obj_label, is_target)
            
            all_images.append(img)
            caption = f"Ep{ep_idx} F{frame_idx}: {task[:40]}... Target: {target_object}"
            all_captions.append(caption)
        
        processed += 1
        if processed % 10 == 0:
            print(f"Processed {processed} episodes...")
    
    print(f"Collected {len(all_images)} annotated frames")
    
    # Log images in batches
    batch_idx = 0
    for i in range(0, len(all_images), images_per_log):
        batch_images = all_images[i:i + images_per_log]
        batch_captions = all_captions[i:i + images_per_log]
        
        # Create grid for this batch
        grid = create_grid_image(batch_images, cols=4)
        
        # Log individual images with captions
        wandb_images = [
            wandb.Image(img, caption=cap) 
            for img, cap in zip(batch_images, batch_captions)
        ]
        
        # Log to wandb
        wandb.log({
            "bbox_vis/grid": wandb.Image(grid, caption=f"Batch {batch_idx}"),
            "bbox_vis/images": wandb_images,
            "batch_idx": batch_idx,
        })
        
        batch_idx += 1
        print(f"Logged batch {batch_idx} ({len(batch_images)} images)")
    
    print(f"Finished logging {len(all_images)} images in {batch_idx} batches")


def log_episodes_to_wandb(
    episodes: list,
    filepath_to_annot: dict,
    max_episodes: int = 50,
):
    """Log episode summaries to wandb (one log per episode with all annotated frames).
    
    Args:
        episodes: List of episodes
        filepath_to_annot: Bbox annotations indexed by file_path
        max_episodes: Maximum episodes to process
    """
    if wandb is None:
        print("wandb not installed. Install with: pip install wandb")
        return
    
    processed = 0
    matched = 0
    
    for ep_idx, episode in enumerate(episodes):
        if processed >= max_episodes:
            break
        
        episode_data = extract_episode_data(episode, ep_idx)
        file_path = episode_data.get('file_path', '')
        
        # Match by file_path
        annot = filepath_to_annot.get(file_path)
        if annot is None:
            continue
        
        matched += 1
        
        if len(episode_data['images']) == 0:
            continue
        
        labels = annot.get("labels", [])
        task = annot.get("task", "")
        
        if not labels:
            continue
        
        # Collect annotated frames for this episode
        episode_images = []
        episode_captions = []
        
        for label_data in labels:
            frame_idx = label_data["frame"]
            if frame_idx >= len(episode_data['images']):
                continue
            
            img = episode_data['images'][frame_idx].copy()
            target_object = label_data.get("target_object", "")
            
            # Draw all bboxes
            for obj in label_data.get("all_objects", []):
                bbox = obj.get("bbox", [])
                obj_label = obj.get("label", "")
                is_target = obj.get("is_target", None)
                if len(bbox) == 4:
                    img = draw_bbox_on_image(img, bbox, obj_label, is_target)
            
            episode_images.append(img)
            n_objects = len(label_data.get("all_objects", []))
            episode_captions.append(f"F{frame_idx}: Target={target_object} ({n_objects} objects)")
        
        if not episode_images:
            continue
        
        # Create grid for this episode
        grid = create_grid_image(episode_images, cols=min(3, len(episode_images)))
        
        # Log individual images with captions
        wandb_images = [
            wandb.Image(img, caption=cap) 
            for img, cap in zip(episode_images, episode_captions)
        ]
        
        # Log to wandb
        wandb.log({
            "episode/grid": wandb.Image(grid, caption=f"Episode {ep_idx}: {task}"),
            "episode/frames": wandb_images,
            "episode/task": task,
            "episode/num_frames": len(episode_images),
            "episode/idx": ep_idx,
        }, step=processed)
        
        processed += 1
        print(f"[{processed}/{max_episodes}] Logged episode {ep_idx}: {task[:50]}...")
    
    print(f"Finished logging {processed} episodes")


def main():
    parser = argparse.ArgumentParser(description='Visualize Bridge dataset with bbox annotations')
    parser.add_argument('--data-dir', type=str, 
                        default='gs://pi0-cot/OXE/bridge_v2_oxe/1.0.0',
                        help='Path to bridge_v2_oxe dataset')
    parser.add_argument('--annotations', type=str,
                        default='bridge-bbox-annotations_bridge_bbox_labels_0_2659.jsonl',
                        help='Path to JSONL file with bbox annotations')
    parser.add_argument('--split', type=str, default='train',
                        help='Dataset split (default: train)')
    parser.add_argument('--max-episodes', type=int, default=100,
                        help='Maximum number of episodes to load (default: 100)')
    parser.add_argument('--save-dir', type=str, default=None,
                        help='Directory to save annotated frames')
    parser.add_argument('--episode-idx', type=int, default=None,
                        help='Visualize a specific episode index')
    
    # Wandb arguments
    parser.add_argument('--wandb', action='store_true',
                        help='Log visualizations to wandb')
    parser.add_argument('--wandb-project', type=str, default='bridge-bbox-vis',
                        help='Wandb project name (default: bridge-bbox-vis)')
    parser.add_argument('--wandb-entity', type=str, default=None,
                        help='Wandb entity/team name')
    parser.add_argument('--wandb-run-name', type=str, default=None,
                        help='Wandb run name (default: auto-generated)')
    parser.add_argument('--wandb-mode', type=str, default='online',
                        choices=['online', 'offline', 'disabled'],
                        help='Wandb mode (default: online)')
    parser.add_argument('--per-episode', action='store_true',
                        help='Log one entry per episode (instead of batched)')
    
    args = parser.parse_args()
    
    # Get annotations path
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    
    if os.path.isabs(args.annotations):
        annotations_path = args.annotations
    else:
        annotations_path = project_root / args.annotations
    
    # Load annotations (returns both uuid-indexed and filepath-indexed dicts)
    annotations, filepath_to_annot = load_bbox_annotations(str(annotations_path))
    
    # Initialize wandb if requested
    if args.wandb:
        if wandb is None:
            print("Error: wandb not installed. Install with: pip install wandb")
            return
        
        # Use non-interactive backend for headless environments
        matplotlib.use('Agg')
        
        run_name = args.wandb_run_name or f"bridge-bbox-{args.max_episodes}ep"
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            mode=args.wandb_mode,
            config={
                "data_dir": args.data_dir,
                "annotations": str(annotations_path),
                "max_episodes": args.max_episodes,
                "split": args.split,
                "num_annotations": len(annotations),
                "num_filepath_entries": len(filepath_to_annot),
            },
        )
        print(f"Initialized wandb run: {wandb.run.name}")
        print(f"View at: {wandb.run.get_url()}")
    
    # Load dataset
    episodes = load_bridge_dataset(args.data_dir, args.split, args.max_episodes)
    
    if len(episodes) == 0:
        print("No episodes loaded!")
        if args.wandb and wandb.run:
            wandb.finish()
        return
    
    if args.wandb:
        # Log to wandb (use filepath_to_annot for matching)
        if args.per_episode:
            log_episodes_to_wandb(episodes, filepath_to_annot, args.max_episodes)
        else:
            log_to_wandb(episodes, filepath_to_annot, args.max_episodes)
        wandb.finish()
        print("Wandb logging complete!")
    elif args.save_dir:
        save_annotated_frames(episodes, filepath_to_annot, args.save_dir, args.max_episodes)
    elif args.episode_idx is not None:
        if args.episode_idx >= len(episodes):
            print(f"Episode index {args.episode_idx} out of range (max: {len(episodes) - 1})")
            return
        
        # Extract episode and match by file_path
        episode_data = extract_episode_data(episodes[args.episode_idx], args.episode_idx)
        file_path = episode_data.get('file_path', '')
        annot = filepath_to_annot.get(file_path)
        
        if annot:
            uuid = annot.get('uuid', f'matched_by_filepath:{file_path}')
            visualize_episode_with_bbox(episode_data, annotations, uuid)
        else:
            print(f"No annotations found for episode {args.episode_idx}")
            print(f"  file_path: {file_path}")
    else:
        create_interactive_viewer(episodes, filepath_to_annot, args.data_dir)


if __name__ == '__main__':
    main()
