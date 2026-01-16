"""Bounding box visualization for BridgeV2 dataset.

Loads TFRecords directly and draws pre-computed bounding boxes from JSONL annotations.

Usage:
    python visualize_bbox_fast_bridge.py \
        --bbox_dir PATH TO FOLDER CONTAINING JSONL FILES \
        --output_dir OUTPUT_DIR \
        --num_samples 100
"""

import argparse
import json
from pathlib import Path
from typing import Dict
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tensorflow as tf
from io import BytesIO


def load_bbox_annotations(bbox_dir: str) -> Dict[str, dict]:
    """Load all bounding box annotations from JSONL files."""
    annotations = {}
    bbox_path = Path(bbox_dir)

    for jsonl_file in sorted(bbox_path.glob("*.jsonl")):
        print(f"Loading {jsonl_file.name}...")
        with open(jsonl_file, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line.strip())
                annotations[data['uuid']] = data

    print(f"Loaded {len(annotations)} annotated episodes\n")
    return annotations


def load_episode_from_tfrecord(tfrecord_path: str, episode_idx: int = 0):
    """Load a specific episode from a TFRecord file for BridgeV2 dataset."""
    dataset = tf.data.TFRecordDataset(tfrecord_path)

    for idx, raw_record in enumerate(dataset):
        if idx != episode_idx:
            continue

        # Parse the example
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        features = example.features.feature

        # Extract task (language instruction)
        if 'steps/language_instruction' in features:
            task_bytes = features['steps/language_instruction'].bytes_list.value[0]
            task = task_bytes.decode('utf-8')
        else:
            task = "Unknown task"

        # image_0 is the primary camera
        if 'steps/observation/image_0' in features:
            image_0 = features['steps/observation/image_0'].bytes_list.value
            num_steps = len(image_0)
        else:
            print(f"Warning: Could not find image_0 in features")
            return None

        return {
            'task': task,
            'num_steps': num_steps,
            'image_0': image_0,
        }

    return None


def draw_bbox_on_image(image, bbox_data: dict, font_size: int = 14) -> Image.Image:
    """Draw bounding boxes and labels on an image.

    Matches the format from bridge_bbox_subtask_gcs.py visualize_debug_frame()
    """
    # Ensure we have a PIL Image
    if isinstance(image, np.ndarray):
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        primary_image = Image.fromarray(image)
    else:
        primary_image = image

    task = bbox_data.get('task', 'Unknown task')
    target_object = bbox_data.get('target_object', 'Unknown target')
    all_objects = bbox_data.get('all_objects', [])

    # Prepare text
    task_text = f"Task: {task[:50]}..." if len(task) > 50 else f"Task: {task}"
    target_text = f"Target: {target_object if target_object else 'NONE'}"
    count_text = f"Objects detected: {len(all_objects)}"

    # Try to use a truetype font
    try:
        font_small = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", font_size)
        font_tiny = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 11)
    except:
        font_small = ImageFont.load_default()
        font_tiny = ImageFont.load_default()

    # Define header height
    header_height = 70

    # Create new image with header space
    new_img = Image.new('RGB', (primary_image.width, primary_image.height + header_height), color='black')
    new_img.paste(primary_image, (0, header_height))

    # Draw text in header
    draw = ImageDraw.Draw(new_img)
    draw.text((10, 5), task_text, fill='lime', font=font_small)
    draw.text((10, 25), target_text, fill='cyan', font=font_small)
    draw.text((10, 45), count_text, fill='yellow', font=font_small)

    # Draw bounding boxes
    img_width, img_height = primary_image.size

    for obj in all_objects:
        label = obj['label']
        bbox = obj['bbox']  # [y_min, x_min, y_max, x_max] - normalized to 0-1000
        is_target = obj.get('is_target', False)

        y_min, x_min, y_max, x_max = bbox

        # Convert normalized coordinates (0-1000) to absolute pixel coordinates
        abs_x_min = int(x_min * img_width / 1000)
        abs_y_min = int(y_min * img_height / 1000) + header_height
        abs_x_max = int(x_max * img_width / 1000)
        abs_y_max = int(y_max * img_height / 1000) + header_height

        # Choose color
        box_color = 'red' if is_target else 'orange'
        fill_color = 'red' if is_target else 'darkorange'

        # Draw bounding box
        draw.rectangle([(abs_x_min, abs_y_min), (abs_x_max, abs_y_max)],
                      outline=box_color, width=3 if is_target else 2)

        # Draw label with background
        label_text = label[:25]
        bbox_top = max(header_height, abs_y_min - 20)

        # Estimate text width
        text_width = len(label_text) * 7
        draw.rectangle([(abs_x_min, bbox_top), (abs_x_min + text_width + 10, bbox_top + 18)],
                      fill=fill_color)
        draw.text((abs_x_min + 3, bbox_top + 2), label_text, fill='white', font=font_tiny)

    return new_img


def main():
    parser = argparse.ArgumentParser(description="Fast bbox visualization for BridgeV2")
    parser.add_argument("--bbox_dir", type=str, default="/n/fs/task-ssl/gc-jepa/src/gcjepa/data/bridge-bbox-annotations")
    parser.add_argument("--output_dir", type=str, default="./test_output/bridge_bbox_visualization")
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--frames_per_episode", type=int, default=4)
    args = parser.parse_args()

    print(f"\n{'='*80}")
    print("BridgeV2 Bounding Box Visualization (Fast)")
    print(f"{'='*80}\n")

    # Load annotations
    annotations = load_bbox_annotations(args.bbox_dir)

    # Create output directory
    output_path = Path(args.output_dir)
    output_path.mkdir(exist_ok=True, parents=True)

    # Process first N annotated episodes
    processed = 0
    for uuid, annotation in annotations.items():
        if processed >= args.num_samples:
            break

        # Parse UUID: gs://path/to/file.tfrecord-XXXXX-of-YYYYY::episode_N
        tfrecord_path, episode_part = uuid.split("::")
        episode_idx = int(episode_part.split("_")[1])

        print(f"{'='*80}")
        print(f"Episode {processed + 1}/{args.num_samples}")
        print(f"TFRecord: {Path(tfrecord_path).name}")
        print(f"Local index: {episode_idx}")
        print(f"Task: {annotation['task']}")
        print(f"Frames: {annotation['total_frames']}")
        print(f"Annotated keyframes: {len(annotation['labels'])}")

        # Load episode
        episode = load_episode_from_tfrecord(tfrecord_path, episode_idx)
        if episode is None or 'image_0' not in episode:
            print("  Warning: Could not load episode")
            continue

        # Process each labeled frame
        for label_data in annotation['labels'][:args.frames_per_episode]:
            frame_num = label_data['frame']

            if frame_num >= episode['num_steps']:
                print(f"  Warning: Frame {frame_num} out of range")
                continue

            # Get image - decode to PIL directly
            img_bytes = episode['image_0'][frame_num]
            image = Image.open(BytesIO(img_bytes))

            # Draw bboxes
            annotated_image = draw_bbox_on_image(
                image,
                {**label_data, 'task': annotation['task']},
                font_size=14
            )

            # Save
            filename = f"ep{processed:04d}_frame{frame_num:04d}_image0.jpg"
            annotated_image.save(output_path / filename)
            print(f"  Saved: {filename}")

        processed += 1
        print()

    print(f"{'='*80}")
    print(f"Done! Saved {processed} episodes to {output_path}")


if __name__ == "__main__":
    main()