"""Common utilities for bounding box VQA datasets.

This module provides shared functionality for all bounding box VQA datasets including:
- Shared prompt templates for bbox detection tasks
- Shared prompt templates for direction classification tasks
- Common bbox-to-text conversion functions
- Helper functions for sampling prompts and computing directions
"""

import tensorflow as tf

# =============================================================================
# SHARED PROMPT TEMPLATES
# =============================================================================

# General bounding box prompts (for LVIS, PACO, OXE datasets)
BBOX_PROMPT_PARTS: list[tuple[str, str]] = [
    ("Show me where the ", " is in the image using a bounding box."),
    ("Draw a bounding box around the ", " in the image."),
    ("Please provide a bounding box for the ", " in this image."),
    ("Locate the ", " in the image by drawing a bounding box."),
    ("mark the ", " with a bounding box."),
    ("Identify the ", " in the image by bounding it."),
    ("Find the ", " and draw a bounding box around it."),
    ("Highlight the ", " with a bounding box."),
    ("Can you draw a bounding box around the ", "?"),
    ("Where is the ", " in the image? Show it with a bounding box."),
    ("Indicate the ", " by marking a bounding box."),
    ("If there is a ", " in the image, draw a bounding box around it."),
    ("If any ", " is present, show one bounding box."),
    ("Show me one ", " in the image by drawing a bounding box."),
    ("Please locate the ", " using a bounding box."),
    ("Detect the ", " and provide its bounding box."),
    ("Find a ", " in the picture and draw a bounding box around it."),
    ("Bounding box task: draw a box around the ", "."),
    ("Object: ", ". Instruction: Draw a bounding box around the object."),
    ("Look for the ", " and mark it with a bounding box."),
    ("Help me find the ", " by drawing a bounding box around it."),
    ("Show me ", " using a bounding box."),
    ("For ", " in the image, draw a bounding box."),
    ("Indicate ", " with a bounding box."),
    ("Please show the region containing the ", " using a bounding box."),
    ("point out the ", " by drawing a bounding box."),
    ("Locate a ", " and provide bounding box."),
    ("Draw a bounding box around all the ", "."),
    ("Find and outline the ", " with a bounding box."),
    ("Mark ", " using bounding box."),
    ("Pick up the ", ", predict where it is in the image."),
    ("Move to the ", ", predict where it is in the image."),
]

# Robot-specific bounding box prompts (for DROID wrist camera)
ROBOT_BBOX_PROMPT_PARTS: list[tuple[str, str]] = [
    ("Pick up the ", ", predict where it is relative to the robot."),
    ("Move to the ", ", predict where it is relative to the robot."),
    ("Move near to the ", ", predict where it is relative to the robot."),
    ("Pick up the ", ", predict where it is in the end-effector frame."),
    ("Move to the ", ", predict where it is in the end-effector frame."),
    ("Move near to the ", ", predict where it is in the end-effector frame."),
    ("Pick up the ", ", predict where it is in the robot base frame."),
    ("Move to the ", ", predict where it is in the robot base frame."),
    ("Move near to the ", ", predict where it is in the robot base frame."),
]

# Direction classification prompts (for directional VQA tasks)
DIRECTION_PROMPT_PARTS: list[tuple[str, str]] = [
    ("From the image center, which direction is the ", " located?"),
    ("Relative to the center point of the image, where is the ", "?"),
    ("If you stand at the center of the image, which way do you go to reach the ", "?"),
    ("Looking from the center of the frame, in what direction is the ", " situated?"),
    ("Which direction from the center is the ", " in this image?"),
]


# =============================================================================
# BBOX TO TEXT CONVERSION
# =============================================================================

def bbox_to_loc_tokens(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    num_bins: int = 1024,
) -> str:
    """Convert normalized bbox coordinates to PaLiGemma loc token string.

    Args:
        x_min: Normalized x coordinate of top-left corner (0-1)
        y_min: Normalized y coordinate of top-left corner (0-1)
        x_max: Normalized x coordinate of bottom-right corner (0-1)
        y_max: Normalized y coordinate of bottom-right corner (0-1)
        num_bins: Number of bins for quantization (default 1024)

    Returns:
        Formatted string in PaLiGemma format: "<locYMIN><locXMIN><locYMAX><locXMAX>"
    """
    N = num_bins
    y_min_idx = int(round(y_min * (N - 1)))
    x_min_idx = int(round(x_min * (N - 1)))
    y_max_idx = int(round(y_max * (N - 1)))
    x_max_idx = int(round(x_max * (N - 1)))

    return f"<loc{y_min_idx:04d}><loc{x_min_idx:04d}><loc{y_max_idx:04d}><loc{x_max_idx:04d}>"


def bbox_to_text_tf(bbox: tf.Tensor, num_bins: int = 1024) -> tf.Tensor:
    """Convert bbox tensor to PaLiGemma loc token string (TensorFlow version).

    Args:
        bbox: Tensor of shape [2, 2] with normalized coordinates
              [[x_min, y_min], [x_max, y_max]] where coordinates are in range [0, 1].
        num_bins: Number of bins for quantization (default 1024)

    Returns:
        tf.string tensor with formatted loc tokens: "<locYMIN><locXMIN><locYMAX><locXMAX>"
    """
    # Extract corner coordinates (already normalized to 0-1)
    top_left = bbox[0]
    bottom_right = bbox[1]

    x_min, y_min = top_left[0], top_left[1]
    x_max, y_max = bottom_right[0], bottom_right[1]

    return bbox_coords_to_text_tf(x_min, y_min, x_max, y_max, num_bins)


def bbox_coords_to_text_tf(
    x_min: tf.Tensor,
    y_min: tf.Tensor,
    x_max: tf.Tensor,
    y_max: tf.Tensor,
    num_bins: int = 1024,
) -> tf.Tensor:
    """Convert bbox coordinate tensors to PaLiGemma loc token string.

    Args:
        x_min: Normalized x coordinate of top-left corner (0-1)
        y_min: Normalized y coordinate of top-left corner (0-1)
        x_max: Normalized x coordinate of bottom-right corner (0-1)
        y_max: Normalized y coordinate of bottom-right corner (0-1)
        num_bins: Number of bins for quantization (default 1024)

    Returns:
        tf.string tensor with formatted loc tokens
    """
    N = num_bins
    y_min_idx = tf.cast(tf.round(y_min * (N - 1)), tf.int32)
    x_min_idx = tf.cast(tf.round(x_min * (N - 1)), tf.int32)
    y_max_idx = tf.cast(tf.round(y_max * (N - 1)), tf.int32)
    x_max_idx = tf.cast(tf.round(x_max * (N - 1)), tf.int32)

    # Format as loc tokens in PaLiGemma order: y_min, x_min, y_max, x_max
    y_min_token = tf.strings.join(["<loc", tf.strings.as_string(y_min_idx, width=4, fill="0"), ">"])
    x_min_token = tf.strings.join(["<loc", tf.strings.as_string(x_min_idx, width=4, fill="0"), ">"])
    y_max_token = tf.strings.join(["<loc", tf.strings.as_string(y_max_idx, width=4, fill="0"), ">"])
    x_max_token = tf.strings.join(["<loc", tf.strings.as_string(x_max_idx, width=4, fill="0"), ">"])

    return tf.strings.join([y_min_token, x_min_token, y_max_token, x_max_token])


# =============================================================================
# PROMPT SAMPLING HELPERS
# =============================================================================

def sample_prompt_tf(
    prompt_parts: list[tuple[str, str]],
    category_name: tf.Tensor,
    seed_pair: tuple[int, tf.Tensor],
) -> tf.Tensor:
    """Sample a random prompt template and fill in the category name.

    Args:
        prompt_parts: List of (prefix, suffix) tuples for prompt templates
        category_name: tf.string tensor with the object category/label
        seed_pair: Tuple of (base_seed, hash_value) for stateless random

    Returns:
        tf.string tensor with the complete prompt
    """
    num_prompts = len(prompt_parts)
    prompt_idx = tf.random.stateless_uniform(
        [], seed=seed_pair, minval=0, maxval=num_prompts, dtype=tf.int32
    )

    # Create branches for each prompt template
    def make_prompt_fn(idx):
        prefix, suffix = prompt_parts[idx]

        def fn():
            return tf.strings.join([prefix, category_name, suffix])

        return fn

    prompt_branches = {i: make_prompt_fn(i) for i in range(num_prompts)}
    return tf.switch_case(prompt_idx, branch_fns=prompt_branches)


# =============================================================================
# DIRECTION CLASSIFICATION
# =============================================================================

def direction_from_bbox_tf(bbox: tf.Tensor, slope: float = 2.0) -> tf.Tensor:
    """Map bbox center to direction string relative to image center.

    Args:
        bbox: Tensor of shape [2, 2] with [[x_min, y_min], [x_max, y_max]]
        slope: Slope parameter for direction boundaries (default 2.0)

    Returns:
        tf.string tensor with direction ("forward", "back", "left", "right",
        or compound like "left and forward")
    """
    top_left = bbox[0]
    bottom_right = bbox[1]
    center = (top_left + bottom_right) / 2.0

    x_rel = center[0] - 0.5  # +x is right
    y_rel = 0.5 - center[1]  # invert so +y is up

    k = tf.constant(slope, dtype=tf.float32)
    inv_k = 1.0 / k

    abs_x = tf.abs(x_rel)
    abs_y = tf.abs(y_rel)

    # Primary axis regions using slopes k and 1/k
    is_forward = y_rel > k * abs_x
    is_back = y_rel < -k * abs_x
    is_right = tf.logical_and(tf.logical_not(is_forward), tf.logical_not(is_back))
    is_right = tf.logical_and(is_right, x_rel > inv_k * abs_y)
    is_left = tf.logical_and(tf.logical_not(is_forward), tf.logical_not(is_back))
    is_left = tf.logical_and(is_left, x_rel < -inv_k * abs_y)

    def forward():
        return tf.constant("forward")

    def back():
        return tf.constant("back")

    def right():
        return tf.constant("right")

    def left():
        return tf.constant("left")

    def diagonal():
        base_dir = tf.where(x_rel < 0.0, tf.constant("left"), tf.constant("right"))
        vert_dir = tf.where(y_rel >= 0.0, tf.constant("forward"), tf.constant("back"))
        return tf.strings.join([base_dir, " and ", vert_dir])

    return tf.case(
        [
            (is_forward, forward),
            (is_back, back),
            (is_right, right),
            (is_left, left),
        ],
        default=diagonal,
        exclusive=True,
    )


# =============================================================================
# LETTERBOX BBOX TRANSFORMATION
# =============================================================================

def transform_bbox_for_letterbox(
    x_min: float,
    y_min: float,
    x_max: float,
    y_max: float,
    orig_w: int,
    orig_h: int,
    target_w: int,
    target_h: int,
) -> tuple[float, float, float, float]:
    """Transform bbox coordinates for letterbox (resize with padding) transformation.

    When an image is resized with letterbox (maintaining aspect ratio with padding),
    the bbox coordinates need to be transformed to account for the scaling and padding.

    Args:
        x_min, y_min, x_max, y_max: Original normalized bbox coordinates (0-1)
        orig_w, orig_h: Original image dimensions
        target_w, target_h: Target image dimensions after letterbox

    Returns:
        Tuple of transformed (x_min, y_min, x_max, y_max) coordinates (0-1)
    """
    # Compute letterbox transformation parameters
    ratio = max(orig_w / target_w, orig_h / target_h)
    resized_w = int(orig_w / ratio)
    resized_h = int(orig_h / ratio)
    pad_w = (target_w - resized_w) / 2.0
    pad_h = (target_h - resized_h) / 2.0

    # Transform bbox coordinates
    new_x_min = x_min * (resized_w / target_w) + (pad_w / target_w)
    new_y_min = y_min * (resized_h / target_h) + (pad_h / target_h)
    new_x_max = x_max * (resized_w / target_w) + (pad_w / target_w)
    new_y_max = y_max * (resized_h / target_h) + (pad_h / target_h)

    # Clamp to valid range
    new_x_min = max(0.0, min(1.0, new_x_min))
    new_y_min = max(0.0, min(1.0, new_y_min))
    new_x_max = max(0.0, min(1.0, new_x_max))
    new_y_max = max(0.0, min(1.0, new_y_max))

    return new_x_min, new_y_min, new_x_max, new_y_max


def format_bbox_caption(
    objects: list[dict],
    orig_w: int,
    orig_h: int,
    target_w: int,
    target_h: int,
    apply_letterbox: bool = True,
) -> tuple[str, str]:
    """Format a list of objects with bboxes into prompt labels and caption.

    Args:
        objects: List of dicts with 'label' and 'bbox' ([x_min, y_min, x_max, y_max] normalized)
        orig_w, orig_h: Original image dimensions
        target_w, target_h: Target image dimensions
        apply_letterbox: Whether to apply letterbox transformation to bbox coordinates

    Returns:
        Tuple of (prompt_labels, caption) strings
    """
    if not objects:
        return "", ""

    # Build prompt with all object labels
    labels = [obj["label"] for obj in objects]
    unique_labels = list(dict.fromkeys(labels))  # Preserve order, remove duplicates
    prompt_labels = ", ".join(unique_labels)

    # Build caption with all bboxes
    caption_parts = []
    for obj in objects:
        label = obj["label"]
        bbox = obj["bbox"]  # [x_min, y_min, x_max, y_max] normalized 0-1

        if apply_letterbox:
            x_min, y_min, x_max, y_max = transform_bbox_for_letterbox(
                bbox[0], bbox[1], bbox[2], bbox[3],
                orig_w, orig_h, target_w, target_h
            )
        else:
            x_min, y_min, x_max, y_max = bbox[0], bbox[1], bbox[2], bbox[3]

        loc_str = bbox_to_loc_tokens(x_min, y_min, x_max, y_max)
        caption_parts.append(f"{loc_str} {label}")

    caption = " ; ".join(caption_parts)

    return prompt_labels, caption


# =============================================================================
# FRAME OBJECTS LOOKUP TABLE
# =============================================================================

def build_frame_objects_table(
    bbox_annotations_dir: str,
    key_extractor: callable,
    dataset_name: str = "",
    orig_size: tuple[int, int] = (256, 256),
    target_size: tuple[int, int] = (224, 224),
) -> "tf.lookup.StaticHashTable":
    """Build a lookup table from key--frame_idx to pre-formatted (labels, caption) strings.

    This function reads JSONL annotation files and builds a TensorFlow lookup table
    that maps frame identifiers to pre-formatted prompt labels and bbox captions.
    The bbox coordinates are pre-transformed for letterbox and converted to loc tokens.

    This eliminates the need for tf.py_function during iteration.

    Args:
        bbox_annotations_dir: Directory containing JSONL annotation files
        key_extractor: Function that takes episode_data dict and returns the key string
                      (e.g., uuid or episode_path). Should return None to skip entries.
        dataset_name: Optional dataset name for logging
        orig_size: Original image (width, height) for letterbox transformation
        target_size: Target image (width, height) for letterbox transformation

    Returns:
        tf.lookup.StaticHashTable mapping "key--frame_idx" to "labels\\tcaption" string
        The value is tab-separated: first part is comma-separated labels, second is bbox caption
    """
    import json
    import logging
    import os

    import tensorflow as tf

    log_prefix = f" for {dataset_name}" if dataset_name else ""
    logging.info(f"Building frame objects lookup table{log_prefix}...")

    orig_w, orig_h = orig_size
    target_w, target_h = target_size

    frame_to_formatted = {}

    jsonl_files = tf.io.gfile.glob(os.path.join(bbox_annotations_dir, "*.jsonl"))

    for jsonl_file in jsonl_files:
        with tf.io.gfile.GFile(jsonl_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    episode_data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # Use the provided key extractor to get the lookup key
                episode_key = key_extractor(episode_data)
                if not episode_key:
                    continue

                labels = episode_data.get("labels", [])
                for label_entry in labels:
                    frame_idx = label_entry.get("frame")
                    all_objects = label_entry.get("all_objects", [])

                    if frame_idx is None or not all_objects:
                        continue

                    key = f"{episode_key}--{frame_idx}"

                    objects_list = []
                    for obj in all_objects:
                        obj_label = obj.get("label", "")
                        bbox = obj.get("bbox", [])

                        if not obj_label or len(bbox) < 4:
                            continue

                        # Normalize bbox (bbox values are in 0-1000 range in JSONL)
                        y_min_raw = max(0.0, min(1.0, float(bbox[0]) / 1000.0))
                        x_min_raw = max(0.0, min(1.0, float(bbox[1]) / 1000.0))
                        y_max_raw = max(0.0, min(1.0, float(bbox[2]) / 1000.0))
                        x_max_raw = max(0.0, min(1.0, float(bbox[3]) / 1000.0))

                        objects_list.append({
                            "label": obj_label,
                            "bbox": [x_min_raw, y_min_raw, x_max_raw, y_max_raw],
                        })

                    if objects_list:
                        # Pre-format the caption using shared function
                        prompt_labels, caption = format_bbox_caption(
                            objects=objects_list,
                            orig_w=orig_w,
                            orig_h=orig_h,
                            target_w=target_w,
                            target_h=target_h,
                            apply_letterbox=True,
                        )

                        if prompt_labels and caption:
                            # Store as tab-separated: "labels\tcaption"
                            if key in frame_to_formatted:
                                # Merge with existing (shouldn't happen often)
                                existing = frame_to_formatted[key]
                                existing_labels, existing_caption = existing.split("\t", 1)
                                merged_labels = existing_labels + ", " + prompt_labels
                                merged_caption = existing_caption + " ; " + caption
                                frame_to_formatted[key] = f"{merged_labels}\t{merged_caption}"
                            else:
                                frame_to_formatted[key] = f"{prompt_labels}\t{caption}"

    # Convert to lookup table
    keys = list(frame_to_formatted.keys())
    values = list(frame_to_formatted.values())

    logging.info(f"Built frame objects table with {len(keys)} entries{log_prefix}")

    if not keys:
        # Return table with dummy entry (TF doesn't allow empty tables)
        return tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(
                tf.constant(["__dummy_key__"], dtype=tf.string),
                tf.constant([""], dtype=tf.string),
            ),
            default_value=tf.constant(b"", dtype=tf.string),
        )

    return tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            tf.constant(keys, dtype=tf.string),
            tf.constant(values, dtype=tf.string),
        ),
        default_value=tf.constant(b"", dtype=tf.string),
    )


def droid_key_extractor(episode_data: dict) -> str | None:
    """Extract episode path key from DROID JSONL entry.

    DROID uses file_path in episode_metadata to identify episodes.
    The path is processed to extract the relative episode path.
    """
    import re

    file_path = episode_data.get("episode_metadata", {}).get("file_path", "")
    if not file_path:
        return None

    # Extract episode path using the same logic as extract_episode_path_from_file_path
    # Remove prefix up to r2d2-data or r2d2-data-full
    rel = re.sub(r"^.*r2d2-data(?:-full)?/", "", file_path)
    # Remove /trajectory... suffix
    episode_path = re.sub(r"/trajectory.*$", "", rel)

    return episode_path if episode_path else None


def oxe_key_extractor(episode_data: dict) -> str | None:
    """Extract UUID key from OXE JSONL entry.

    OXE datasets use a 'uuid' field directly in the JSONL.
    """
    return episode_data.get("uuid", "") or None


def build_annotated_keys_set(
    bbox_annotations_dir: str,
    key_extractor: callable,
) -> set[str]:
    """Build a set of keys (uuids or episode_paths) that have bbox annotations.

    This is used for trajectory-level filtering to skip entire trajectories
    that have no annotated frames, which significantly speeds up iteration.

    Args:
        bbox_annotations_dir: Directory containing JSONL annotation files
        key_extractor: Function that takes episode_data dict and returns the key string

    Returns:
        Set of keys that have at least one annotated frame
    """
    import json
    import os

    import tensorflow as tf

    annotated_keys = set()

    jsonl_files = tf.io.gfile.glob(os.path.join(bbox_annotations_dir, "*.jsonl"))

    for jsonl_file in jsonl_files:
        with tf.io.gfile.GFile(jsonl_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    episode_data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                episode_key = key_extractor(episode_data)
                if episode_key:
                    annotated_keys.add(episode_key)

    return annotated_keys
