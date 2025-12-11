"""Image processing utilities for CoT RLDS datasets."""

import tensorflow as tf


def make_decode_images_fn(
    *,
    primary_key: str,
    wrist_key: str | None,
    wrist_right_key: str | None = None,
    resize_to: tuple[int, int] | None = (224, 224),
):
    """Return a frame_map function that decodes encoded image bytes to uint8 tensors.
    Preserves aspect ratio, pads symmetrically, and returns the original dtype semantics
    (uint8 clamped 0-255, float32 clamped to [-1, 1]).
    """

    def _tf_resize_with_pad(image: tf.Tensor, target_h: int, target_w: int) -> tf.Tensor:
        # Compute resized dimensions preserving aspect ratio
        in_h = tf.shape(image)[0]
        in_w = tf.shape(image)[1]
        orig_dtype = image.dtype

        h_f = tf.cast(in_h, tf.float32)
        w_f = tf.cast(in_w, tf.float32)
        th_f = tf.cast(target_h, tf.float32)
        tw_f = tf.cast(target_w, tf.float32)

        ratio = tf.maximum(w_f / tw_f, h_f / th_f)
        resized_h = tf.cast(tf.math.floor(h_f / ratio), tf.int32)
        resized_w = tf.cast(tf.math.floor(w_f / ratio), tf.int32)

        # Resize in float32
        img_f32 = tf.cast(image, tf.float32)
        resized_f32 = tf.image.resize(img_f32, [resized_h, resized_w], method=tf.image.ResizeMethod.BILINEAR)

        # Dtype-specific postprocess (python conditional on static dtype)
        if orig_dtype == tf.uint8:
            resized = tf.cast(tf.clip_by_value(tf.round(resized_f32), 0.0, 255.0), tf.uint8)
            const_val = tf.constant(0, dtype=resized.dtype)
        else:
            resized = tf.clip_by_value(resized_f32, -1.0, 1.0)
            const_val = tf.constant(-1.0, dtype=resized.dtype)

        # Compute symmetric padding
        pad_h_total = target_h - resized_h
        pad_w_total = target_w - resized_w
        pad_h0 = pad_h_total // 2
        pad_h1 = pad_h_total - pad_h0
        pad_w0 = pad_w_total // 2
        pad_w1 = pad_w_total - pad_w0

        padded = tf.pad(resized, [[pad_h0, pad_h1], [pad_w0, pad_w1], [0, 0]], constant_values=const_val)
        return padded

    def _decode_single(img_bytes):
        # If already numeric, cast to uint8 and return
        if img_bytes.dtype != tf.string:
            img = tf.cast(img_bytes, tf.uint8)
        else:
            # Guard against empty placeholders (e.g., padding "")
            has_data = tf.greater(tf.strings.length(img_bytes), 0)
            img = tf.cond(
                has_data,
                lambda: tf.io.decode_image(
                    img_bytes,
                    channels=3,
                    expand_animations=False,
                    dtype=tf.uint8,
                ),
                lambda: tf.zeros([1, 1, 3], dtype=tf.uint8),
            )
        # Optional resize-with-pad to ensure batching shape compatibility
        if resize_to is not None:
            h, w = resize_to
            img = _tf_resize_with_pad(img, h, w)
        return img

    def decode_with_time_dim(img_tensor):
        """Decode images that may have time dimension.

        Handles:
        - Rank 0: scalar encoded string (single image)
        - Rank 1: [T] vector of encoded strings (prediction mode with multiple frames)
        - Rank 3: [H, W, C] single decoded image
        - Rank 4: [T, H, W, C] decoded images with time dimension
        """
        rank = len(img_tensor.shape)

        if rank == 1:  # [T] - multiple encoded strings (prediction mode)
            # Decode each encoded string separately
            # Output: [T, H, W, C] after decoding
            decoded_frames = tf.map_fn(_decode_single, img_tensor, fn_output_signature=tf.uint8)
            # Set explicit shape for downstream processing
            if resize_to is not None:
                h, w = resize_to
                decoded_frames.set_shape([h, w, 3])
            return decoded_frames
        if rank == 4:  # [T, H, W, C] - already decoded with time dimension
            # Apply resize if needed (shouldn't normally happen in this path)
            return img_tensor
        # rank == 0 (scalar string) or rank == 3 ([H, W, C])
        # Single frame: decode if string, otherwise return as-is
        return _decode_single(img_tensor)

    def _decode_frame(traj: dict) -> dict:
        traj["observation"][primary_key] = decode_with_time_dim(traj["observation"][primary_key])
        traj["observation"][wrist_key] = decode_with_time_dim(traj["observation"][wrist_key])
        traj["observation"][wrist_right_key] = decode_with_time_dim(traj["observation"][wrist_right_key])

        return traj

    return _decode_frame
