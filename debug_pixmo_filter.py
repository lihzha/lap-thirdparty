"""Debug script to check what's happening with pixmo_point filtering."""
import tensorflow as tf
import numpy as np

# Test what happens with points_to_text when there are 0 points
def points_to_text_test(points):
    """Test version of points_to_text."""
    if tf.shape(points)[0] == 0:
        print("WARNING: Got 0 points!")

    # Sort points by x*10000 + y for consistent ordering
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    sort_keys = x_coords * 10000.0 + y_coords
    sorted_indices = tf.argsort(sort_keys)
    sorted_points = tf.gather(points, sorted_indices)

    # Extract sorted x and y coordinates
    x_coords = sorted_points[:, 0]
    y_coords = sorted_points[:, 1]

    # Convert to paligemma loc token indices (0-1023)
    N = 1024
    x_indices = tf.cast(tf.round(x_coords / 100.0 * (N - 1)), tf.int32)
    y_indices = tf.cast(tf.round(y_coords / 100.0 * (N - 1)), tf.int32)

    # Format as loc tokens: <locYYYY><locXXXX> for each point
    y_tokens = tf.strings.join(["<loc", tf.strings.as_string(y_indices, width=4, fill="0"), ">"])
    x_tokens = tf.strings.join(["<loc", tf.strings.as_string(x_indices, width=4, fill="0"), ">"])

    # Interleave y and x tokens (y comes first for each point)
    yx_pairs = tf.reshape(tf.stack([y_tokens, x_tokens], axis=1), [-1])

    # Join all tokens together without separator
    result = tf.strings.reduce_join(yx_pairs, separator="")
    return result


# Test with 0 points
print("Testing with 0 points:")
empty_points = tf.zeros([0, 2], dtype=tf.float32)
caption_empty = points_to_text_test(empty_points)
print(f"Caption with 0 points: '{caption_empty.numpy().decode()}'")
print(f"Caption length: {tf.strings.length(caption_empty).numpy()}")
print(f"Is empty: {tf.strings.length(caption_empty).numpy() == 0}")
print()

# Test with 1 point
print("Testing with 1 point:")
one_point = tf.constant([[50.0, 50.0]], dtype=tf.float32)
caption_one = points_to_text_test(one_point)
print(f"Caption with 1 point: '{caption_one.numpy().decode()}'")
print(f"Caption length: {tf.strings.length(caption_one).numpy()}")
print(f"Is empty: {tf.strings.length(caption_one).numpy() == 0}")
print()

# Test with multiple points
print("Testing with 3 points:")
three_points = tf.constant([[25.0, 30.0], [75.0, 80.0], [50.0, 50.0]], dtype=tf.float32)
caption_three = points_to_text_test(three_points)
print(f"Caption with 3 points: '{caption_three.numpy().decode()}'")
print(f"Caption length: {tf.strings.length(caption_three).numpy()}")
print(f"Is empty: {tf.strings.length(caption_three).numpy() == 0}")
