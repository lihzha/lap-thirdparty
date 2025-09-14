from openpi.shared import normalize as _normalize


def save(directory: str, norm_stats: dict[str, _normalize.NormStats]) -> None:
    import os

    import tensorflow as tf

    """Save the normalization stats to a directory (supports gs:// or local)."""
    path = tf.io.gfile.join(directory, "norm_stats.json")
    tf.io.gfile.makedirs(os.path.dirname(str(path)))
    with tf.io.gfile.GFile(path, "w") as f:
        f.write(_normalize.serialize_json(norm_stats))


def load(directory: str) -> dict[str, _normalize.NormStats]:
    import tensorflow as tf

    """Load the normalization stats from a directory (supports gs:// and local)."""
    path = tf.io.gfile.join(directory, "norm_stats.json")
    if not tf.io.gfile.exists(path):
        raise FileNotFoundError(f"Norm stats file not found at: {path}")
    with tf.io.gfile.GFile(path, "r") as f:
        return _normalize.deserialize_json(f.read())
