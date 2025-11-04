"""Comprehensive test for logging metrics.

This test verifies that all logged metrics are computed correctly by:
1. Creating fake model outputs with known values
2. Using the same logging logic as train.py
3. Computing expected values manually
4. Comparing actual vs expected metrics
"""

import logging
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Import the logging utilities we're testing
import openpi_cot.training.log_util as log_util


class FakeTokenizer:
    """Fake tokenizer for testing."""

    def decode(self, tokens):
        """Decode tokens to dataset name."""
        # Use first token as dataset identifier
        dataset_map = {
            100: "dataset_a",
            101: "dataset_b",
            102: "dataset_c",
            103: "dataset_d",
        }
        first_token = int(tokens[0]) if isinstance(tokens, (list, np.ndarray, jnp.ndarray)) else int(tokens)
        return dataset_map.get(first_token, "unknown")


class FakeBatchObservation:
    """Fake observation for testing."""

    def __init__(self, tokenized_dataset_name):
        self.tokenized_dataset_name = tokenized_dataset_name


def create_fake_batch(
    batch_size: int,
    dataset_ids: list[int],
    seed: int = 42,
) -> tuple[tuple[FakeBatchObservation, Any], dict[str, np.ndarray]]:
    """Create a fake batch with known dataset assignments and metrics.

    Args:
        batch_size: Number of samples in batch
        dataset_ids: List of dataset IDs (100, 101, 102, 103) for each sample
        seed: Random seed for reproducibility

    Returns:
        Tuple of (batch, expected_metrics_dict)
    """
    np.random.seed(seed)

    # Create tokenized dataset names
    tokenized_names = np.zeros((batch_size, 10), dtype=np.int32)
    for i, dataset_id in enumerate(dataset_ids):
        tokenized_names[i, 0] = dataset_id

    # Create fake observation
    obs = FakeBatchObservation(tokenized_dataset_name=tokenized_names)
    batch = (obs, None)  # (observation, actions)

    # Generate random metrics for each sample
    per_sample_losses = np.random.uniform(0.1, 2.0, size=batch_size).astype(np.float32)

    # Generate token-level metrics (critical, number, direction)
    # Each sample has random number of tokens and correct predictions
    critical_correct = np.random.randint(0, 20, size=batch_size)
    critical_total = critical_correct + np.random.randint(0, 10, size=batch_size)

    number_correct = np.random.randint(0, 15, size=batch_size)
    number_total = number_correct + np.random.randint(0, 8, size=batch_size)

    direction_correct = np.random.randint(0, 10, size=batch_size)
    direction_total = direction_correct + np.random.randint(0, 5, size=batch_size)

    # Store expected values for verification
    expected = {
        "dataset_ids": dataset_ids,
        "per_sample_losses": per_sample_losses,
        "critical_correct": critical_correct,
        "critical_total": critical_total,
        "number_correct": number_correct,
        "number_total": number_total,
        "direction_correct": direction_correct,
        "direction_total": direction_total,
    }

    return batch, expected


def create_fake_info_from_expected(
    expected: dict[str, np.ndarray],
    sample_type: str = "all",
    is_pred_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Create a fake info dict in the format returned by compute_loss.

    Args:
        expected: Expected metrics dict from create_fake_batch
        sample_type: Type of samples ('all', 'pred', 'langact')
        is_pred_mask: Boolean mask indicating which samples are predictions

    Returns:
        Info dict with metrics in the format expected by buffer_dataset_metrics_from_batch
    """
    batch_size = len(expected["per_sample_losses"])

    if sample_type == "all":
        # No separation between pred and langact
        info = {
            "per_sample_loss": expected["per_sample_losses"],
            "per_sample_critical_correct": expected["critical_correct"],
            "per_sample_critical_total": expected["critical_total"],
            "per_sample_number_correct": expected["number_correct"],
            "per_sample_number_total": expected["number_total"],
            "per_sample_direction_correct": expected["direction_correct"],
            "per_sample_direction_total": expected["direction_total"],
        }
    else:
        # Separate pred and langact samples
        if is_pred_mask is None:
            # Randomly split samples
            is_pred_mask = np.random.rand(batch_size) > 0.5

        # Create separate metrics for pred samples
        pred_losses = np.where(is_pred_mask, expected["per_sample_losses"], 0.0)
        pred_critical_correct = np.where(is_pred_mask, expected["critical_correct"], 0)
        pred_critical_total = np.where(is_pred_mask, expected["critical_total"], 0)
        pred_number_correct = np.where(is_pred_mask, expected["number_correct"], 0)
        pred_number_total = np.where(is_pred_mask, expected["number_total"], 0)
        pred_direction_correct = np.where(is_pred_mask, expected["direction_correct"], 0)
        pred_direction_total = np.where(is_pred_mask, expected["direction_total"], 0)

        # Create separate metrics for langact samples
        langact_losses = np.where(~is_pred_mask, expected["per_sample_losses"], 0.0)
        langact_critical_correct = np.where(~is_pred_mask, expected["critical_correct"], 0)
        langact_critical_total = np.where(~is_pred_mask, expected["critical_total"], 0)
        langact_number_correct = np.where(~is_pred_mask, expected["number_correct"], 0)
        langact_number_total = np.where(~is_pred_mask, expected["number_total"], 0)
        langact_direction_correct = np.where(~is_pred_mask, expected["direction_correct"], 0)
        langact_direction_total = np.where(~is_pred_mask, expected["direction_total"], 0)

        info = {
            "per_sample_loss": expected["per_sample_losses"],
            "pred_per_sample_loss": pred_losses,
            "pred_per_sample_critical_correct": pred_critical_correct,
            "pred_per_sample_critical_total": pred_critical_total,
            "pred_per_sample_number_correct": pred_number_correct,
            "pred_per_sample_number_total": pred_number_total,
            "pred_per_sample_direction_correct": pred_direction_correct,
            "pred_per_sample_direction_total": pred_direction_total,
            "langact_per_sample_loss": langact_losses,
            "langact_per_sample_critical_correct": langact_critical_correct,
            "langact_per_sample_critical_total": langact_critical_total,
            "langact_per_sample_number_correct": langact_number_correct,
            "langact_per_sample_number_total": langact_number_total,
            "langact_per_sample_direction_correct": langact_direction_correct,
            "langact_per_sample_direction_total": langact_direction_total,
        }

        # Store mask for verification
        expected["is_pred_mask"] = is_pred_mask

    return info


def compute_expected_metrics(
    expected_list: list[dict[str, np.ndarray]],
    sample_type: str = "all",
    prefix: str = "",
) -> dict[str, float]:
    """Manually compute expected metrics from expected values.

    Args:
        expected_list: List of expected dicts from create_fake_batch
        sample_type: Type of samples to compute metrics for
        prefix: Prefix for metric names (e.g., "val_")

    Returns:
        Dictionary of expected metrics
    """
    # Collect all data
    all_dataset_ids = []
    all_losses = []
    all_critical_correct = []
    all_critical_total = []
    all_number_correct = []
    all_number_total = []
    all_direction_correct = []
    all_direction_total = []

    for expected in expected_list:
        dataset_ids = expected["dataset_ids"]
        losses = expected["per_sample_losses"]
        is_pred_mask = expected.get("is_pred_mask")

        # Apply sample type filter if needed
        if sample_type == "pred" and is_pred_mask is not None:
            mask = is_pred_mask
        elif sample_type == "langact" and is_pred_mask is not None:
            mask = ~is_pred_mask
        else:
            mask = np.ones(len(dataset_ids), dtype=bool)

        # Collect data for masked samples
        for i, dataset_id in enumerate(dataset_ids):
            if not mask[i]:
                continue

            all_dataset_ids.append(dataset_id)
            all_losses.append(losses[i])
            all_critical_correct.append(expected["critical_correct"][i])
            all_critical_total.append(expected["critical_total"][i])
            all_number_correct.append(expected["number_correct"][i])
            all_number_total.append(expected["number_total"][i])
            all_direction_correct.append(expected["direction_correct"][i])
            all_direction_total.append(expected["direction_total"][i])

    # Group by dataset
    dataset_map = {100: "dataset_a", 101: "dataset_b", 102: "dataset_c", 103: "dataset_d"}
    dataset_stats = {}

    for i, dataset_id in enumerate(all_dataset_ids):
        dataset_name = dataset_map[dataset_id]

        if dataset_name not in dataset_stats:
            dataset_stats[dataset_name] = {
                "total_loss": 0.0,
                "count": 0,
                "critical_correct": 0,
                "critical_total": 0,
                "number_correct": 0,
                "number_total": 0,
                "direction_correct": 0,
                "direction_total": 0,
            }

        stats = dataset_stats[dataset_name]
        stats["total_loss"] += float(all_losses[i])
        stats["count"] += 1
        stats["critical_correct"] += int(all_critical_correct[i])
        stats["critical_total"] += int(all_critical_total[i])
        stats["number_correct"] += int(all_number_correct[i])
        stats["number_total"] += int(all_number_total[i])
        stats["direction_correct"] += int(all_direction_correct[i])
        stats["direction_total"] += int(all_direction_total[i])

    # Compute metrics
    metrics = {}
    type_prefix = f"{sample_type}_" if sample_type != "all" else ""

    for dataset_name, stats in dataset_stats.items():
        base_key = f"{prefix}dataset/{dataset_name}/{type_prefix}"

        # Average loss
        metrics[f"{base_key}avg_loss"] = stats["total_loss"] / stats["count"]
        metrics[f"{base_key}count"] = stats["count"]

        # Token accuracies
        if stats["critical_total"] > 0:
            metrics[f"{base_key}avg_critical_token_acc"] = stats["critical_correct"] / stats["critical_total"]
            metrics[f"{base_key}critical_token_count"] = stats["critical_total"]

        if stats["number_total"] > 0:
            metrics[f"{base_key}avg_number_token_acc"] = stats["number_correct"] / stats["number_total"]
            metrics[f"{base_key}number_token_count"] = stats["number_total"]

        if stats["direction_total"] > 0:
            metrics[f"{base_key}avg_direction_token_acc"] = stats["direction_correct"] / stats["direction_total"]
            metrics[f"{base_key}direction_token_count"] = stats["direction_total"]

    return metrics


def test_dataset_stats_tracker_single_type():
    """Test DatasetStatsTracker with single sample type (all)."""
    print("\n" + "=" * 80)
    print("TEST: DatasetStatsTracker with single sample type")
    print("=" * 80)

    tokenizer = FakeTokenizer()
    tracker = log_util.DatasetStatsTracker()
    buffer = log_util.LocalDatasetInfoBuffer(tokenizer)

    # Create multiple batches with known dataset distributions
    batches_and_expected = []

    # Batch 1: dataset_a=2, dataset_b=2
    batch1, expected1 = create_fake_batch(4, [100, 100, 101, 101], seed=1)
    batches_and_expected.append((batch1, expected1))

    # Batch 2: dataset_a=1, dataset_b=1, dataset_c=2
    batch2, expected2 = create_fake_batch(4, [100, 101, 102, 102], seed=2)
    batches_and_expected.append((batch2, expected2))

    # Process batches through logging system
    for batch, expected in batches_and_expected:
        info = create_fake_info_from_expected(expected, sample_type="all")
        log_util.buffer_dataset_metrics_from_batch(buffer, batch, info)

    # Gather and update stats (simulates what happens at log_interval)
    buffer.gather_and_update_stats(tracker)

    # Get computed metrics
    computed_metrics = tracker.get_metrics(prefix="")

    # Compute expected metrics manually
    expected_metrics = compute_expected_metrics([expected1, expected2], sample_type="all", prefix="")

    # Verify metrics match
    print(f"\nExpected {len(expected_metrics)} metrics, got {len(computed_metrics)}")

    all_keys = set(expected_metrics.keys()) | set(computed_metrics.keys())
    mismatches = []

    for key in sorted(all_keys):
        expected_val = expected_metrics.get(key)
        computed_val = computed_metrics.get(key)

        if expected_val is None:
            mismatches.append(f"  ❌ {key}: missing in expected")
        elif computed_val is None:
            mismatches.append(f"  ❌ {key}: missing in computed")
        elif not np.isclose(expected_val, computed_val, rtol=1e-5):
            mismatches.append(f"  ❌ {key}: expected={expected_val:.6f}, got={computed_val:.6f}")
        else:
            print(f"  ✓ {key}: {computed_val:.6f}")

    if mismatches:
        print("\nMismatches found:")
        for msg in mismatches:
            print(msg)
        raise AssertionError(f"Found {len(mismatches)} metric mismatches")

    print("\n✅ All metrics match for single sample type!")


def test_dataset_stats_tracker_multiple_types():
    """Test DatasetStatsTracker with multiple sample types (pred and langact)."""
    print("\n" + "=" * 80)
    print("TEST: DatasetStatsTracker with multiple sample types")
    print("=" * 80)

    tokenizer = FakeTokenizer()
    tracker = log_util.DatasetStatsTracker()
    buffer = log_util.LocalDatasetInfoBuffer(tokenizer)

    # Create batches with pred/langact split
    batches_and_expected = []

    # Batch 1: 4 samples, first 2 are pred, last 2 are langact
    batch1, expected1 = create_fake_batch(4, [100, 101, 100, 101], seed=10)
    is_pred_mask1 = np.array([True, True, False, False])
    expected1["is_pred_mask"] = is_pred_mask1
    batches_and_expected.append((batch1, expected1))

    # Batch 2: 4 samples, alternating pred/langact
    batch2, expected2 = create_fake_batch(4, [102, 102, 103, 103], seed=20)
    is_pred_mask2 = np.array([True, False, True, False])
    expected2["is_pred_mask"] = is_pred_mask2
    batches_and_expected.append((batch2, expected2))

    # Process batches
    for batch, expected in batches_and_expected:
        info = create_fake_info_from_expected(expected, sample_type="mixed", is_pred_mask=expected["is_pred_mask"])
        log_util.buffer_dataset_metrics_from_batch(buffer, batch, info)

    # Gather and update stats
    buffer.gather_and_update_stats(tracker)

    # Get computed metrics
    computed_metrics = tracker.get_metrics(prefix="")

    # Compute expected metrics for both types
    expected_metrics_pred = compute_expected_metrics(batches_and_expected, sample_type="pred", prefix="")
    expected_metrics_langact = compute_expected_metrics(
        [exp for _, exp in batches_and_expected], sample_type="langact", prefix=""
    )
    expected_metrics = {**expected_metrics_pred, **expected_metrics_langact}

    # Verify metrics match
    print(f"\nExpected {len(expected_metrics)} metrics, got {len(computed_metrics)}")

    all_keys = set(expected_metrics.keys()) | set(computed_metrics.keys())
    mismatches = []

    for key in sorted(all_keys):
        expected_val = expected_metrics.get(key)
        computed_val = computed_metrics.get(key)

        if expected_val is None:
            mismatches.append(f"  ❌ {key}: missing in expected")
        elif computed_val is None:
            mismatches.append(f"  ❌ {key}: missing in computed")
        elif not np.isclose(expected_val, computed_val, rtol=1e-5):
            mismatches.append(f"  ❌ {key}: expected={expected_val:.6f}, got={computed_val:.6f}")
        else:
            print(f"  ✓ {key}: {computed_val:.6f}")

    if mismatches:
        print("\nMismatches found:")
        for msg in mismatches:
            print(msg)
        raise AssertionError(f"Found {len(mismatches)} metric mismatches")

    print("\n✅ All metrics match for multiple sample types!")


def test_cumulative_stats():
    """Test that cumulative stats are preserved across resets."""
    print("\n" + "=" * 80)
    print("TEST: Cumulative stats preservation")
    print("=" * 80)

    tokenizer = FakeTokenizer()
    tracker = log_util.DatasetStatsTracker()
    buffer = log_util.LocalDatasetInfoBuffer(tokenizer)

    # Process first batch
    batch1, expected1 = create_fake_batch(4, [100, 100, 101, 101], seed=30)
    info1 = create_fake_info_from_expected(expected1, sample_type="all")
    log_util.buffer_dataset_metrics_from_batch(buffer, batch1, info1)
    buffer.gather_and_update_stats(tracker)

    # Get metrics before reset
    metrics_before_reset = tracker.get_metrics(prefix="")
    print("\nMetrics before reset:")
    for key in sorted(metrics_before_reset.keys()):
        if "cumulative" in key:
            print(f"  {key}: {metrics_before_reset[key]:.6f}")

    # Reset interval stats
    tracker.reset()

    # Process second batch
    batch2, expected2 = create_fake_batch(4, [100, 100, 102, 102], seed=40)
    info2 = create_fake_info_from_expected(expected2, sample_type="all")
    buffer2 = log_util.LocalDatasetInfoBuffer(tokenizer)
    log_util.buffer_dataset_metrics_from_batch(buffer2, batch2, info2)
    buffer2.gather_and_update_stats(tracker)

    # Get metrics after reset and second batch
    metrics_after = tracker.get_metrics(prefix="")
    print("\nMetrics after reset and second batch:")
    for key in sorted(metrics_after.keys()):
        if "cumulative" in key:
            print(f"  {key}: {metrics_after[key]:.6f}")

    # Verify cumulative stats include both batches
    expected_cumulative = compute_expected_metrics([expected1, expected2], sample_type="all", prefix="")

    mismatches = []
    for key in sorted(expected_cumulative.keys()):
        # Convert to cumulative key
        cumulative_key = key.replace("avg_loss", "cumulative_avg_loss")
        cumulative_key = cumulative_key.replace("avg_critical", "cumulative_avg_critical")
        cumulative_key = cumulative_key.replace("avg_number", "cumulative_avg_number")
        cumulative_key = cumulative_key.replace("avg_direction", "cumulative_avg_direction")
        cumulative_key = cumulative_key.replace("count", "cumulative_count")

        # Skip non-cumulative keys
        if not cumulative_key.startswith("dataset/") or "_count" in cumulative_key or "cumulative" not in cumulative_key:
            continue

        expected_val = expected_cumulative[key]
        computed_val = metrics_after.get(cumulative_key)

        if computed_val is None:
            mismatches.append(f"  ❌ {cumulative_key}: missing in computed")
        elif not np.isclose(expected_val, computed_val, rtol=1e-5):
            mismatches.append(f"  ❌ {cumulative_key}: expected={expected_val:.6f}, got={computed_val:.6f}")
        else:
            print(f"  ✓ {cumulative_key}: {computed_val:.6f}")

    if mismatches:
        print("\nMismatches found:")
        for msg in mismatches:
            print(msg)
        raise AssertionError(f"Found {len(mismatches)} cumulative metric mismatches")

    print("\n✅ Cumulative stats correctly preserved!")


def test_validation_metrics_with_prefix():
    """Test that validation metrics get correct prefix."""
    print("\n" + "=" * 80)
    print("TEST: Validation metrics with val_ prefix")
    print("=" * 80)

    tokenizer = FakeTokenizer()
    tracker = log_util.DatasetStatsTracker()
    buffer = log_util.LocalDatasetInfoBuffer(tokenizer)

    # Create validation batch
    batch, expected = create_fake_batch(4, [100, 101, 102, 103], seed=50)
    info = create_fake_info_from_expected(expected, sample_type="all")
    log_util.buffer_dataset_metrics_from_batch(buffer, batch, info)
    buffer.gather_and_update_stats(tracker)

    # Get metrics with val_ prefix
    val_metrics = tracker.get_metrics(prefix="val_")

    # Verify all keys have val_ prefix
    print(f"\nFound {len(val_metrics)} validation metrics")
    non_prefixed = [key for key in val_metrics.keys() if not key.startswith("val_")]

    if non_prefixed:
        print(f"❌ Found {len(non_prefixed)} keys without val_ prefix:")
        for key in non_prefixed:
            print(f"  - {key}")
        raise AssertionError("Some metrics missing val_ prefix")

    # Verify metrics are correct
    expected_metrics = compute_expected_metrics([expected], sample_type="all", prefix="val_")

    mismatches = []
    for key in sorted(expected_metrics.keys()):
        expected_val = expected_metrics[key]
        computed_val = val_metrics.get(key)

        if computed_val is None:
            mismatches.append(f"  ❌ {key}: missing in computed")
        elif not np.isclose(expected_val, computed_val, rtol=1e-5):
            mismatches.append(f"  ❌ {key}: expected={expected_val:.6f}, got={computed_val:.6f}")
        else:
            print(f"  ✓ {key}: {computed_val:.6f}")

    if mismatches:
        print("\nMismatches found:")
        for msg in mismatches:
            print(msg)
        raise AssertionError(f"Found {len(mismatches)} validation metric mismatches")

    print("\n✅ All validation metrics have correct prefix!")


def test_edge_cases():
    """Test edge cases like empty batches, zero tokens, etc."""
    print("\n" + "=" * 80)
    print("TEST: Edge cases")
    print("=" * 80)

    tokenizer = FakeTokenizer()
    tracker = log_util.DatasetStatsTracker()

    # Test 1: Batch with zero critical tokens for some samples
    print("\n--- Test 1: Zero critical tokens ---")
    batch1, expected1 = create_fake_batch(4, [100, 100, 101, 101], seed=60)
    expected1["critical_total"][0] = 0  # First sample has no critical tokens
    expected1["critical_correct"][0] = 0
    info1 = create_fake_info_from_expected(expected1, sample_type="all")

    buffer1 = log_util.LocalDatasetInfoBuffer(tokenizer)
    log_util.buffer_dataset_metrics_from_batch(buffer1, batch1, info1)
    buffer1.gather_and_update_stats(tracker)

    metrics1 = tracker.get_metrics(prefix="")
    print("✓ No crash with zero critical tokens")

    # Test 2: Single sample batch
    print("\n--- Test 2: Single sample batch ---")
    tracker2 = log_util.DatasetStatsTracker()
    batch2, expected2 = create_fake_batch(1, [100], seed=70)
    info2 = create_fake_info_from_expected(expected2, sample_type="all")

    buffer2 = log_util.LocalDatasetInfoBuffer(tokenizer)
    log_util.buffer_dataset_metrics_from_batch(buffer2, batch2, info2)
    buffer2.gather_and_update_stats(tracker2)

    metrics2 = tracker2.get_metrics(prefix="")
    assert "dataset/dataset_a/avg_loss" in metrics2
    print("✓ Single sample batch handled correctly")

    # Test 3: All samples from same dataset
    print("\n--- Test 3: All samples from same dataset ---")
    tracker3 = log_util.DatasetStatsTracker()
    batch3, expected3 = create_fake_batch(8, [102] * 8, seed=80)
    info3 = create_fake_info_from_expected(expected3, sample_type="all")

    buffer3 = log_util.LocalDatasetInfoBuffer(tokenizer)
    log_util.buffer_dataset_metrics_from_batch(buffer3, batch3, info3)
    buffer3.gather_and_update_stats(tracker3)

    metrics3 = tracker3.get_metrics(prefix="")
    # Should only have metrics for dataset_c
    dataset_keys = [k for k in metrics3.keys() if k.startswith("dataset/")]
    assert all("dataset_c" in k for k in dataset_keys)
    print("✓ Single dataset batch handled correctly")

    print("\n✅ All edge cases passed!")


def run_all_tests():
    """Run all tests."""
    print("\n" + "=" * 80)
    print("RUNNING ALL LOGGING METRICS TESTS")
    print("=" * 80)

    tests = [
        test_dataset_stats_tracker_single_type,
        test_dataset_stats_tracker_multiple_types,
        test_cumulative_stats,
        test_validation_metrics_with_prefix,
        test_edge_cases,
    ]

    passed = 0
    failed = 0

    for test_func in tests:
        try:
            test_func()
            passed += 1
        except Exception as e:
            failed += 1
            print(f"\n❌ {test_func.__name__} FAILED:")
            print(f"   {str(e)}")
            import traceback

            traceback.print_exc()

    print("\n" + "=" * 80)
    print(f"TEST SUMMARY: {passed} passed, {failed} failed")
    print("=" * 80)

    if failed > 0:
        raise AssertionError(f"{failed} test(s) failed")


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(level=logging.WARNING)

    # Run all tests
    run_all_tests()

    print("\n🎉 ALL TESTS PASSED! 🎉\n")
