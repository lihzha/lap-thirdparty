"""Verification script for hard example tracker fix.

This script verifies that:
1. High-loss samples are not dropped due to image extraction failures
2. Lazy extraction correctly defers image processing to log time
3. Loss values are unique and accurate

Usage:
    python scripts/verify_hard_example_fix.py
"""

import logging
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from openpi_cot.models.adapters.model_adapter import CoTObservation
import openpi_cot.training.vis_tools as vis_tools

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


class SimpleTokenizer:
    def decode(self, tokens):
        return f"action_{len(tokens)}_tokens"


def create_batch_with_missing_images(batch_size: int = 16):
    """Create batch where some samples have missing/corrupted images."""
    rng = np.random.default_rng(seed=42)

    # Intentionally create batch where half the samples have "corrupted" images
    # (represented by very small dimensions that will fail in visualize_language_actions)
    images = {}
    for i in range(batch_size):
        if i % 2 == 0:
            # Normal image
            if "primary" not in images:
                images["primary"] = []
            images["primary"].append(rng.integers(0, 256, (1, 224, 224, 3), dtype=np.uint8))
        else:
            # Corrupted/missing - will cause extraction to fail
            if "primary" not in images:
                images["primary"] = []
            images["primary"].append(rng.integers(0, 256, (1, 1, 1, 3), dtype=np.uint8))  # Too small!

    images["primary"] = np.stack(images["primary"])

    seq_len = 128
    obs = CoTObservation(
        images=images,
        image_masks={"primary": np.ones((batch_size, 1), dtype=bool)},
        state=rng.random((batch_size, 7), dtype=np.float32),
        tokenized_prompt=rng.integers(0, 32000, (batch_size, seq_len), dtype=np.int32),
        tokenized_prompt_mask=rng.random((batch_size, seq_len)) > 0.2,
        tokenized_langact_mask=rng.random((batch_size, seq_len)) < 0.2,
        token_ar_mask=np.ones((batch_size, seq_len), dtype=np.int32),
        token_loss_mask=np.ones((batch_size, seq_len), dtype=bool),
        sample_mask=np.ones(batch_size, dtype=bool),
    )

    return (obs, None)


def verify_lazy_extraction():
    """Verify that lazy extraction allows storing high-loss samples even when images fail."""

    print("=" * 80)
    print("VERIFICATION: Lazy Image Extraction")
    print("=" * 80)

    tokenizer = SimpleTokenizer()
    tracker = vis_tools.HardExampleTracker(
        tokenizer=tokenizer,
        max_hard_examples=10,
        resize_hw=(64, 64),
    )

    batch_size = 16

    # Create losses where the HIGHEST loss samples have corrupted images
    # Old implementation: these would be dropped!
    # New implementation: these should be stored!
    losses = np.array([1.0] * batch_size, dtype=np.float32)
    losses[1] = 10.0  # Highest loss, but image is corrupted (odd index)
    losses[3] = 9.5  # Second highest, corrupted
    losses[5] = 9.0  # Third highest, corrupted
    losses[0] = 8.0  # Fourth highest, has good image

    print(f"\nGenerated losses: {losses}")
    print(f"Samples with corrupted images (odd indices): {[1, 3, 5, 7, ...]}")
    print("Expected: Top-3 losses (10.0, 9.5, 9.0) should be in buffer even though images are corrupted")

    # Create batch with some corrupted images
    batch = create_batch_with_missing_images(batch_size)

    # Update tracker
    tracker.update(losses)

    # Add examples - NEW: should store metadata even if images fail
    tracker.add_local_examples(
        step_idx=0,
        host_batch_local=batch,
        local_losses=losses,
        global_idx_base=0,
        process_idx=0,
    )

    print("\nBuffer after add_local_examples:")
    print(f"  Buffer size: {len(tracker._hard_example_buffer)}")

    if len(tracker._hard_example_buffer) == 0:
        print("❌ FAIL: Buffer is empty! Old eager extraction logic still present?")
        return False

    buffer_losses = sorted([e["loss"] for e in tracker._hard_example_buffer], reverse=True)
    print(f"  Buffer losses (top-5): {buffer_losses[:5]}")

    # Check if high-loss samples are in buffer
    has_10_0 = any(abs(e["loss"] - 10.0) < 0.01 for e in tracker._hard_example_buffer)
    has_9_5 = any(abs(e["loss"] - 9.5) < 0.01 for e in tracker._hard_example_buffer)
    has_9_0 = any(abs(e["loss"] - 9.0) < 0.01 for e in tracker._hard_example_buffer)

    print("\nHigh-loss samples in buffer:")
    print(f"  Loss=10.0 (corrupted image): {'✓' if has_10_0 else '✗'}")
    print(f"  Loss=9.5  (corrupted image): {'✓' if has_9_5 else '✗'}")
    print(f"  Loss=9.0  (corrupted image): {'✓' if has_9_0 else '✗'}")

    # Check if images are None (not extracted yet)
    images_none = sum(1 for e in tracker._hard_example_buffer if e["image"] is None)
    print(f"\nEntries with image=None (lazy): {images_none}/{len(tracker._hard_example_buffer)}")

    if images_none == 0:
        print("❌ FAIL: All images were extracted immediately! Lazy extraction not working?")
        return False

    # Now call log_if_ready to trigger lazy extraction
    print("\nCalling log_if_ready() to trigger lazy extraction...")
    payload = tracker.log_if_ready(step_idx=0)

    if payload is None:
        print("❌ FAIL: No payload returned!")
        return False

    entries = payload.get("entries", [])
    print("\nPayload after lazy extraction:")
    print(f"  Entries with images: {len(entries)}")

    if len(entries) > 0:
        logged_losses = [e["loss"] for e in entries]
        print(f"  Logged losses: {logged_losses}")
        max_logged = max(logged_losses)
        print(f"  Max logged loss: {max_logged:.2f}")
        print("  Expected max: 10.00")

        if abs(max_logged - 10.0) < 0.01:
            print("\n✅ PASS: Lazy extraction successfully captured highest-loss sample!")
            print("  Even though its image extraction failed, metadata was stored")
            print("  and it's available for logging (image may be None/placeholder)")
            return True
        print(f"\n❌ FAIL: Max logged loss {max_logged:.2f} != expected 10.0")
        return False
    print("❌ FAIL: No entries with images!")
    return False


def verify_no_duplicate_losses():
    """Verify that each sample has a unique loss value."""

    print("\n" + "=" * 80)
    print("VERIFICATION: Unique Loss Values")
    print("=" * 80)

    tokenizer = SimpleTokenizer()
    tracker = vis_tools.HardExampleTracker(
        tokenizer=tokenizer,
        max_hard_examples=20,
        resize_hw=(64, 64),
    )

    # Create unique losses
    rng = np.random.default_rng(seed=123)
    losses = rng.uniform(1.0, 10.0, size=32).astype(np.float32)

    print(f"\nGenerated {len(losses)} unique losses")
    print(f"  Sample: {losses[:5]}")

    # Simple batch (all good images)
    images = {"primary": rng.integers(0, 256, (32, 224, 224, 3), dtype=np.uint8)}
    obs = CoTObservation(
        images=images,
        image_masks={"primary": np.ones((32, 1), dtype=bool)},
        state=rng.random((32, 7), dtype=np.float32),
        tokenized_prompt=rng.integers(0, 32000, (32, 128), dtype=np.int32),
        tokenized_prompt_mask=rng.random((32, 128)) > 0.2,
        tokenized_langact_mask=rng.random((32, 128)) < 0.2,
        token_ar_mask=np.ones((32, 128), dtype=np.int32),
        token_loss_mask=np.ones((32, 128), dtype=bool),
        sample_mask=np.ones(32, dtype=bool),
    )
    batch = (obs, None)

    tracker.update(losses)
    tracker.add_local_examples(
        step_idx=0,
        host_batch_local=batch,
        local_losses=losses,
        global_idx_base=0,
        process_idx=0,
    )

    buffer_losses = [e["loss"] for e in tracker._hard_example_buffer]
    unique_losses = len(set(buffer_losses))

    print("\nBuffer statistics:")
    print(f"  Total entries: {len(tracker._hard_example_buffer)}")
    print(f"  Unique losses: {unique_losses}")
    print("  Expected: All unique (no duplicates)")

    if unique_losses == len(buffer_losses):
        print("\n✅ PASS: All losses are unique!")
        return True
    duplicates = len(buffer_losses) - unique_losses
    print(f"\n❌ FAIL: Found {duplicates} duplicate loss values!")
    return False


def main():
    print("\n" + "=" * 80)
    print("HARD EXAMPLE TRACKER FIX VERIFICATION")
    print("=" * 80)

    tests = [
        ("Lazy Extraction", verify_lazy_extraction),
        ("Unique Losses", verify_no_duplicate_losses),
    ]

    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ ERROR in {name}: {e}")
            import traceback

            traceback.print_exc()
            results.append((name, False))

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    passed = sum(1 for _, r in results if r)
    total = len(results)

    for name, result in results:
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"{status}: {name}")

    print(f"\n{passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All verifications passed! The fix is working correctly.")
        return 0
    print(f"\n⚠️  {total - passed} verification(s) failed.")
    print("\nThis may indicate:")
    print("1. The lazy extraction logic is not implemented correctly")
    print("2. High-loss samples are still being dropped due to image failures")
    print("3. The batch cache is not working as expected")
    return 1


if __name__ == "__main__":
    sys.exit(main())
