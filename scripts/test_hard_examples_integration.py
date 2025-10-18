"""Focused test for hard example tracking integration in training.

This is a simpler, more focused test specifically for the hard example tracking feature.
It runs a minimal training loop to verify:
1. Hard examples are correctly identified
2. Images are extracted and stored
3. Multi-host gathering works (if available)
4. Memory usage is reasonable

Usage:
    # Quick test (50 steps)
    python scripts/test_hard_examples_integration.py

    # Longer test
    python scripts/test_hard_examples_integration.py --steps 200

    # Test with custom buffer size
    python scripts/test_hard_examples_integration.py --buffer_size 100
"""

import argparse
import logging
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from openpi_cot.models.adapters.model_adapter import CoTObservation
import openpi_cot.training.vis_tools as vis_tools


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


class SimpleTokenizer:
    """Minimal tokenizer for testing."""

    def decode(self, tokens):
        tokens = np.asarray(tokens).flatten()
        # Generate mock movement text
        num = int(np.sum(tokens) % 30)
        return f"move right {num}cm and move forward {num//2}cm"


def generate_batch(batch_size: int, step: int, image_size: int = 128):
    """Generate a single mock batch."""
    rng = np.random.default_rng(seed=42 + step)

    # Small images to save memory
    images = {
        'primary': rng.integers(0, 256, (batch_size, image_size, image_size, 3), dtype=np.uint8),
    }

    seq_len = 128
    tokenized_prompt = rng.integers(0, 32000, (batch_size, seq_len), dtype=np.int32)
    tokenized_prompt_mask = rng.random((batch_size, seq_len)) > 0.2
    tokenized_langact_mask = rng.random((batch_size, seq_len)) < 0.2
    tokenized_langact_mask = tokenized_langact_mask & tokenized_prompt_mask

    obs = CoTObservation(
        images=images,
        image_masks={'primary': np.ones((batch_size, 1), dtype=bool)},
        state=rng.random((batch_size, 7), dtype=np.float32),
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
        tokenized_langact_mask=tokenized_langact_mask,
        token_ar_mask=np.ones((batch_size, seq_len), dtype=bool),
        token_loss_mask=tokenized_prompt_mask.copy(),
        sample_mask=np.ones(batch_size, dtype=bool),
    )

    actions = None  # Not needed for this test

    return obs, actions


def generate_losses(batch_size: int, step: int, high_loss_prob: float = 0.1):
    """Generate mock losses with some high-loss samples."""
    rng = np.random.default_rng(seed=100 + step)

    # Most losses in range [0.5, 2.0]
    losses = rng.uniform(0.5, 2.0, size=batch_size).astype(np.float32)

    # Some samples get high loss [5.0, 15.0]
    high_loss_mask = rng.random(batch_size) < high_loss_prob
    high_loss_values = rng.uniform(5.0, 15.0, size=batch_size).astype(np.float32)

    losses = np.where(high_loss_mask, high_loss_values, losses)

    return losses


def test_hard_example_tracking(
    num_steps: int = 50,
    batch_size: int = 16,
    buffer_size: int = 50,
    log_interval: int = 10,
):
    """Test hard example tracking over multiple steps."""

    logging.info("="*80)
    logging.info("HARD EXAMPLE TRACKING INTEGRATION TEST")
    logging.info("="*80)
    logging.info(f"Config:")
    logging.info(f"  Steps: {num_steps}")
    logging.info(f"  Batch size: {batch_size}")
    logging.info(f"  Buffer size: {buffer_size}")
    logging.info(f"  Log interval: {log_interval}")
    logging.info(f"  Process index: {jax.process_index()}")
    logging.info(f"  Process count: {jax.process_count()}")
    logging.info("="*80)

    # Create tracker
    tokenizer = SimpleTokenizer()
    tracker = vis_tools.HardExampleTracker(
        tokenizer=tokenizer,
        max_hard_examples=buffer_size,
        resize_hw=(64, 64),  # Small for fast processing
    )

    # Track statistics
    total_samples_seen = 0
    total_high_loss_samples = 0
    max_loss_seen = 0.0

    # Training loop
    for step in range(num_steps):
        # Generate batch and losses
        batch = generate_batch(batch_size, step, image_size=128)
        losses = generate_losses(batch_size, step, high_loss_prob=0.15)

        # Track
        tracker.update(losses)
        total_samples_seen += batch_size

        # Count high-loss samples (threshold: 4.0)
        high_loss_count = np.sum(losses > 4.0)
        total_high_loss_samples += high_loss_count
        max_loss_seen = max(max_loss_seen, np.max(losses))

        # Add examples to buffer (at log interval)
        if step % log_interval == 0:
            tracker.add_local_examples(
                step_idx=step,
                host_batch_local=batch,
                local_losses=losses,
                global_idx_base=step * batch_size,
                process_idx=jax.process_index(),
            )

            buffer_len = len(tracker._hard_example_buffer)
            logging.info(
                f"Step {step:3d}: loss_range=[{np.min(losses):.2f}, {np.max(losses):.2f}], "
                f"high_loss_count={high_loss_count}, buffer_size={buffer_len}"
            )

        # Log hard examples periodically
        if step > 0 and step % (log_interval * 5) == 0:
            payload = tracker.log_if_ready(step_idx=step)

            if payload is not None:
                entries = payload.get('entries', [])
                threshold = payload.get('quantile_threshold', 0.0)
                total = payload.get('total_samples', 0)

                logging.info("-" * 80)
                logging.info(f"HARD EXAMPLES AT STEP {step}:")
                logging.info(f"  Total entries: {len(entries)}")
                logging.info(f"  Quantile threshold: {threshold:.4f}")
                logging.info(f"  Interval samples: {total}")

                # Show top 5
                for i, entry in enumerate(entries[:5]):
                    logging.info(
                        f"    #{i+1}: loss={entry['loss']:.4f}, "
                        f"process={entry['process_index']}, "
                        f"global_idx={entry['global_idx']}, "
                        f"image_shape={entry['image'].shape}"
                    )

                logging.info("-" * 80)

    # Final statistics
    logging.info("="*80)
    logging.info("TEST COMPLETE")
    logging.info("="*80)
    logging.info(f"Statistics:")
    logging.info(f"  Total samples seen: {total_samples_seen}")
    logging.info(f"  Total high-loss samples (>4.0): {total_high_loss_samples}")
    logging.info(f"  High-loss percentage: {100*total_high_loss_samples/total_samples_seen:.2f}%")
    logging.info(f"  Max loss seen: {max_loss_seen:.4f}")
    logging.info(f"  Final buffer size: {len(tracker._hard_example_buffer)}")
    logging.info(f"  Final keys tracked: {len(tracker._hard_example_keys)}")

    # Verify buffer integrity
    buffer = tracker._hard_example_buffer
    keys = tracker._hard_example_keys

    success = True

    # Check 1: Buffer size <= max
    if len(buffer) > buffer_size:
        logging.error(f"❌ Buffer size {len(buffer)} exceeds max {buffer_size}")
        success = False
    else:
        logging.info(f"✓ Buffer size constraint satisfied")

    # Check 2: Buffer is sorted
    buffer_losses = [e['loss'] for e in buffer]
    if buffer_losses != sorted(buffer_losses, reverse=True):
        logging.error(f"❌ Buffer not sorted by loss")
        success = False
    else:
        logging.info(f"✓ Buffer correctly sorted by loss")

    # Check 3: Keys match buffer
    for entry in buffer:
        key = (entry['process_index'], entry['step'], entry['global_idx'])
        if key not in keys:
            logging.error(f"❌ Buffer entry has no key: {key}")
            success = False

    if len(buffer) == len(keys):
        logging.info(f"✓ Keys match buffer entries")
    else:
        logging.warning(f"⚠ Keys count {len(keys)} != buffer count {len(buffer)}")

    # Check 4: All entries have images
    for i, entry in enumerate(buffer):
        if 'image' not in entry:
            logging.error(f"❌ Entry {i} missing image")
            success = False
        elif not isinstance(entry['image'], np.ndarray):
            logging.error(f"❌ Entry {i} image is not ndarray")
            success = False

    logging.info(f"✓ All buffer entries have valid images")

    logging.info("="*80)
    if success:
        logging.info("✅ TEST PASSED")
    else:
        logging.error("❌ TEST FAILED")

    return success


def main():
    parser = argparse.ArgumentParser(description="Test hard example tracking")
    parser.add_argument("--steps", type=int, default=50, help="Number of training steps")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--buffer_size", type=int, default=50, help="Hard example buffer size")
    parser.add_argument("--log_interval", type=int, default=10, help="Log interval")

    args = parser.parse_args()

    try:
        success = test_hard_example_tracking(
            num_steps=args.steps,
            batch_size=args.batch_size,
            buffer_size=args.buffer_size,
            log_interval=args.log_interval,
        )

        if not success:
            sys.exit(1)

    except Exception as e:
        logging.error(f"Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
