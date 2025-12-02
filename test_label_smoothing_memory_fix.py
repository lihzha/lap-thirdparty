"""Test to verify the memory-efficient label smoothing implementation.

This script compares the old (memory-heavy) and new (memory-efficient) implementations
to ensure they produce identical loss values.
"""

import sys

import flax.nnx as nnx
import jax
import jax.numpy as jnp

from openpi_cot.models.label_smoothing import create_digit_smoothing_kernel, get_digit_to_token_mapping

PALIGEMMA_VOCAB_SIZE = 257_152


def old_implementation(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    token_mask: jnp.ndarray,
    units_number_mask: jnp.ndarray,
    digit_values: jnp.ndarray,
    smoothing_kernel: jnp.ndarray,
) -> jnp.ndarray:
    """Original memory-heavy implementation."""
    vocab_size = logits.shape[2]
    log_probs = nnx.log_softmax(logits, axis=-1)

    # Create full vocab-sized arrays (MEMORY HEAVY!)
    hard_targets = jax.nn.one_hot(labels, vocab_size, dtype=jnp.float32)  # [b, s, V]

    valid_digit_mask = digit_values >= 0
    clipped_digit_values = jnp.clip(digit_values, 0, 9)
    smoothed_dists = smoothing_kernel[clipped_digit_values]  # [b, s, V]

    blend_mask = jnp.logical_and(units_number_mask, valid_digit_mask)[..., None]
    target_distribution = jnp.where(blend_mask, smoothed_dists, hard_targets)  # [b, s, V]

    # KL divergence
    loss = -jnp.sum(target_distribution * log_probs, axis=-1)  # [b, s]

    # Reduce
    reduce_axes = tuple(range(1, loss.ndim))
    masked_loss = loss * token_mask
    denom = jnp.maximum(token_mask.sum(axis=reduce_axes), 1)
    return masked_loss.sum(axis=reduce_axes) / denom


def new_implementation(
    logits: jnp.ndarray,
    labels: jnp.ndarray,
    token_mask: jnp.ndarray,
    units_number_mask: jnp.ndarray,
    digit_values: jnp.ndarray,
    smoothing_kernel: jnp.ndarray,
) -> jnp.ndarray:
    """New memory-efficient implementation."""
    log_probs = nnx.log_softmax(logits, axis=-1)

    # Identify which tokens need smoothing
    valid_digit_mask = digit_values >= 0
    smoothed_token_mask = jnp.logical_and(units_number_mask, valid_digit_mask)

    # 1. Compute hard-target loss (memory efficient)
    gather_idx = jnp.expand_dims(labels.astype(jnp.int32), axis=-1)
    gold_logp = jnp.take_along_axis(log_probs, gather_idx, axis=-1).squeeze(-1)
    hard_loss = -gold_logp

    # 2. Compute smoothed loss for units digits
    clipped_digit_values = jnp.clip(digit_values, 0, 9)
    smoothed_dists = smoothing_kernel[clipped_digit_values]  # [b, s, V]
    smoothed_loss = -jnp.sum(smoothed_dists * log_probs, axis=-1)

    # 3. Blend
    per_token_loss = jnp.where(smoothed_token_mask, smoothed_loss, hard_loss)

    # Reduce
    reduce_axes = tuple(range(1, per_token_loss.ndim))
    masked_loss = per_token_loss * token_mask
    denom = jnp.maximum(token_mask.sum(axis=reduce_axes), 1)
    per_sample_loss = masked_loss.sum(axis=reduce_axes) / denom

    return per_sample_loss


def test_correctness():
    """Test that both implementations produce identical results."""
    print("Testing label smoothing memory fix...")
    print("=" * 80)

    # Create test data
    rng = jax.random.PRNGKey(42)
    batch_size = 4
    seq_len = 32
    vocab_size = PALIGEMMA_VOCAB_SIZE

    # Random logits
    rng, key = jax.random.split(rng)
    logits = jax.random.normal(key, (batch_size, seq_len, vocab_size))

    # Random labels (mix of digit tokens and other tokens)
    rng, key = jax.random.split(rng)
    digit_to_token = get_digit_to_token_mapping()
    digit_tokens = list(digit_to_token.values())

    # Create labels: some are digit tokens, some are not
    labels = jax.random.randint(key, (batch_size, seq_len), 0, 1000)
    # Replace some positions with actual digit tokens
    rng, key = jax.random.split(rng)
    is_digit_pos = jax.random.bernoulli(key, 0.3, (batch_size, seq_len))
    rng, key = jax.random.split(rng)
    random_digits = jax.random.choice(key, jnp.array(digit_tokens), (batch_size, seq_len))
    labels = jnp.where(is_digit_pos, random_digits, labels)

    # Create digit_values: -1 for non-digits, 0-9 for digits
    digit_values = jnp.full((batch_size, seq_len), -1, dtype=jnp.int32)
    token_to_digit = {v: k for k, v in digit_to_token.items()}
    for i in range(batch_size):
        for j in range(seq_len):
            label = int(labels[i, j])
            if label in token_to_digit:
                digit_values = digit_values.at[i, j].set(token_to_digit[label])

    # Create masks
    rng, key = jax.random.split(rng)
    token_mask = jax.random.bernoulli(key, 0.9, (batch_size, seq_len))

    rng, key = jax.random.split(rng)
    units_number_mask = jax.random.bernoulli(key, 0.2, (batch_size, seq_len))

    # Create smoothing kernel
    smoothing_kernel = create_digit_smoothing_kernel(
        vocab_size=vocab_size,
        digit_to_token_id=digit_to_token,
        sigma=1.0,
        support=3,
    )

    print(f"Test setup:")
    print(f"  Batch size: {batch_size}")
    print(f"  Sequence length: {seq_len}")
    print(f"  Vocab size: {vocab_size:,}")
    print(f"  Num units digits: {int(units_number_mask.sum())}")
    print()

    # Compute losses
    print("Computing losses...")
    old_loss = old_implementation(
        logits, labels, token_mask, units_number_mask, digit_values, smoothing_kernel
    )
    new_loss = new_implementation(
        logits, labels, token_mask, units_number_mask, digit_values, smoothing_kernel
    )

    # Compare
    print(f"Old implementation loss: {old_loss}")
    print(f"New implementation loss: {new_loss}")
    print()

    # Check if results match
    max_diff = jnp.max(jnp.abs(old_loss - new_loss))
    mean_diff = jnp.mean(jnp.abs(old_loss - new_loss))

    print(f"Max absolute difference: {max_diff:.2e}")
    print(f"Mean absolute difference: {mean_diff:.2e}")
    print()

    if max_diff < 1e-5:
        print("✅ SUCCESS: Both implementations produce identical results!")
    else:
        print("❌ FAILED: Results differ significantly!")
        print(f"Difference: {old_loss - new_loss}")
        return False

    # Memory estimate
    print()
    print("=" * 80)
    print("Memory comparison:")
    print("=" * 80)

    # Old implementation memory
    old_mem_per_array = batch_size * seq_len * vocab_size * 4  # float32 = 4 bytes
    old_total_mem = 3 * old_mem_per_array  # hard_targets, smoothed_dists, target_distribution

    # New implementation memory (only smoothed_dists)
    new_mem = old_mem_per_array

    print(f"Old implementation:")
    print(f"  - hard_targets: {old_mem_per_array / 1e9:.2f} GB")
    print(f"  - smoothed_dists: {old_mem_per_array / 1e9:.2f} GB")
    print(f"  - target_distribution: {old_mem_per_array / 1e9:.2f} GB")
    print(f"  - TOTAL: {old_total_mem / 1e9:.2f} GB")
    print()
    print(f"New implementation:")
    print(f"  - smoothed_dists only: {new_mem / 1e9:.2f} GB")
    print(f"  - TOTAL: {new_mem / 1e9:.2f} GB")
    print()
    print(f"Memory reduction: {(1 - new_mem / old_total_mem) * 100:.1f}%")
    print()

    return True


if __name__ == "__main__":
    success = test_correctness()
    exit(0 if success else 1)
