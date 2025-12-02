"""Label smoothing utilities for number tokens.

This module provides functionality to create truncated Gaussian distributions
over digit tokens for label smoothing during training.
"""

import jax.numpy as jnp
import numpy as np
from scipy import stats


def create_digit_smoothing_kernel(
    vocab_size: int,
    digit_to_token_id: dict[int, int],
    sigma: float = 1.0,
    support: int = 3,
) -> jnp.ndarray:
    """Create a smoothing kernel for digit tokens using truncated Gaussian.

    For each digit 0-9, creates a probability distribution over nearby digits.
    The distribution is centered on the digit itself and decreases according
    to a Gaussian function for neighboring digits.

    Special handling at boundaries:
    - Digit 0: Only smooths towards 1, 2, 3, ... (not negative)
    - Digit 9: Only smooths towards 8, 7, 6, ... (not beyond 9)

    Args:
        vocab_size: Size of the vocabulary
        digit_to_token_id: Mapping from digit value (0-9) to token ID
        sigma: Standard deviation of the Gaussian distribution.
               Smaller values = more concentrated around the target digit.
               Larger values = more spread to neighbors.
               Typical values: 0.5 (very concentrated) to 2.0 (very spread).
        support: Maximum distance from center digit to include in smoothing.
                 For support=3, digit 5 will have non-zero probability for
                 digits [2, 3, 4, 5, 6, 7, 8] (i.e., ±3 positions).

    Returns:
        Smoothing kernel of shape [10, vocab_size] where kernel[digit] is
        the smoothed probability distribution for that digit.

    Example:
        >>> digit_to_token = {0: 1000, 1: 1001, ..., 9: 1009}
        >>> kernel = create_digit_smoothing_kernel(
        ...     vocab_size=32000,
        ...     digit_to_token_id=digit_to_token,
        ...     sigma=1.0,
        ...     support=3
        ... )
        >>> # kernel[5] will have high prob at token 1005 (digit 5),
        >>> # lower prob at tokens 1004, 1006 (digits 4, 6),
        >>> # even lower at tokens 1003, 1007 (digits 3, 7), etc.
    """
    # Initialize kernel with zeros
    kernel = np.zeros((10, vocab_size), dtype=np.float32)

    for digit in range(10):
        # Create probability weights for nearby digits using Gaussian
        probs = {}
        for offset in range(-support, support + 1):
            neighbor = digit + offset

            # Only include valid digits [0, 9]
            if 0 <= neighbor <= 9:
                # Exclude boundary edge cases to prevent smoothing toward extremes
                # Digit 1 should not smooth to 0, digit 8 should not smooth to 9
                if (digit == 1 and neighbor == 0) or (digit == 8 and neighbor == 9):
                    continue
                # Gaussian weight based on distance from center
                weight = stats.norm.pdf(offset, loc=0, scale=sigma)
                probs[neighbor] = weight

        # Normalize to ensure it's a valid probability distribution
        total = sum(probs.values())
        probs = {k: v / total for k, v in probs.items()}

        # Fill kernel row with probabilities
        for neighbor, prob in probs.items():
            token_id = digit_to_token_id[neighbor]
            kernel[digit, token_id] = prob

    # Convert to JAX array
    return jnp.asarray(kernel, dtype=jnp.float32)


def get_digit_to_token_mapping() -> dict[int, int]:
    """Get the mapping from digit values (0-9) to token IDs for PaliGemma tokenizer.

    This mapping was determined empirically by tokenizing each digit individually.

    Returns:
        Dictionary mapping digit (0-9) to its token ID in the PaliGemma vocabulary.
    """
    return {
        0: 235276,
        1: 235274,
        2: 235284,
        3: 235304,
        4: 235310,
        5: 235308,
        6: 235318,
        7: 235324,
        8: 235321,
        9: 235315,
    }


def visualize_smoothing_kernel(kernel: np.ndarray, digit: int) -> None:
    """Visualize the smoothing distribution for a specific digit.

    Useful for debugging and understanding the smoothing behavior.

    Args:
        kernel: Smoothing kernel [10, vocab_size]
        digit: Digit to visualize (0-9)
    """
    digit_to_token = get_digit_to_token_mapping()
    token_to_digit = {v: k for k, v in digit_to_token.items()}

    print(f"\nSmoothing distribution for digit {digit}:")
    print("=" * 50)

    # Get non-zero probabilities
    non_zero_idx = np.where(kernel[digit] > 0)[0]

    # Sort by probability (descending)
    sorted_idx = non_zero_idx[np.argsort(-kernel[digit, non_zero_idx])]

    for token_id in sorted_idx:
        prob = kernel[digit, token_id]
        target_digit = token_to_digit.get(token_id, "?")
        bar = "#" * int(prob * 50)  # Scale to 50 chars max
        print(f"  Digit {target_digit}: {prob:6.4f} {bar}")

    print(f"Total probability: {kernel[digit].sum():.6f}")


if __name__ == "__main__":
    # Demo: Create and visualize smoothing kernels
    print("="*80)
    print("LABEL SMOOTHING KERNEL DEMO")
    print("="*80)

    digit_to_token = get_digit_to_token_mapping()
    vocab_size = 256000  # Approximate vocab size for PaliGemma

    # Test different sigma values
    for sigma in [0.5, 1.0, 1.5]:
        print(f"\n{'='*80}")
        print(f"Sigma = {sigma}")
        print('='*80)

        kernel = create_digit_smoothing_kernel(
            vocab_size=vocab_size,
            digit_to_token_id=digit_to_token,
            sigma=sigma,
            support=3,
        )

        # Visualize a few examples
        for digit in [0, 3, 5, 9]:
            visualize_smoothing_kernel(kernel, digit)

    # Test boundary cases
    print(f"\n{'='*80}")
    print("BOUNDARY CASES (sigma=1.0)")
    print('='*80)

    kernel = create_digit_smoothing_kernel(
        vocab_size=vocab_size,
        digit_to_token_id=digit_to_token,
        sigma=1.0,
        support=3,
    )

    print("\nDigit 0 (no negative neighbors):")
    visualize_smoothing_kernel(kernel, 0)

    print("\nDigit 9 (no neighbors above 9):")
    visualize_smoothing_kernel(kernel, 9)
