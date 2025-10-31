"""Test to visualize attention mask for action loss computation.

This test verifies that action tokens cannot attend to langact (language/reasoning) tokens
during training, which is critical for proper action diffusion training.
"""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import pytest
from unittest.mock import patch

from openpi_cot.models.adapters.model_adapter import CoTObservation
from openpi_cot.models.pi_cot_config import PiCoTConfig
from openpi_cot.models.pi_cot import PiCoT


def create_realistic_observation(
    batch_size: int = 2,
    num_frames: int = 4,
    seq_len: int = 32,
    langact_positions: list[tuple[int, int]] = None,
    use_gemma3: bool = False,
) -> CoTObservation:
    """Create a realistic observation for testing.

    Args:
        batch_size: Number of examples in batch
        num_frames: Number of video frames per image key
        seq_len: Length of tokenized sequence
        langact_positions: List of (start, end) positions marking langact tokens.
                          E.g., [(10, 20)] means tokens 10-20 are langact tokens.
        use_gemma3: Whether to use Gemma3 format (with placeholders)

    Returns:
        CoTObservation with realistic structure
    """
    if langact_positions is None:
        # Default: middle third of sequence is langact (reasoning/language)
        langact_positions = [(seq_len // 3, 2 * seq_len // 3)]

    # Create images for 2 camera views
    image_keys = ("base_0_rgb", "left_wrist_0_rgb")
    images = {}
    image_masks = {}

    for key in image_keys:
        # Images: [batch, time, height, width, channels]
        images[key] = jnp.ones((batch_size, num_frames, 224, 224, 3), dtype=jnp.float32) * 0.5
        image_masks[key] = jnp.ones((batch_size,), dtype=bool)

    # Create tokenized prompt
    if use_gemma3:
        # Gemma3: includes -2 placeholders for images
        # Assume first 512 tokens are image placeholders (2 cameras * 256 patches)
        num_img_tokens = 2 * 256
        tokenized_prompt = jnp.concatenate([
            jnp.full((batch_size, num_img_tokens), -2, dtype=jnp.int32),  # Placeholders
            jnp.arange(100, 100 + seq_len, dtype=jnp.int32)[None, :].repeat(batch_size, axis=0),  # Text tokens
        ], axis=1)
        total_len = num_img_tokens + seq_len

        # Adjust langact positions to account for image placeholders
        langact_positions_adjusted = [(start + num_img_tokens, end + num_img_tokens)
                                       for start, end in langact_positions]
    else:
        # Legacy: text only (images concatenated separately)
        tokenized_prompt = jnp.arange(100, 100 + seq_len, dtype=jnp.int32)[None, :].repeat(batch_size, axis=0)
        total_len = seq_len
        langact_positions_adjusted = langact_positions

    # Create masks
    tokenized_prompt_mask = jnp.ones((batch_size, total_len), dtype=bool)

    # Create langact mask: True for langact tokens, False for others
    tokenized_langact_mask = jnp.zeros((batch_size, total_len), dtype=bool)
    for start, end in langact_positions_adjusted:
        tokenized_langact_mask = tokenized_langact_mask.at[:, start:end].set(True)

    # Create state
    state = jnp.zeros((batch_size, 7), dtype=jnp.float32)

    observation = CoTObservation(
        images=images,
        image_masks=image_masks,
        state=state,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
        tokenized_langact_mask=tokenized_langact_mask,
        crictical_token_mask=jnp.zeros((batch_size, total_len), dtype=bool),
        number_token_mask=None,
        direction_token_mask=None,
        sample_mask=None,
    )

    return observation


def visualize_attention_mask(
    attn_mask: jnp.ndarray,
    langact_mask: jnp.ndarray,
    prefix_len: int,
    suffix_len: int,
    save_path: str = "/tmp/attention_mask_visualization.png",
    model_name: str = "PiCoT",
):
    """Visualize the attention mask to verify langact masking.

    Args:
        attn_mask: Attention mask [batch, query_len, key_len]
        langact_mask: Langact mask indicating which positions are langact tokens
        prefix_len: Length of prefix (images + text)
        suffix_len: Length of suffix (action tokens)
        save_path: Where to save the visualization
        model_name: Name of the model variant being tested
    """
    # Take first example in batch
    if attn_mask.ndim == 3:
        attn_mask = attn_mask[0]  # [query_len, key_len]

    total_len = attn_mask.shape[1]

    # Convert to numpy for visualization
    attn_mask_np = np.array(attn_mask, dtype=float)
    langact_mask_np = np.array(langact_mask[0], dtype=bool)

    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))

    # === Subplot 1: Full attention mask ===
    ax1 = axes[0, 0]
    im1 = ax1.imshow(attn_mask_np, cmap='Blues', aspect='auto', interpolation='nearest')
    ax1.set_title(f'{model_name}: Full Attention Mask\n(White=Can Attend, Black=Masked)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Key Position (what tokens can be attended to)', fontsize=10)
    ax1.set_ylabel('Query Position (tokens doing the attending)', fontsize=10)

    # Add vertical lines to separate regions
    ax1.axvline(x=prefix_len - 0.5, color='red', linestyle='--', linewidth=2, label='Prefix/Suffix boundary')
    ax1.axhline(y=prefix_len - 0.5, color='red', linestyle='--', linewidth=2)

    # Mark langact regions
    langact_starts = np.where(np.diff(langact_mask_np.astype(int), prepend=0) == 1)[0]
    langact_ends = np.where(np.diff(langact_mask_np.astype(int), append=0) == -1)[0]
    for start, end in zip(langact_starts, langact_ends):
        ax1.axvspan(start - 0.5, end + 0.5, alpha=0.3, color='orange', label='Langact tokens' if start == langact_starts[0] else '')

    ax1.legend(loc='upper right', fontsize=8)
    plt.colorbar(im1, ax=ax1, label='Attention (1=allowed, 0=masked)')
    ax1.grid(True, alpha=0.3)

    # === Subplot 2: Focus on suffix attending to prefix ===
    ax2 = axes[0, 1]
    # Extract suffix queries attending to all keys
    suffix_to_all = attn_mask_np[prefix_len:, :]
    im2 = ax2.imshow(suffix_to_all, cmap='Blues', aspect='auto', interpolation='nearest')
    ax2.set_title(f'Action Tokens Attention to Prefix+Suffix\n(Critical: Actions should NOT see langact)', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Key Position', fontsize=10)
    ax2.set_ylabel('Action Token (Query Position)', fontsize=10)

    # Mark langact regions
    for start, end in zip(langact_starts, langact_ends):
        ax2.axvspan(start - 0.5, end + 0.5, alpha=0.3, color='orange')

    ax2.axvline(x=prefix_len - 0.5, color='red', linestyle='--', linewidth=2)
    plt.colorbar(im2, ax=ax2, label='Attention (1=allowed, 0=masked)')
    ax2.grid(True, alpha=0.3)

    # === Subplot 3: Heatmap showing action attention to langact tokens ===
    ax3 = axes[1, 0]
    # Check which action tokens can see langact tokens
    action_sees_langact = suffix_to_all[:, langact_mask_np]
    num_langact = langact_mask_np.sum()

    if num_langact > 0:
        im3 = ax3.imshow(action_sees_langact, cmap='RdYlGn_r', aspect='auto', interpolation='nearest', vmin=0, vmax=1)
        ax3.set_title(f'VERIFICATION: Action→Langact Attention\n(Should be ALL BLACK if correctly masked)', fontsize=12, fontweight='bold')
        ax3.set_xlabel(f'Langact Token Index (out of {num_langact} total)', fontsize=10)
        ax3.set_ylabel('Action Token (Query Position)', fontsize=10)
        plt.colorbar(im3, ax=ax3, label='Attention (0=✓correct, 1=✗leaked)')
        ax3.grid(True, alpha=0.3)

        # Add text showing max attention value
        max_attn = action_sees_langact.max()
        status = "✓ PASS" if max_attn == 0 else "✗ FAIL"
        color = 'green' if max_attn == 0 else 'red'
        ax3.text(0.5, 1.05, f'Max Attention: {max_attn:.4f} - {status}',
                transform=ax3.transAxes, ha='center', fontsize=11,
                fontweight='bold', color=color, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    else:
        ax3.text(0.5, 0.5, 'No langact tokens in sequence', ha='center', va='center', fontsize=12)
        ax3.set_title('VERIFICATION: Action→Langact Attention', fontsize=12, fontweight='bold')

    # === Subplot 4: Summary statistics ===
    ax4 = axes[1, 1]
    ax4.axis('off')

    # Compute statistics
    total_queries = attn_mask_np.shape[0]
    total_keys = attn_mask_np.shape[1]
    suffix_queries = suffix_to_all.shape[0]

    # Check if action tokens can see langact tokens
    langact_violation = action_sees_langact.max() if num_langact > 0 else 0.0
    langact_count = num_langact

    # Check if action tokens can see non-langact prefix tokens
    non_langact_mask = ~langact_mask_np[:prefix_len]
    action_sees_nonlangact = suffix_to_all[:, :prefix_len][:, non_langact_mask]
    nonlangact_visible = action_sees_nonlangact.mean() if non_langact_mask.sum() > 0 else 0.0

    summary_text = f"""
    === ATTENTION MASK SUMMARY ({model_name}) ===

    Sequence Structure:
    • Total length: {total_len} tokens
    • Prefix length: {prefix_len} tokens
      - Non-langact: {prefix_len - langact_count} tokens
      - Langact: {langact_count} tokens
    • Suffix (actions): {suffix_len} tokens

    Verification Results:
    ✓ Actions see non-langact prefix: {nonlangact_visible:.1%} visible
      (Should be ~100% - these are OK to attend to)

    {'✓' if langact_violation == 0 else '✗'} Actions see langact tokens: {langact_violation:.4f} max attention
      (Should be 0.0 - langact must be masked!)

    {'✓ TEST PASSED' if langact_violation == 0 else '✗ TEST FAILED'}

    Details:
    • Langact token positions: {list(zip(langact_starts, langact_ends))}
    • Action query positions: [{prefix_len}, {total_len})
    """

    ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=10,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")

    # Also return the verification result
    return langact_violation == 0.0


class TestActionAttentionMask:
    """Test suite for verifying action attention masking."""

    @pytest.mark.parametrize("use_gemma3", [False, True])
    def test_action_cannot_attend_to_langact(self, use_gemma3: bool):
        """Test that action tokens are properly masked from attending to langact tokens.

        This is the core test that verifies the fix in _compute_action_loss.
        """
        # Create observation with clear langact regions
        seq_len = 48
        langact_start, langact_end = 16, 32  # Middle third is langact

        observation = create_realistic_observation(
            batch_size=2,
            num_frames=1,  # Action loss uses first frame only
            seq_len=seq_len,
            langact_positions=[(langact_start, langact_end)],
            use_gemma3=use_gemma3,
        )

        # Create model config
        variant = "gemma3_1b" if use_gemma3 else "gemma_2b"
        config = PiCoTConfig(
            paligemma_variant=variant,
            action_expert_variant=variant,
            action_dim=7,
            action_horizon=4,
            max_token_len=seq_len,
            enable_action_training=True,
        )

        # Create fake actions
        actions = jnp.zeros((2, 4, 7), dtype=jnp.float32)

        # Variable to capture the attention mask
        captured_attn_mask = {}

        # Patch make_attn_mask to capture the mask
        original_make_attn_mask = None
        try:
            import openpi.models.pi0 as _pi0
            original_make_attn_mask = _pi0.make_attn_mask

            def capture_attn_mask(input_mask, ar_mask):
                result = original_make_attn_mask(input_mask, ar_mask)
                captured_attn_mask['mask'] = result
                captured_attn_mask['input_mask'] = input_mask
                captured_attn_mask['ar_mask'] = ar_mask
                return result

            _pi0.make_attn_mask = capture_attn_mask

            # Initialize model and compute action loss
            rng = jax.random.PRNGKey(42)
            model = config.create(rng)

            # Compute loss (this will trigger the patched make_attn_mask)
            try:
                loss, metrics = model.compute_loss(
                    rng,
                    observation,
                    actions,
                    train=True,
                )
            except Exception as e:
                # Model may fail due to incomplete initialization, but we should have captured the mask
                if 'mask' not in captured_attn_mask:
                    raise RuntimeError(f"Failed to capture attention mask: {e}") from e
                print(f"Note: Model forward pass failed (expected for test), but mask was captured: {e}")

        finally:
            # Restore original function
            if original_make_attn_mask is not None:
                _pi0.make_attn_mask = original_make_attn_mask

        # Verify we captured the mask
        assert 'mask' in captured_attn_mask, "Failed to capture attention mask"

        attn_mask = captured_attn_mask['mask']
        input_mask = captured_attn_mask['input_mask']

        # Determine prefix and suffix lengths
        total_len = input_mask.shape[1]
        prefix_len = observation.tokenized_prompt.shape[1]
        suffix_len = total_len - prefix_len

        # Visualize the mask
        model_name = "Gemma3" if use_gemma3 else "Legacy"
        save_path = f"/tmp/attention_mask_{model_name.lower()}.png"

        # Get langact mask for visualization
        if use_gemma3:
            # For Gemma3, langact mask corresponds directly to tokenized_prompt
            langact_mask_full = observation.tokenized_langact_mask
        else:
            # For legacy, need to account for image tokens prepended
            # Compute expected image token count: 2 cameras * 1 frame * 256 patches = 512
            img_tokens = 2 * 256
            langact_mask_full = jnp.concatenate([
                jnp.zeros((observation.tokenized_langact_mask.shape[0], img_tokens), dtype=bool),
                observation.tokenized_langact_mask,
            ], axis=1)

        test_passed = visualize_attention_mask(
            attn_mask,
            langact_mask_full,
            prefix_len,
            suffix_len,
            save_path=save_path,
            model_name=model_name,
        )

        # Extract action queries (suffix) and check they can't attend to langact keys
        action_queries_start = prefix_len
        action_attn = attn_mask[0, action_queries_start:, :]  # [suffix_len, total_len]

        # Check attention to langact positions
        langact_positions = langact_mask_full[0]  # [total_len]
        action_to_langact = action_attn[:, langact_positions]  # [suffix_len, num_langact]

        # Verify: should be all zeros (no attention allowed)
        max_attention_to_langact = float(action_to_langact.max())

        print(f"\n{'='*60}")
        print(f"Test Results ({model_name}):")
        print(f"{'='*60}")
        print(f"Sequence length: {total_len}")
        print(f"Prefix length: {prefix_len}")
        print(f"Suffix (action) length: {suffix_len}")
        print(f"Langact token count: {langact_positions.sum()}")
        print(f"Max attention from actions to langact: {max_attention_to_langact:.6f}")
        print(f"Status: {'✓ PASS' if test_passed else '✗ FAIL'}")
        print(f"{'='*60}\n")

        # Assert the test passes
        assert max_attention_to_langact == 0.0, (
            f"Action tokens can attend to langact tokens! "
            f"Max attention: {max_attention_to_langact}. "
            f"Check visualization at {save_path}"
        )

    def test_visualization_only_simple(self):
        """Simple test that just creates a visualization without full model initialization.

        This is useful for quick visual inspection of the masking logic.
        """
        # Create simple observation
        observation = create_realistic_observation(
            batch_size=1,
            num_frames=1,
            seq_len=32,
            langact_positions=[(10, 20)],
            use_gemma3=False,
        )

        # Manually create attention mask to visualize the concept
        # Prefix: [img_tokens (512), text_tokens (32)] = 544 total
        # Suffix: [action_tokens (20)] = 20 total
        img_tokens = 2 * 256  # 2 cameras * 256 patches
        text_tokens = 32
        action_tokens = 20

        prefix_len = img_tokens + text_tokens
        total_len = prefix_len + action_tokens

        # Create input_mask and ar_mask
        input_mask = jnp.ones((1, total_len), dtype=bool)
        ar_mask = jnp.concatenate([
            jnp.zeros((1, prefix_len), dtype=bool),  # Prefix: no AR masking
            jnp.ones((1, action_tokens), dtype=bool),  # Suffix: AR masking
        ], axis=1)

        # Create langact mask (positions 512+10 to 512+20 are langact)
        langact_mask = jnp.zeros((1, total_len), dtype=bool)
        langact_mask = langact_mask.at[:, img_tokens + 10:img_tokens + 20].set(True)

        # Manually compute attention mask with langact masking
        # Simulate what the fixed code should do
        prefix_mask_action = jnp.logical_and(
            input_mask[:, :prefix_len],
            jnp.logical_not(langact_mask[:, :prefix_len])
        )

        # Build full input mask for attention
        input_mask_with_langact_masked = jnp.concatenate([
            prefix_mask_action,
            input_mask[:, prefix_len:],  # Suffix unchanged
        ], axis=1)

        # Create causal attention mask
        import openpi.models.pi0 as _pi0
        attn_mask = _pi0.make_attn_mask(input_mask_with_langact_masked, ar_mask)

        # Visualize
        test_passed = visualize_attention_mask(
            attn_mask,
            langact_mask,
            prefix_len,
            action_tokens,
            save_path="/tmp/attention_mask_manual_simple.png",
            model_name="Manual (Fixed Logic)",
        )

        print("\nSimple visualization created at: /tmp/attention_mask_manual_simple.png")
        assert test_passed, "Manual test should pass with correct masking logic"


if __name__ == "__main__":
    # Run tests
    print("Running attention mask visualization tests...")
    print("=" * 80)

    # Run simple visualization first
    test = TestActionAttentionMask()
    print("\n1. Running simple visualization test...")
    test.test_visualization_only_simple()

    # Run full tests with both model variants
    print("\n2. Running full test with legacy model...")
    test.test_action_cannot_attend_to_langact(use_gemma3=False)

    print("\n3. Running full test with Gemma3 model...")
    test.test_action_cannot_attend_to_langact(use_gemma3=True)

    print("\n" + "=" * 80)
    print("All tests completed! Check /tmp/ for visualizations.")
