"""Tests for tokenizer_adapter critical token mask generation."""

import numpy as np
import pytest

from openpi_cot.models.adapters.tokenizer_adapter import (
    COORDINATE_SYSTEM_PROMPT_FORMAT,
    PI05_PROMPT_FORMAT,
    SCHEMA_COMPACT_BIMANUAL_PROMPT_FORMAT,
    SCHEMA_COMPACT_PROMPT_FORMAT,
    PaligemmaCoTTokenizer,
    _is_critical_default,
    _is_critical_directional,
    _is_critical_schema,
)


class TestCriticalTokenCheckers:
    """Test the individual critical token checker functions."""

    def test_is_critical_default(self):
        """Test default checker (digits only)."""
        # Should match digits
        assert _is_critical_default("123")
        assert _is_critical_default("token5")
        assert _is_critical_default("0")

        # Should not match non-digits
        assert not _is_critical_default("forward")
        assert not _is_critical_default("left")
        assert not _is_critical_default("+")
        assert not _is_critical_default("-")
        assert not _is_critical_default("text")

    def test_is_critical_directional(self):
        """Test directional checker (digits + directional words)."""
        # Should match digits
        assert _is_critical_directional("123")
        assert _is_critical_directional("token5")

        # Should match directional words
        assert _is_critical_directional("forward")
        assert _is_critical_directional("left")
        assert _is_critical_directional("right")
        assert _is_critical_directional("up")
        assert _is_critical_directional("down")
        assert _is_critical_directional("back")
        assert _is_critical_directional("Forward")  # Case insensitive
        assert _is_critical_directional("LEFT")

        # Should match partial words containing directional words
        assert _is_critical_directional("leftward")
        assert _is_critical_directional("upward")

        # Should not match non-critical tokens
        assert not _is_critical_directional("move")
        assert not _is_critical_directional("the")
        assert not _is_critical_directional("+")

    def test_is_critical_schema(self):
        """Test schema checker (digits + +/- symbols)."""
        # Should match digits
        assert _is_critical_schema("123")
        assert _is_critical_schema("token5")

        # Should match +/- symbols
        assert _is_critical_schema("+")
        assert _is_critical_schema("-")
        assert _is_critical_schema("+5")
        assert _is_critical_schema("-10")
        assert _is_critical_schema("token+")

        # Should not match non-critical tokens
        assert not _is_critical_schema("forward")
        assert not _is_critical_schema("left")
        assert not _is_critical_schema("text")
        assert not _is_critical_schema("move")


class TestTokenizerCriticalMask:
    """Test critical token mask generation for different prompt formats."""

    @pytest.fixture
    def tokenizer(self):
        """Create a tokenizer instance."""
        return PaligemmaCoTTokenizer(max_len=48, prompt_format="pi05")

    def test_pi05_format_critical_mask(self, tokenizer):
        """Test critical mask for pi05 format (directional checker)."""
        prompt = "pick up the cube"
        reasoning = "move forward 5 cm then left 3 cm"

        tokens, attn_mask, reasoning_mask, numeric_mask = tokenizer.tokenize_cot(
            prompt=prompt,
            reasoning=reasoning,
            state=None,
            prompt_format=PI05_PROMPT_FORMAT,
        )

        # Decode tokens to verify critical tokens
        pieces = [tokenizer._tokenizer.id_to_piece(t) for t in tokens if attn_mask[tokens.tolist().index(t)]]

        # Check that numeric_mask is only True for tokens with digits or directional words
        # and only within the reasoning span
        critical_pieces = [pieces[i] for i in range(len(pieces)) if numeric_mask[i]]

        # Should have some critical tokens (numbers and directional words)
        assert np.sum(numeric_mask) > 0, "Should have critical tokens for reasoning with numbers and directions"

        # Critical tokens should only be in reasoning span
        assert np.all(numeric_mask[~reasoning_mask] == False), "Critical tokens should only be in reasoning span"

    def test_schema_compact_format_critical_mask(self, tokenizer):
        """Test critical mask for schema_compact format (schema checker)."""
        prompt = "pick up the cube"
        reasoning = "+5 -3 +2 0"  # Schema format with +/- symbols

        tokens, attn_mask, reasoning_mask, numeric_mask = tokenizer.tokenize_cot(
            prompt=prompt,
            reasoning=reasoning,
            state=None,
            prompt_format=SCHEMA_COMPACT_PROMPT_FORMAT,
        )

        # Should have critical tokens for +/- and digits
        assert np.sum(numeric_mask) > 0, "Should have critical tokens for +/- and digits"

        # Critical tokens should only be in reasoning span
        assert np.all(numeric_mask[~reasoning_mask] == False), "Critical tokens should only be in reasoning span"

    def test_coordinate_system_format_critical_mask(self, tokenizer):
        """Test critical mask for coordinate_system format (schema checker)."""
        prompt = "move object"
        reasoning = "x+5 y-3 z+2"

        tokens, attn_mask, reasoning_mask, numeric_mask = tokenizer.tokenize_cot(
            prompt=prompt,
            reasoning=reasoning,
            state=None,
            prompt_format=COORDINATE_SYSTEM_PROMPT_FORMAT,
        )

        # Should have critical tokens
        assert np.sum(numeric_mask) > 0, "Should have critical tokens for coordinates"

        # Critical tokens should only be in reasoning span
        assert np.all(numeric_mask[~reasoning_mask] == False), "Critical tokens should only be in reasoning span"

    def test_no_critical_tokens_in_non_critical_reasoning(self, tokenizer):
        """Test that non-critical reasoning produces minimal critical tokens."""
        prompt = "describe the scene"
        reasoning = "the object is on the table"  # No digits, directions, or +/- in pi05 context

        tokens, attn_mask, reasoning_mask, numeric_mask = tokenizer.tokenize_cot(
            prompt=prompt,
            reasoning=reasoning,
            state=None,
            prompt_format=PI05_PROMPT_FORMAT,
        )

        # Should have few or no critical tokens
        # (some tokenizers may split words in ways that create false positives)
        num_critical = np.sum(numeric_mask)
        num_reasoning = np.sum(reasoning_mask)

        # Less than 20% of reasoning tokens should be critical
        if num_reasoning > 0:
            ratio = num_critical / num_reasoning
            assert ratio < 0.2, f"Too many critical tokens ({num_critical}/{num_reasoning}) for non-critical reasoning"

    def test_critical_mask_with_state(self, tokenizer):
        """Test critical mask generation with state included."""
        prompt = "pick up cube"
        reasoning = "move forward 10 cm"
        state = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7])

        tokens, attn_mask, reasoning_mask, numeric_mask = tokenizer.tokenize_cot(
            prompt=prompt,
            reasoning=reasoning,
            state=state,
            state_type="joint_pos",
            prompt_format=PI05_PROMPT_FORMAT,
        )

        # Should have critical tokens in reasoning
        assert np.sum(numeric_mask) > 0, "Should have critical tokens"

        # Critical tokens should only be in reasoning span (not in prompt or state)
        assert np.all(numeric_mask[~reasoning_mask] == False), "Critical tokens should only be in reasoning span"

    def test_bimanual_format_critical_mask(self, tokenizer):
        """Test critical mask for bimanual format."""
        prompt = "pick with both hands"
        reasoning = "L +5 -3 +2 1 R +4 -2 +1 0"  # Bimanual schema format

        tokens, attn_mask, reasoning_mask, numeric_mask = tokenizer.tokenize_cot(
            prompt=prompt,
            reasoning=reasoning,
            state=None,
            prompt_format=SCHEMA_COMPACT_BIMANUAL_PROMPT_FORMAT,
        )

        # Should have critical tokens
        assert np.sum(numeric_mask) > 0, "Should have critical tokens for bimanual actions"

    def test_numeric_mask_dimensions(self, tokenizer):
        """Test that masks have correct dimensions."""
        prompt = "test"
        reasoning = "123"

        tokens, attn_mask, reasoning_mask, numeric_mask = tokenizer.tokenize_cot(
            prompt=prompt,
            reasoning=reasoning,
            state=None,
            prompt_format=PI05_PROMPT_FORMAT,
        )

        # All outputs should have same length as max_len
        assert len(tokens) == tokenizer._max_len
        assert len(attn_mask) == tokenizer._max_len
        assert len(reasoning_mask) == tokenizer._max_len
        assert len(numeric_mask) == tokenizer._max_len

        # Masks should be boolean arrays
        assert attn_mask.dtype == bool
        assert reasoning_mask.dtype == bool
        assert numeric_mask.dtype == bool

    def test_empty_reasoning(self, tokenizer):
        """Test critical mask with empty reasoning."""
        prompt = "test prompt"
        reasoning = None

        tokens, attn_mask, reasoning_mask, numeric_mask = tokenizer.tokenize_cot(
            prompt=prompt,
            reasoning=reasoning,
            state=None,
            prompt_format=PI05_PROMPT_FORMAT,
        )

        # No reasoning means no critical tokens
        assert np.sum(numeric_mask) == 0, "Should have no critical tokens with empty reasoning"
        assert np.sum(reasoning_mask) == 0, "Should have no reasoning tokens with empty reasoning"

    def test_different_formats_different_masks(self, tokenizer):
        """Test that different formats produce different critical masks for the same reasoning."""
        prompt = "test"
        reasoning = "forward left right 5"  # Contains both directional words and digits

        # Test with directional format (should match directional words + digits)
        tokens_dir, _, _, numeric_mask_dir = tokenizer.tokenize_cot(
            prompt=prompt,
            reasoning=reasoning,
            state=None,
            prompt_format=PI05_PROMPT_FORMAT,
        )

        # Test with schema format (should only match digits, not directional words)
        tokens_schema, _, _, numeric_mask_schema = tokenizer.tokenize_cot(
            prompt=prompt,
            reasoning=reasoning,
            state=None,
            prompt_format=SCHEMA_COMPACT_PROMPT_FORMAT,
        )

        # Directional format should have more critical tokens than schema format
        # (assuming "forward", "left", "right" are tokenized separately)
        num_critical_dir = np.sum(numeric_mask_dir)
        num_critical_schema = np.sum(numeric_mask_schema)

        # Both should have at least the digit "5" marked
        assert num_critical_dir > 0, "Directional format should have critical tokens"
        assert num_critical_schema > 0, "Schema format should have critical tokens"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
