"""Unit tests for VLA-0 action format.

Tests the serialization (summarize) and deserialization (parse) of actions
in VLA-0 format, which represents normalized actions as discretized integers.
"""

import numpy as np
import pytest

from openpi_cot.policies.lang_action_formats import (
    VLA0ActionFormat,
    VLA0_FORMAT,
    VLA0_CHUNKED_FORMAT,
    get_language_action_format,
)


class TestVLA0ActionFormat:
    """Test VLA0ActionFormat class."""

    def test_format_registered(self):
        """Test that VLA0 formats are registered in the registry."""
        vla0 = get_language_action_format("vla0")
        assert vla0 is not None
        assert vla0.style == "vla0"
        assert vla0.action_horizon == 1

        vla0_chunked = get_language_action_format("vla0_chunked")
        assert vla0_chunked is not None
        assert vla0_chunked.style == "vla0"
        assert vla0_chunked.action_horizon == 10  # VLA0 chunked uses 10-step horizon

    def test_summarize_single_step(self):
        """Test summarizing single-step actions."""
        fmt = VLA0ActionFormat(num_bins=1000, action_horizon=1, action_dim=7)

        # Test with zeros (should map to 500)
        actions = np.zeros((1, 7))
        result = fmt.summarize_actions(actions)
        assert result == "500 500 500 500 500 500 500"

        # Test with -1 (should map to 0)
        actions = np.full((1, 7), -1.0)
        result = fmt.summarize_actions(actions)
        assert result == "0 0 0 0 0 0 0"

        # Test with +1 (should map to 1000)
        actions = np.full((1, 7), 1.0)
        result = fmt.summarize_actions(actions)
        assert result == "1000 1000 1000 1000 1000 1000 1000"

    def test_summarize_multi_step(self):
        """Test summarizing multi-step actions."""
        fmt = VLA0ActionFormat(num_bins=1000, action_horizon=2, action_dim=7)

        # Test with zeros (should produce 14 values all at 500)
        actions = np.zeros((2, 7))
        result = fmt.summarize_actions(actions)
        expected = " ".join(["500"] * 14)
        assert result == expected

    def test_summarize_1d_input(self):
        """Test that 1D input is handled correctly."""
        fmt = VLA0ActionFormat(num_bins=1000, action_horizon=1, action_dim=7)

        # 1D input should be treated as single timestep
        actions = np.zeros(7)
        result = fmt.summarize_actions(actions)
        assert result == "500 500 500 500 500 500 500"

    def test_parse_single_step(self):
        """Test parsing single-step VLA0 format."""
        fmt = VLA0ActionFormat(num_bins=1000, action_horizon=1, action_dim=7)

        # Parse center values (500 -> 0.0)
        text = "500 500 500 500 500 500 500"
        actions = fmt.parse_to_full_actions(text)
        assert actions.shape == (1, 7)
        np.testing.assert_array_almost_equal(actions, np.zeros((1, 7)), decimal=2)

        # Parse min values (0 -> -1.0)
        text = "0 0 0 0 0 0 0"
        actions = fmt.parse_to_full_actions(text)
        np.testing.assert_array_almost_equal(actions, np.full((1, 7), -1.0), decimal=2)

        # Parse max values (1000 -> 1.0)
        text = "1000 1000 1000 1000 1000 1000 1000"
        actions = fmt.parse_to_full_actions(text)
        np.testing.assert_array_almost_equal(actions, np.full((1, 7), 1.0), decimal=2)

    def test_parse_multi_step(self):
        """Test parsing multi-step VLA0 format."""
        fmt = VLA0ActionFormat(num_bins=1000, action_horizon=2, action_dim=7)

        # Parse 14 values (2 timesteps x 7 dims)
        text = " ".join(["500"] * 14)
        actions = fmt.parse_to_full_actions(text)
        assert actions.shape == (2, 7)
        np.testing.assert_array_almost_equal(actions, np.zeros((2, 7)), decimal=2)

    def test_roundtrip_single_step(self):
        """Test that summarize -> parse is a near-identity (within discretization error)."""
        fmt = VLA0ActionFormat(num_bins=1000, action_horizon=1, action_dim=7)

        # Random actions in [-1, 1]
        np.random.seed(42)
        original = np.random.uniform(-1, 1, (1, 7))

        # Roundtrip
        text = fmt.summarize_actions(original)
        recovered = fmt.parse_to_full_actions(text)

        # Should be close (within 1/1000 = 0.002 due to discretization)
        np.testing.assert_array_almost_equal(original, recovered, decimal=2)

    def test_roundtrip_multi_step(self):
        """Test roundtrip for multi-step actions."""
        fmt = VLA0ActionFormat(num_bins=1000, action_horizon=16, action_dim=7)

        # Random actions
        np.random.seed(42)
        original = np.random.uniform(-1, 1, (16, 7))

        # Roundtrip
        text = fmt.summarize_actions(original)
        recovered = fmt.parse_to_full_actions(text)

        # Should be close
        np.testing.assert_array_almost_equal(original, recovered, decimal=2)

    def test_clipping(self):
        """Test that out-of-range values are clipped."""
        fmt = VLA0ActionFormat(num_bins=1000, action_horizon=1, action_dim=7)

        # Values outside [-1, 1] should be clipped
        actions = np.array([[2.0, -2.0, 0.5, 0.0, 0.0, 0.0, 0.0]])
        text = fmt.summarize_actions(actions)

        # 2.0 clips to 1.0 -> 1000, -2.0 clips to -1.0 -> 0
        assert text.startswith("1000 0")

    def test_parse_invalid_format(self):
        """Test parsing invalid format returns zeros."""
        fmt = VLA0ActionFormat(num_bins=1000, action_horizon=1, action_dim=7)

        # Empty string
        text = ""
        actions = fmt.parse_to_full_actions(text)
        np.testing.assert_array_equal(actions, np.zeros((1, 7)))

        # Non-numeric string
        text = "abc def"
        actions = fmt.parse_to_full_actions(text)
        np.testing.assert_array_equal(actions, np.zeros((1, 7)))

    def test_parse_language_to_deltas(self):
        """Test the legacy parse_language_to_deltas interface."""
        fmt = VLA0ActionFormat(num_bins=1000, action_horizon=1, action_dim=7)

        text = "500 500 500 500 500 500 750"
        movement, gripper = fmt.parse_language_to_deltas(text)

        # Movement should be first 6 dims
        assert movement.shape == (6,)
        np.testing.assert_array_almost_equal(movement, np.zeros(6), decimal=2)

        # Gripper should be 7th dim (750 -> 0.5)
        assert gripper is not None
        assert abs(gripper - 0.5) < 0.01

    def test_get_sum_decimal(self):
        """Test that get_sum_decimal returns 'vla0'."""
        fmt = VLA0ActionFormat()
        assert fmt.get_sum_decimal() == "vla0"


class TestVLA0Integration:
    """Integration tests for VLA0 format with the policy pipeline."""

    def test_format_selection(self):
        """Test that VLA0 format is correctly selected by name."""
        fmt = get_language_action_format("vla0")
        assert isinstance(fmt, VLA0ActionFormat)
        assert fmt.action_horizon == 1

        fmt = get_language_action_format("vla0_chunked")
        assert isinstance(fmt, VLA0ActionFormat)
        assert fmt.action_horizon == 10  # VLA0 chunked uses 10-step horizon

    def test_predefined_formats(self):
        """Test the predefined VLA0 format instances."""
        assert VLA0_FORMAT.name == "vla0"
        assert VLA0_FORMAT.num_bins == 1000
        assert VLA0_FORMAT.action_horizon == 1
        assert VLA0_FORMAT.action_dim == 7

        assert VLA0_CHUNKED_FORMAT.name == "vla0_chunked"
        assert VLA0_CHUNKED_FORMAT.num_bins == 1000
        assert VLA0_CHUNKED_FORMAT.action_horizon == 10  # VLA0 chunked uses 10-step horizon
        assert VLA0_CHUNKED_FORMAT.action_dim == 7


class TestVLA0Unnormalization:
    """Test VLA0 action unnormalization in CoTOutputs."""

    def test_cot_outputs_unnormalize_vla0(self):
        """Test that CoTOutputs unnormalizes VLA0 actions correctly."""
        from openpi_cot.policies.cot_policy import CoTOutputs
        from openpi.shared.normalize import NormStats

        # Create mock norm_stats with q01/q99 for bounds_q99 normalization
        q01 = np.array([0.0, 0.0, 0.0, -3.14, -3.14, -3.14, 0.0])
        q99 = np.array([1.0, 1.0, 1.0, 3.14, 3.14, 3.14, 1.0])
        actions_stats = NormStats(
            mean=np.zeros(7),
            std=np.ones(7),
            min=q01,
            max=q99,
            q01=q01,
            q99=q99,
        )
        norm_stats = {"actions": actions_stats}

        # Create CoTOutputs with VLA0 format and norm_stats
        cot_outputs = CoTOutputs(
            language_action_format="vla0_chunked",
            norm_stats=norm_stats,
            normalization_type="bounds_q99",
        )

        # Create mock reasoning (VLA0 format: space-separated integers)
        # "500" maps to 0.0 in normalized space, which should unnormalize to middle of q01-q99 range
        reasoning = " ".join(["500"] * 70)  # 10 timesteps * 7 dims
        data = {"reasoning": reasoning}

        result = cot_outputs(data)

        # Actions should be unnormalized
        assert "actions" in result
        actions = result["actions"]
        assert actions.shape == (10, 7)

        # For bins=1000, token 500 -> normalized value = 500/1000 * 2 - 1 = 0.0
        # Unnormalized: (0 + 1) / 2 * (q99 - q01) + q01 = 0.5 * range + q01
        # For dims 0-2: 0.5 * 1.0 + 0.0 = 0.5
        # For dims 3-5: 0.5 * 6.28 - 3.14 = 0.0
        # For dim 6: 0.5 * 1.0 + 0.0 = 0.5
        expected_translation = 0.5
        expected_rotation = 0.0
        expected_gripper = 0.5

        np.testing.assert_array_almost_equal(actions[:, :3], expected_translation, decimal=2)
        np.testing.assert_array_almost_equal(actions[:, 3:6], expected_rotation, decimal=2)
        np.testing.assert_array_almost_equal(actions[:, 6], expected_gripper, decimal=2)

    def test_cot_outputs_no_unnormalize_without_norm_stats(self):
        """Test that CoTOutputs returns normalized actions when norm_stats is None."""
        from openpi_cot.policies.cot_policy import CoTOutputs

        # Create CoTOutputs without norm_stats
        cot_outputs = CoTOutputs(
            language_action_format="vla0_chunked",
            norm_stats=None,
        )

        # Create mock reasoning
        reasoning = " ".join(["500"] * 70)
        data = {"reasoning": reasoning}

        result = cot_outputs(data)

        # Actions should still be in normalized [-1, 1] space
        actions = result["actions"]
        # Token 500 -> 0.0 in normalized space
        np.testing.assert_array_almost_equal(actions, np.zeros((10, 7)), decimal=2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
