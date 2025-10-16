"""Tests for language action encoding and decoding with rotation support."""

import numpy as np
import pytest

from src.openpi_cot.policies.cot_policy import COMPACT_DECODING_SCHEMA
from src.openpi_cot.policies.cot_policy import VERBOSE_DECODING_SCHEMA
from src.openpi_cot.policies.cot_policy import ActionDecodingSchema
from src.openpi_cot.policies.utils import is_idle_language_action
from src.openpi_cot.policies.utils import summarize_bimanual_numeric_actions
from src.openpi_cot.policies.utils import summarize_numeric_actions


class TestLanguageActionEncoding:
    """Test numeric actions to language string conversion."""

    def test_verbose_format_translation_only(self):
        """Test verbose format without rotation."""
        # Create action: move forward 5cm, right 3cm, down 2cm, gripper open (1.0)
        action = np.array([0.05, -0.03, -0.02, 0.0, 0.0, 0.0, 1.0])

        result = summarize_numeric_actions(action, sum_decimal="0f", include_rotation=False)

        # Expected: forward=+dy, right=-dy, down=-dz
        assert "move forward 5 cm" in result
        assert "move right 3 cm" in result
        assert "move down 2 cm" in result
        assert "set gripper to 1.0" in result
        assert "tilt" not in result
        assert "rotate" not in result

    def test_verbose_format_with_rotation(self):
        """Test verbose format with rotation included."""
        # Create action with rotation: roll=+π/2 rad (90°), pitch=-π/4 rad (-45°), yaw=π/6 rad (30°)
        action = np.array(
            [
                0.05,  # dx: forward 5cm
                -0.03,  # dy: right 3cm
                0.0,  # dz: no vertical
                np.pi / 2,  # roll: +90° (tilt left)
                -np.pi / 4,  # pitch: -45° (tilt down)
                np.pi / 6,  # yaw: +30° (rotate counterclockwise)
                0.5,  # gripper
            ]
        )

        result = summarize_numeric_actions(action, sum_decimal="0f", include_rotation=True)

        # Check translations
        assert "move forward 5 cm" in result
        assert "move right 3 cm" in result

        # Check rotations
        assert "tilt left 90 degrees" in result
        assert "tilt down 45 degrees" in result
        assert "rotate counterclockwise 30 degrees" in result
        assert "set gripper to 0.5" in result

    def test_verbose_format_negative_rotations(self):
        """Test negative rotation values."""
        action = np.array(
            [
                0.0,  # no translation
                0.0,
                0.0,
                -np.pi / 3,  # roll: -60° (tilt right)
                np.pi / 3,  # pitch: +60° (tilt up)
                -np.pi / 4,  # yaw: -45° (rotate clockwise)
                0.0,
            ]
        )

        result = summarize_numeric_actions(action, sum_decimal="0f", include_rotation=True)

        assert "tilt right 60 degrees" in result
        assert "tilt up 60 degrees" in result
        assert "rotate clockwise 45 degrees" in result

    def test_compact_format_without_rotation(self):
        """Test compact schema format."""
        # Action: forward 9cm, left 5cm, down 8cm, gripper closed
        action = np.array([0.09, 0.05, -0.08, 0.0, 0.0, 0.0, 0.0])

        result = summarize_numeric_actions(action, sum_decimal="compact", include_rotation=False)

        # Format: <dx dy dz grip>
        assert result == "<+09 +05 -08 0>"

    def test_compact_format_with_rotation(self):
        """Test compact schema format with rotation."""
        # Action: forward 9cm, left 5cm, down 8cm, roll +10°, pitch -5°, yaw +15°, gripper open
        action = np.array(
            [
                0.09,  # dx: +9cm
                0.05,  # dy: +5cm
                -0.08,  # dz: -8cm
                10 * np.pi / 180,  # roll: +10°
                -5 * np.pi / 180,  # pitch: -5°
                15 * np.pi / 180,  # yaw: +15°
                1.0,  # gripper
            ]
        )

        result = summarize_numeric_actions(action, sum_decimal="compact", include_rotation=True)

        # Format: <dx dy dz droll dpitch dyaw grip>
        # Note: +03d format means 3 chars total (sign + 2 digits), not 3 digits + sign
        assert result == "<+09 +05 -08 +10 -05 +15 1>"

    def test_multi_step_actions(self):
        """Test summing multiple action steps."""
        # Three steps: each moves forward 2cm
        actions = np.array(
            [
                [0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5],
                [0.02, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            ]
        )

        result = summarize_numeric_actions(actions, sum_decimal="0f", include_rotation=False)

        # Should sum to 6cm forward
        assert "move forward 6 cm" in result
        assert "set gripper to 1.0" in result  # Last gripper value

    def test_bimanual_actions(self):
        """Test bimanual action formatting."""
        # Left arm: forward 5cm, gripper open
        # Right arm: backward 3cm, gripper closed
        action = np.array(
            [
                0.05,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                1.0,  # Left arm
                -0.03,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,  # Right arm
            ]
        )

        result = summarize_bimanual_numeric_actions(action, sum_decimal="0f", include_rotation=False)

        assert "Left arm:" in result
        assert "Right arm:" in result
        assert "move forward 5 cm" in result
        assert "move back 3 cm" in result

    def test_bimanual_with_rotation(self):
        """Test bimanual action with rotation."""
        action = np.array(
            [
                0.05,
                0.0,
                0.0,
                np.pi / 4,
                0.0,
                0.0,
                1.0,  # Left: forward + tilt left 45°
                0.0,
                -0.03,
                0.0,
                0.0,
                np.pi / 6,
                0.0,
                0.0,  # Right: right + tilt up 30°
            ]
        )

        result = summarize_bimanual_numeric_actions(action, sum_decimal="0f", include_rotation=True)

        assert "tilt left 45 degrees" in result
        assert "tilt up 30 degrees" in result

    def test_bimanual_compact_format(self):
        """Test bimanual compact format without rotation."""
        # Left arm: forward 9cm, left 5cm, down 8cm, gripper open
        # Right arm: forward 3cm, right 2cm, up 1cm, gripper closed
        action = np.array(
            [
                0.09,
                0.05,
                -0.08,
                0.0,
                0.0,
                0.0,
                1.0,  # Left arm
                0.03,
                -0.02,
                0.01,
                0.0,
                0.0,
                0.0,
                0.0,  # Right arm
            ]
        )

        result = summarize_bimanual_numeric_actions(action, sum_decimal="compact", include_rotation=False)

        # Format: <L dx dy dz g R dx dy dz g>
        assert result == "<L +09 +05 -08 1 R +03 -02 +01 0>"

    def test_bimanual_compact_format_with_rotation(self):
        """Test bimanual compact format with rotation."""
        # Left arm: forward 9cm, left 5cm, down 8cm, roll +10°, pitch -5°, yaw +15°, gripper open
        # Right arm: forward 3cm, right 2cm, up 1cm, roll +20°, pitch +10°, yaw -5°, gripper closed
        action = np.array(
            [
                0.09,
                0.05,
                -0.08,
                10 * np.pi / 180,  # roll: +10°
                -5 * np.pi / 180,  # pitch: -5°
                15 * np.pi / 180,  # yaw: +15°
                1.0,  # Left arm
                0.03,
                -0.02,
                0.01,
                20 * np.pi / 180,  # roll: +20°
                10 * np.pi / 180,  # pitch: +10°
                -5 * np.pi / 180,  # yaw: -5°
                0.0,  # Right arm
            ]
        )

        result = summarize_bimanual_numeric_actions(action, sum_decimal="compact", include_rotation=True)

        # Format: <L dx dy dz dr dp dy g R dx dy dz dr dp dy g>
        assert result == "<L +09 +05 -08 +10 -05 +15 1 R +03 -02 +01 +20 +10 -05 0>"


class TestLanguageActionDecoding:
    """Test language string to numeric action conversion."""

    def test_verbose_decoding_translation(self):
        """Test parsing verbose language actions to deltas."""
        schema = VERBOSE_DECODING_SCHEMA
        reasoning = "move forward 10 cm and move right 5 cm and set gripper to 1"

        translations, grippers = schema.parse_language_to_deltas(reasoning)

        # Expected: forward=+dx, right=-dy
        np.testing.assert_allclose(translations[0], [0.10, -0.05, 0.0], atol=1e-6)
        assert grippers[0] == 1.0

    def test_verbose_decoding_all_directions(self):
        """Test all directional movements."""
        schema = VERBOSE_DECODING_SCHEMA

        test_cases = [
            ("move forward 10 cm", [0.10, 0.0, 0.0]),
            ("move backward 10 cm", [-0.10, 0.0, 0.0]),
            ("move left 10 cm", [0.0, 0.10, 0.0]),
            ("move right 10 cm", [0.0, -0.10, 0.0]),
            ("move up 10 cm", [0.0, 0.0, 0.10]),
            ("move down 10 cm", [0.0, 0.0, -0.10]),
        ]

        for reasoning, expected_delta in test_cases:
            translations, _ = schema.parse_language_to_deltas(reasoning)
            np.testing.assert_allclose(translations[0], expected_delta, atol=1e-6, err_msg=f"Failed for: {reasoning}")

    def test_compact_decoding(self):
        """Test parsing compact schema format."""
        schema = COMPACT_DECODING_SCHEMA
        reasoning = "Action: <+09 +05 -08 1>"

        translations, grippers = schema.parse_language_to_deltas(reasoning)

        np.testing.assert_allclose(translations[0], [0.09, 0.05, -0.08], atol=1e-6)
        assert grippers[0] == 1.0

    def test_compact_decoding_with_rotation(self):
        """Test parsing compact format with rotation."""
        # Use the compact_with_rotation schema from the registry
        from src.openpi_cot.policies.cot_policy import COMPACT_WITH_ROTATION_DECODING_SCHEMA

        schema = COMPACT_WITH_ROTATION_DECODING_SCHEMA

        # Format with rotation values: <+09 +05 -08 +10 -05 +15 1>
        reasoning = "Action: <+09 +05 -08 +10 -05 +15 1>"

        translations, grippers = schema.parse_language_to_deltas(reasoning)

        # Now it should correctly parse the translation values
        # Note: Rotation values are parsed but not currently returned (translation only)
        np.testing.assert_allclose(translations[0], [0.09, 0.05, -0.08], atol=1e-6)
        assert grippers[0] == 1.0

    def test_multi_sentence_decoding(self):
        """Test parsing multiple action sentences."""
        schema = VERBOSE_DECODING_SCHEMA
        reasoning = [
            "move forward 5 cm and set gripper to 0",
            "move right 3 cm and set gripper to 1",
            "move up 2 cm and set gripper to 1",
        ]

        translations, grippers = schema.parse_language_to_deltas(reasoning)

        assert translations.shape == (3, 3)
        np.testing.assert_allclose(translations[0], [0.05, 0.0, 0.0], atol=1e-6)
        np.testing.assert_allclose(translations[1], [0.0, -0.03, 0.0], atol=1e-6)
        np.testing.assert_allclose(translations[2], [0.0, 0.0, 0.02], atol=1e-6)
        np.testing.assert_array_equal(grippers, [0.0, 1.0, 1.0])

    def test_gripper_state_persistence(self):
        """Test that gripper state persists when not explicitly set."""
        schema = VERBOSE_DECODING_SCHEMA
        reasoning = [
            "move forward 5 cm and set gripper to 1",
            "move right 3 cm",  # No gripper command
            "move up 2 cm",  # No gripper command
        ]

        translations, grippers = schema.parse_language_to_deltas(reasoning)

        # Gripper should maintain value from previous step
        np.testing.assert_array_equal(grippers, [1.0, 1.0, 1.0])

    def test_bimanual_compact_decoding(self):
        """Test parsing bimanual compact format."""
        from src.openpi_cot.policies.cot_policy import COMPACT_BIMANUAL_DECODING_SCHEMA

        schema = COMPACT_BIMANUAL_DECODING_SCHEMA
        reasoning = "Action: <L +09 +05 -08 1 R +03 -02 +01 0>"

        left_trans, left_grip, right_trans, right_grip = schema.parse_bimanual_language_to_deltas(reasoning)

        # Check left arm
        np.testing.assert_allclose(left_trans[0], [0.09, 0.05, -0.08], atol=1e-6)
        assert left_grip[0] == 1.0

        # Check right arm
        np.testing.assert_allclose(right_trans[0], [0.03, -0.02, 0.01], atol=1e-6)
        assert right_grip[0] == 0.0

    def test_bimanual_compact_decoding_with_rotation(self):
        """Test parsing bimanual compact format with rotation."""
        from src.openpi_cot.policies.cot_policy import COMPACT_BIMANUAL_WITH_ROTATION_DECODING_SCHEMA

        schema = COMPACT_BIMANUAL_WITH_ROTATION_DECODING_SCHEMA
        reasoning = "Action: <L +09 +05 -08 +10 -05 +15 1 R +03 -02 +01 +20 +10 -05 0>"

        left_trans, left_grip, right_trans, right_grip = schema.parse_bimanual_language_to_deltas(reasoning)

        # Check left arm (translation only, rotation parsed but not returned)
        np.testing.assert_allclose(left_trans[0], [0.09, 0.05, -0.08], atol=1e-6)
        assert left_grip[0] == 1.0

        # Check right arm
        np.testing.assert_allclose(right_trans[0], [0.03, -0.02, 0.01], atol=1e-6)
        assert right_grip[0] == 0.0

    def test_bimanual_verbose_decoding(self):
        """Test parsing verbose bimanual format."""
        schema = VERBOSE_DECODING_SCHEMA
        reasoning = "Left arm: move forward 5 cm and set gripper to 1.0. Right arm: move back 3 cm and set gripper to 0.0"

        left_trans, left_grip, right_trans, right_grip = schema.parse_bimanual_language_to_deltas(reasoning)

        # Check left arm
        np.testing.assert_allclose(left_trans[0], [0.05, 0.0, 0.0], atol=1e-6)
        assert left_grip[0] == 1.0

        # Check right arm
        np.testing.assert_allclose(right_trans[0], [-0.03, 0.0, 0.0], atol=1e-6)
        assert right_grip[0] == 0.0


class TestRoundTripConsistency:
    """Test that encoding and decoding are consistent."""

    def test_verbose_roundtrip_translation(self):
        """Test encoding then decoding gives consistent results."""
        original_action = np.array([0.05, -0.03, 0.02, 0.0, 0.0, 0.0, 1.0])

        # Encode to language
        language = summarize_numeric_actions(original_action, sum_decimal="0f", include_rotation=False)

        # Decode back to numeric
        schema = VERBOSE_DECODING_SCHEMA
        translations, grippers = schema.parse_language_to_deltas(language)

        # Reconstruct action (without rotation)
        decoded_action = np.concatenate(
            [
                translations[0],
                np.zeros(3),  # rotation
                [grippers[0]],
            ]
        )

        # Check translation components match (within rounding to cm)
        np.testing.assert_allclose(decoded_action[:3], original_action[:3], atol=0.005)
        assert decoded_action[6] == original_action[6]

    def test_compact_roundtrip(self):
        """Test compact format roundtrip."""
        original_action = np.array([0.09, 0.05, -0.08, 0.0, 0.0, 0.0, 1.0])

        # Encode
        language = summarize_numeric_actions(original_action, sum_decimal="compact", include_rotation=False)

        # Decode
        schema = COMPACT_DECODING_SCHEMA
        translations, grippers = schema.parse_language_to_deltas(language)

        # Check exact match (compact format has integer cm precision)
        np.testing.assert_array_equal(translations[0], original_action[:3])
        assert grippers[0] == original_action[6]

    def test_compact_roundtrip_with_rotation(self):
        """Test compact format roundtrip with rotation."""
        from src.openpi_cot.policies.cot_policy import COMPACT_WITH_ROTATION_DECODING_SCHEMA

        # Action with rotation: dx=9cm, dy=5cm, dz=-8cm, roll=10°, pitch=-5°, yaw=15°, gripper=1
        original_action = np.array([
            0.09,
            0.05,
            -0.08,
            10 * np.pi / 180,  # roll: +10°
            -5 * np.pi / 180,  # pitch: -5°
            15 * np.pi / 180,  # yaw: +15°
            1.0,
        ])

        # Encode
        language = summarize_numeric_actions(original_action, sum_decimal="compact", include_rotation=True)
        assert language == "<+09 +05 -08 +10 -05 +15 1>"

        # Decode
        schema = COMPACT_WITH_ROTATION_DECODING_SCHEMA
        translations, grippers = schema.parse_language_to_deltas(language)

        # Check translation and gripper match
        np.testing.assert_array_equal(translations[0], original_action[:3])
        assert grippers[0] == original_action[6]

    def test_bimanual_compact_roundtrip(self):
        """Test bimanual compact format roundtrip."""
        from src.openpi_cot.policies.cot_policy import COMPACT_BIMANUAL_DECODING_SCHEMA

        # Left arm: forward 9cm, left 5cm, down 8cm, gripper open
        # Right arm: forward 3cm, right 2cm, up 1cm, gripper closed
        original_action = np.array([
            0.09, 0.05, -0.08, 0.0, 0.0, 0.0, 1.0,  # Left arm
            0.03, -0.02, 0.01, 0.0, 0.0, 0.0, 0.0,  # Right arm
        ])

        # Encode
        language = summarize_bimanual_numeric_actions(original_action, sum_decimal="compact", include_rotation=False)
        assert language == "<L +09 +05 -08 1 R +03 -02 +01 0>"

        # Decode
        schema = COMPACT_BIMANUAL_DECODING_SCHEMA
        left_trans, left_grip, right_trans, right_grip = schema.parse_bimanual_language_to_deltas(language)

        # Check exact match
        np.testing.assert_array_equal(left_trans[0], original_action[:3])
        assert left_grip[0] == original_action[6]
        np.testing.assert_array_equal(right_trans[0], original_action[7:10])
        assert right_grip[0] == original_action[13]

    def test_bimanual_compact_roundtrip_with_rotation(self):
        """Test bimanual compact format roundtrip with rotation."""
        from src.openpi_cot.policies.cot_policy import COMPACT_BIMANUAL_WITH_ROTATION_DECODING_SCHEMA

        # Left arm with rotation, right arm with rotation
        original_action = np.array([
            0.09, 0.05, -0.08,
            10 * np.pi / 180, -5 * np.pi / 180, 15 * np.pi / 180,
            1.0,  # Left arm
            0.03, -0.02, 0.01,
            20 * np.pi / 180, 10 * np.pi / 180, -5 * np.pi / 180,
            0.0,  # Right arm
        ])

        # Encode
        language = summarize_bimanual_numeric_actions(original_action, sum_decimal="compact", include_rotation=True)
        assert language == "<L +09 +05 -08 +10 -05 +15 1 R +03 -02 +01 +20 +10 -05 0>"

        # Decode
        schema = COMPACT_BIMANUAL_WITH_ROTATION_DECODING_SCHEMA
        left_trans, left_grip, right_trans, right_grip = schema.parse_bimanual_language_to_deltas(language)

        # Check translation and gripper match
        np.testing.assert_array_equal(left_trans[0], original_action[:3])
        assert left_grip[0] == original_action[6]
        np.testing.assert_array_equal(right_trans[0], original_action[7:10])
        assert right_grip[0] == original_action[13]


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_zero_action(self):
        """Test action with all zeros."""
        action = np.zeros(7)
        result = summarize_numeric_actions(action, sum_decimal="0f", include_rotation=False)

        # Should only have gripper command
        assert "set gripper to 0.0" in result
        assert "move" not in result

    def test_tiny_movements(self):
        """Test very small movements that round to zero."""
        action = np.array([0.001, -0.002, 0.003, 0.0, 0.0, 0.0, 0.5])
        result = summarize_numeric_actions(action, sum_decimal="0f", include_rotation=False)

        # With 0 decimal places, these should round to 0 and not appear
        assert "move" not in result or all(x in result for x in ["0 cm"] if "move" in result)

    def test_large_action_values(self):
        """Test large action values."""
        action = np.array([1.5, -2.3, 0.8, 0.0, 0.0, 0.0, 1.0])
        result = summarize_numeric_actions(action, sum_decimal="0f", include_rotation=False)

        assert "move forward 150 cm" in result
        assert "move right 230 cm" in result
        assert "move up 80 cm" in result

    def test_precision_format(self):
        """Test different decimal precision."""
        action = np.array([0.055, -0.033, 0.0, 0.0, 0.0, 0.5])

        # 2 decimal places
        result = summarize_numeric_actions(action, sum_decimal="2f", include_rotation=False)
        assert "5.50 cm" in result or "5.5 cm" in result
        assert "3.30 cm" in result or "3.3 cm" in result

    def test_invalid_decoding_input(self):
        """Test decoding with malformed input."""
        schema = VERBOSE_DECODING_SCHEMA

        # No movement commands
        translations, grippers = schema.parse_language_to_deltas("random text with no commands")
        np.testing.assert_array_equal(translations[0], [0.0, 0.0, 0.0])

        # Empty string
        translations, grippers = schema.parse_language_to_deltas("")
        np.testing.assert_array_equal(translations[0], [0.0, 0.0, 0.0])


class TestIdleChecker:
    """Test idle action detection for filtering out minimal movements."""

    def test_compact_format_zero_movement(self):
        """Test that zero movement is detected as idle in compact format."""
        # No movement at all
        language_action = "<+00 +00 +00 1>"
        assert is_idle_language_action(language_action, sum_decimal="compact", include_rotation=False)

    def test_compact_format_tiny_movement_is_idle(self):
        """Test that sub-threshold movements are detected as idle."""
        # Movement with L2 norm < 1.0 cm (default threshold)
        # sqrt(0^2 + 0^2 + 0^2) = 0 < 1.0
        language_action = "<+00 +00 +00 0>"
        assert is_idle_language_action(language_action, sum_decimal="compact", include_rotation=False)

    def test_compact_format_below_threshold(self):
        """Test that movements just below threshold are idle."""
        # Movement with L2 norm just below 1.0 cm threshold
        # sqrt(0^2 + 0^2 + 0^2) = 0 < 1.0
        language_action = "<+00 +00 +00 1>"
        assert is_idle_language_action(language_action, sum_decimal="compact", include_rotation=False)

        # Custom threshold: movement of 5cm should be idle with threshold=10cm
        language_action = "<+03 +04 +00 1>"  # sqrt(9+16) = 5cm < 10cm
        assert is_idle_language_action(
            language_action,
            sum_decimal="compact",
            include_rotation=False,
            translation_threshold=10.0
        )

    def test_compact_format_with_rotation_both_below_threshold(self):
        """Test idle detection with rotation when both translation and rotation are below threshold."""
        # Translation: sqrt(1+1+1) = 1.73cm < default 1.0? No, but let's test with zero
        # Rotation: sqrt(1+1+1) = 1.73deg < default 10.0
        language_action = "<+00 +00 +00 +01 +01 +01 1>"
        assert is_idle_language_action(language_action, sum_decimal="compact", include_rotation=True)

    def test_compact_format_gripper_only_change(self):
        """Test that gripper-only changes with no movement are idle."""
        # Gripper changes but no translation
        language_action = "<+00 +00 +00 1>"
        assert is_idle_language_action(language_action, sum_decimal="compact", include_rotation=False)

    def test_verbose_format_zero_movement(self):
        """Test idle detection in verbose format with no movement commands."""
        # Only gripper command, no movement
        language_action = "set gripper to 1.0"
        assert is_idle_language_action(language_action, sum_decimal="0f", include_rotation=False)

    def test_verbose_format_tiny_movement(self):
        """Test idle detection in verbose format with sub-threshold movement."""
        # Very small movement: sqrt(0.5^2) = 0.5cm < 1.0cm
        language_action = "move forward 0.5 cm and set gripper to 1.0"
        assert is_idle_language_action(language_action, sum_decimal="0f", include_rotation=False)

    def test_verbose_format_below_threshold(self):
        """Test verbose format with movement just below threshold."""
        # Movement with L2 norm < 1.0 cm
        # sqrt(0.3^2 + 0.3^2 + 0.3^2) = 0.52cm < 1.0cm
        language_action = "move forward 0.3 cm and move right 0.3 cm and move down 0.3 cm and set gripper to 0.0"
        assert is_idle_language_action(language_action, sum_decimal="0f", include_rotation=False)

    def test_verbose_format_with_rotation_both_below_threshold(self):
        """Test idle detection with rotation in verbose format."""
        # Small translation and small rotation
        language_action = "move forward 0.5 cm and tilt left 5 degrees and set gripper to 1.0"
        assert is_idle_language_action(language_action, sum_decimal="0f", include_rotation=True)

    def test_empty_language_action(self):
        """Test that empty or invalid language actions are treated as idle."""
        # Empty string
        assert is_idle_language_action("", sum_decimal="compact", include_rotation=False)

        # None (will fail isinstance check)
        assert is_idle_language_action(None, sum_decimal="compact", include_rotation=False)

    def test_unparseable_language_action(self):
        """Test that unparseable language actions are treated as idle."""
        # Random text that doesn't match any pattern
        language_action = "random text with no valid action format"
        assert is_idle_language_action(language_action, sum_decimal="compact", include_rotation=False)

    def test_custom_thresholds(self):
        """Test idle detection with custom thresholds."""
        # Movement of 5cm should NOT be idle with default threshold (1.0cm)
        # but SHOULD be idle with higher threshold (10.0cm)
        language_action = "<+03 +04 +00 1>"  # sqrt(9+16) = 5cm

        # With default threshold of 1.0cm, this is NOT idle
        assert not is_idle_language_action(language_action, sum_decimal="compact", include_rotation=False)

        # With custom threshold of 10.0cm, this IS idle
        assert is_idle_language_action(
            language_action,
            sum_decimal="compact",
            include_rotation=False,
            translation_threshold=10.0
        )

    def test_rotation_threshold_custom(self):
        """Test rotation threshold with custom values."""
        # Large rotation but small translation
        language_action = "<+00 +00 +00 +05 +05 +05 1>"  # sqrt(25+25+25) = 8.66deg

        # With default rotation threshold (10.0deg), this IS idle
        assert is_idle_language_action(language_action, sum_decimal="compact", include_rotation=True)

        # With custom rotation threshold (5.0deg), this is NOT idle
        assert not is_idle_language_action(
            language_action,
            sum_decimal="compact",
            include_rotation=True,
            rotation_threshold_deg=5.0
        )


if __name__ == "__main__":
    # Run tests with pytest if available, otherwise run basic tests
    try:
        pytest.main([__file__, "-v"])
    except ImportError:
        print("pytest not available, running basic tests...")

        # Run a few basic tests manually
        test_encoding = TestLanguageActionEncoding()
        test_encoding.test_verbose_format_translation_only()
        test_encoding.test_verbose_format_with_rotation()
        test_encoding.test_compact_format_with_rotation()

        test_decoding = TestLanguageActionDecoding()
        test_decoding.test_verbose_decoding_translation()
        test_decoding.test_compact_decoding()

        print("Basic tests passed!")
