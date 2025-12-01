"""Test units digit extraction in tokenizer.

This validates that:
1. units_number_mask correctly identifies the rightmost digit before unit words
2. digit_values correctly extracts digit values (0-9) from tokens
"""

import numpy as np

from openpi_cot.models.tokenizer import PaligemmaCoTTokenizer


def test_units_digit_single_digit():
    """Test single-digit numbers before units."""
    tokenizer = PaligemmaCoTTokenizer(
        max_len=256,
        prompt_format="pi05",
        tokenizer_type="paligemma",
    )

    prompt = "Robot is at position (0, 0, 0)"
    reasoning = "move forward 3 cm"

    tokens, attn_mask, reasoning_mask, number_mask, direction_mask, units_number_mask, digit_values = (
        tokenizer.tokenize_cot(
            prompt=prompt,
            reasoning=reasoning,
            state=None,
        )
    )

    print("\n" + "="*80)
    print("TEST: Single-digit number (3 cm)")
    print("="*80)

    # Get pieces for debugging
    pieces = []
    for t in tokens:
        if t == -2:
            pieces.append("<IMG>")
        elif t == tokenizer._tokenizer.pad_id():
            pieces.append("<PAD>")
        else:
            pieces.append(tokenizer._tokenizer.id_to_piece(int(t)))

    # Find positions with units_number_mask
    units_positions = np.where(units_number_mask)[0]
    print(f"Units digit positions: {units_positions}")

    for pos in units_positions:
        print(f"  Position {pos}: token '{pieces[pos]}', digit_value={digit_values[pos]}")

    # Verify
    assert units_number_mask.sum() == 1, f"Expected 1 units digit, got {units_number_mask.sum()}"
    assert digit_values[units_positions[0]] == 3, f"Expected digit value 3, got {digit_values[units_positions[0]]}"

    print("✓ Single-digit test passed")


def test_units_digit_multi_digit():
    """Test multi-digit numbers before units."""
    tokenizer = PaligemmaCoTTokenizer(
        max_len=256,
        prompt_format="pi05",
        tokenizer_type="paligemma",
    )

    prompt = "Robot is at position (0, 0, 0)"
    reasoning = "move left 15 cm, rotate right 23 degrees"

    tokens, attn_mask, reasoning_mask, number_mask, direction_mask, units_number_mask, digit_values = (
        tokenizer.tokenize_cot(
            prompt=prompt,
            reasoning=reasoning,
            state=None,
        )
    )

    print("\n" + "="*80)
    print("TEST: Multi-digit numbers (15 cm, 23 degrees)")
    print("="*80)

    # Get pieces for debugging
    pieces = []
    for t in tokens:
        if t == -2:
            pieces.append("<IMG>")
        elif t == tokenizer._tokenizer.pad_id():
            pieces.append("<PAD>")
        else:
            pieces.append(tokenizer._tokenizer.id_to_piece(int(t)))

    # Find positions with number_mask
    number_positions = np.where(number_mask)[0]
    print(f"All number positions: {number_positions}")
    for pos in number_positions:
        is_units = units_number_mask[pos]
        marker = " <- UNITS DIGIT" if is_units else ""
        print(f"  Position {pos}: token '{pieces[pos]}', digit_value={digit_values[pos]}{marker}")

    # Find positions with units_number_mask
    units_positions = np.where(units_number_mask)[0]
    print(f"\nUnits digit positions: {units_positions}")

    # Verify
    assert units_number_mask.sum() == 2, f"Expected 2 units digits, got {units_number_mask.sum()}"

    # Check digit values
    units_digit_values = [digit_values[pos] for pos in units_positions]
    print(f"Units digit values: {units_digit_values}")

    # Should be [5, 3] for "15" and "23"
    assert 5 in units_digit_values, "Expected units digit 5 from '15 cm'"
    assert 3 in units_digit_values, "Expected units digit 3 from '23 degrees'"

    print("✓ Multi-digit test passed")


def test_units_digit_comprehensive():
    """Test comprehensive language actions."""
    tokenizer = PaligemmaCoTTokenizer(
        max_len=512,
        prompt_format="pi05",
        tokenizer_type="paligemma",
    )

    prompt = "Robot is at position (0, 0, 0)"
    reasoning = "move forward 3 cm, move left 15 cm, rotate right 7 degrees, tilt up 100 degrees"

    tokens, attn_mask, reasoning_mask, number_mask, direction_mask, units_number_mask, digit_values = (
        tokenizer.tokenize_cot(
            prompt=prompt,
            reasoning=reasoning,
            state=None,
        )
    )

    print("\n" + "="*80)
    print("TEST: Comprehensive (3 cm, 15 cm, 7 degrees, 100 degrees)")
    print("="*80)

    # Get pieces for debugging
    pieces = []
    for t in tokens:
        if t == -2:
            pieces.append("<IMG>")
        elif t == tokenizer._tokenizer.pad_id():
            pieces.append("<PAD>")
        else:
            pieces.append(tokenizer._tokenizer.id_to_piece(int(t)))

    # Show all number tokens
    number_positions = np.where(number_mask)[0]
    print(f"All number tokens ({len(number_positions)}):")
    for pos in number_positions:
        is_units = units_number_mask[pos]
        marker = " <- UNITS DIGIT" if is_units else ""
        print(f"  Position {pos}: '{pieces[pos]}', digit_value={digit_values[pos]}{marker}")

    # Verify units digits
    units_positions = np.where(units_number_mask)[0]
    print(f"\nUnits digit positions: {units_positions}")

    units_digit_values = [digit_values[pos] for pos in units_positions]
    print(f"Units digit values: {units_digit_values}")

    # Expected: 3 (from "3 cm"), 5 (from "15 cm"), 7 (from "7 degrees"), 0 (from "100 degrees")
    assert units_number_mask.sum() == 4, f"Expected 4 units digits, got {units_number_mask.sum()}"
    assert set(units_digit_values) == {3, 5, 7, 0}, f"Unexpected units digit values: {units_digit_values}"

    print("✓ Comprehensive test passed")


def test_digit_values_all_numbers():
    """Test that digit_values is correctly populated for all number tokens."""
    tokenizer = PaligemmaCoTTokenizer(
        max_len=256,
        prompt_format="pi05",
        tokenizer_type="paligemma",
    )

    prompt = "Robot is at position (0, 0, 0)"
    reasoning = "move forward 123 cm"

    tokens, attn_mask, reasoning_mask, number_mask, direction_mask, units_number_mask, digit_values = (
        tokenizer.tokenize_cot(
            prompt=prompt,
            reasoning=reasoning,
            state=None,
        )
    )

    print("\n" + "="*80)
    print("TEST: Digit values for all number tokens (123 cm)")
    print("="*80)

    # Get pieces for debugging
    pieces = []
    for t in tokens:
        if t == -2:
            pieces.append("<IMG>")
        elif t == tokenizer._tokenizer.pad_id():
            pieces.append("<PAD>")
        else:
            pieces.append(tokenizer._tokenizer.id_to_piece(int(t)))

    # Check all number positions have digit values
    number_positions = np.where(number_mask)[0]
    print(f"Number tokens:")
    for pos in number_positions:
        is_units = units_number_mask[pos]
        marker = " <- UNITS DIGIT" if is_units else ""
        print(f"  Position {pos}: '{pieces[pos]}', digit_value={digit_values[pos]}{marker}")
        assert digit_values[pos] >= 0, f"Position {pos} has invalid digit_value {digit_values[pos]}"

    # Verify units digit is 3 (last digit of 123)
    units_positions = np.where(units_number_mask)[0]
    assert len(units_positions) == 1, f"Expected 1 units digit, got {len(units_positions)}"
    assert digit_values[units_positions[0]] == 3, f"Expected units digit 3, got {digit_values[units_positions[0]]}"

    # Verify all three digits have correct values
    digit_values_in_number = [digit_values[pos] for pos in number_positions]
    assert digit_values_in_number == [1, 2, 3], f"Expected [1, 2, 3], got {digit_values_in_number}"

    print("✓ All digit values test passed")


def test_no_unit_word():
    """Test that numbers without unit words are not marked as units digits."""
    tokenizer = PaligemmaCoTTokenizer(
        max_len=256,
        prompt_format="pi05",
        tokenizer_type="paligemma",
    )

    prompt = "Robot is at position (0, 0, 0)"
    reasoning = "move forward by 3"  # No "cm" or "degrees"

    tokens, attn_mask, reasoning_mask, number_mask, direction_mask, units_number_mask, digit_values = (
        tokenizer.tokenize_cot(
            prompt=prompt,
            reasoning=reasoning,
            state=None,
        )
    )

    print("\n" + "="*80)
    print("TEST: Number without unit word (3 without cm/degrees)")
    print("="*80)

    # Get pieces for debugging
    pieces = []
    for t in tokens:
        if t == -2:
            pieces.append("<IMG>")
        elif t == tokenizer._tokenizer.pad_id():
            pieces.append("<PAD>")
        else:
            pieces.append(tokenizer._tokenizer.id_to_piece(int(t)))

    number_positions = np.where(number_mask)[0]
    units_positions = np.where(units_number_mask)[0]

    print(f"Number positions: {number_positions}")
    print(f"Units digit positions: {units_positions}")

    # Number should be detected but NOT marked as units digit
    assert len(number_positions) > 0, "Expected to find number token '3'"
    assert len(units_positions) == 0, f"Expected 0 units digits (no unit word), got {len(units_positions)}"

    print("✓ No unit word test passed")


if __name__ == "__main__":
    test_units_digit_single_digit()
    test_units_digit_multi_digit()
    test_units_digit_comprehensive()
    test_digit_values_all_numbers()
    test_no_unit_word()

    print("\n" + "="*80)
    print("ALL TESTS PASSED ✓")
    print("="*80)
