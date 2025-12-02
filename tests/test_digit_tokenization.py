"""Unit test to understand tokenizer behavior for digit tokens.

This test explores:
1. How single digits (0-9) are tokenized
2. How multi-digit numbers are tokenized
3. How numbers in language action contexts are tokenized
4. Mapping from token IDs to digit values
"""

import numpy as np
import pytest

from openpi_cot.models.tokenizer import PaligemmaCoTTokenizer


def test_single_digit_tokenization():
    """Test how single digits 0-9 are tokenized."""
    tokenizer = PaligemmaCoTTokenizer(
        max_len=48,
        prompt_format="pi05",
        tokenizer_type="paligemma",
    )

    print("\n" + "="*80)
    print("SINGLE DIGIT TOKENIZATION")
    print("="*80)

    digit_to_token_id = {}
    digit_to_piece = {}

    for digit in range(10):
        # Encode just the digit
        tokens = tokenizer.encode(str(digit), add_bos=False, add_eos=False)

        # Get the piece representation
        pieces = [tokenizer._tokenizer.id_to_piece(t) for t in tokens]

        digit_to_token_id[digit] = tokens[0] if len(tokens) == 1 else tokens
        digit_to_piece[digit] = pieces[0] if len(pieces) == 1 else pieces

        print(f"Digit {digit}:")
        print(f"  Token IDs: {tokens}")
        print(f"  Pieces: {pieces}")
        print(f"  Is single token: {len(tokens) == 1}")

    print("\n" + "-"*80)
    print("DIGIT TO TOKEN MAPPING:")
    for digit, token_id in digit_to_token_id.items():
        print(f"  {digit} -> {token_id} ('{digit_to_piece[digit]}')")

    return digit_to_token_id, digit_to_piece


def test_multi_digit_tokenization():
    """Test how multi-digit numbers are tokenized."""
    tokenizer = PaligemmaCoTTokenizer(
        max_len=48,
        prompt_format="pi05",
        tokenizer_type="paligemma",
    )

    print("\n" + "="*80)
    print("MULTI-DIGIT TOKENIZATION")
    print("="*80)

    test_numbers = ["10", "23", "45", "100", "123", "0.5", "3.14"]

    for number in test_numbers:
        tokens = tokenizer.encode(number, add_bos=False, add_eos=False)
        pieces = [tokenizer._tokenizer.id_to_piece(t) for t in tokens]

        print(f"\nNumber '{number}':")
        print(f"  Token IDs: {tokens}")
        print(f"  Pieces: {pieces}")
        print(f"  Number of tokens: {len(tokens)}")

        # Try to identify the rightmost digit
        rightmost_digit = number.rstrip("0123456789")  # Remove trailing digits
        rightmost_digit = number[len(rightmost_digit):] if len(rightmost_digit) < len(number) else ""
        if rightmost_digit and rightmost_digit[-1].isdigit():
            rightmost_digit = rightmost_digit[-1]
            print(f"  Rightmost digit: {rightmost_digit}")


def test_language_action_tokenization():
    """Test tokenization of language actions with numbers."""
    tokenizer = PaligemmaCoTTokenizer(
        max_len=256,
        prompt_format="pi05",
        tokenizer_type="paligemma",
    )

    print("\n" + "="*80)
    print("LANGUAGE ACTION TOKENIZATION")
    print("="*80)

    # Sample language actions following the user's format
    test_actions = [
        "move forward 3 cm",
        "move left 5 cm",
        "rotate right 15 degrees",
        "tilt up 23 degrees",
        "move backward 100 cm",
        "rotate clockwise 7 degrees",
    ]

    for action in test_actions:
        # Tokenize the full action
        tokens = tokenizer.encode(action, add_bos=False, add_eos=False)
        pieces = [tokenizer._tokenizer.id_to_piece(t) for t in tokens]

        print(f"\nAction: '{action}'")
        print(f"  Tokens: {tokens}")
        print(f"  Pieces: {pieces}")
        print(f"  Piece-by-piece:")

        # Show each token and check if it contains a digit
        for i, (token_id, piece) in enumerate(zip(tokens, pieces)):
            has_digit = any(c.isdigit() for c in piece)
            if has_digit:
                print(f"    [{i}] Token {token_id}: '{piece}' <- CONTAINS DIGIT")

                # Extract rightmost digit from the piece
                rightmost = None
                for c in reversed(piece):
                    if c.isdigit():
                        rightmost = c
                        break
                if rightmost:
                    print(f"         Rightmost digit in piece: {rightmost}")
            else:
                print(f"    [{i}] Token {token_id}: '{piece}'")


def test_units_digit_extraction_logic():
    """Test logic for extracting units digits from language actions."""
    tokenizer = PaligemmaCoTTokenizer(
        max_len=256,
        prompt_format="pi05",
        tokenizer_type="paligemma",
    )

    print("\n" + "="*80)
    print("UNITS DIGIT EXTRACTION LOGIC")
    print("="*80)

    # Test with a full tokenization including reasoning
    prompt = "Robot is at position (0, 0, 0)"
    reasoning = "move forward 3 cm, move left 15 cm, rotate right 7 degrees"

    tokens, attn_mask, reasoning_mask, number_mask, direction_mask = tokenizer.tokenize_cot(
        prompt=prompt,
        reasoning=reasoning,
        state=None,
    )

    print(f"\nPrompt: {prompt}")
    print(f"Reasoning: {reasoning}")
    print(f"\nToken sequence length: {len(tokens)}")
    print(f"Number mask sum: {number_mask.sum()} (total number tokens)")

    # Decode and show tokens that are marked as numbers
    print("\nTokens marked as numbers:")
    pieces = []
    for t in tokens:
        if t == -2:  # Placeholder
            pieces.append("<IMG>")
        else:
            pieces.append(tokenizer._tokenizer.id_to_piece(int(t)))

    number_positions = np.where(number_mask)[0]
    for pos in number_positions:
        token_id = tokens[pos]
        piece = pieces[pos]
        print(f"  Position {pos}: Token {token_id}, Piece '{piece}'")

        # Check if this token represents a units digit
        # Strategy: token piece should contain exactly one digit, or if multiple,
        # we want the rightmost one
        digits_in_piece = [c for c in piece if c.isdigit()]
        if digits_in_piece:
            rightmost_digit = digits_in_piece[-1]
            print(f"    -> Rightmost digit: {rightmost_digit}")

            # This is a candidate for units_number_mask
            # Additional check: is this the last numeric token before a unit word (cm, degrees)?
            # For now, let's just check if it's a single digit token
            is_single_digit_token = len(digits_in_piece) == 1 and piece.strip() in "0123456789"
            print(f"    -> Is single-digit token: {is_single_digit_token}")


def test_contextual_digit_extraction():
    """Test extracting digits in the context of 'number + unit' patterns."""
    tokenizer = PaligemmaCoTTokenizer(
        max_len=256,
        prompt_format="pi05",
        tokenizer_type="paligemma",
    )

    print("\n" + "="*80)
    print("CONTEXTUAL DIGIT EXTRACTION (number + unit)")
    print("="*80)

    # Test specific patterns to understand tokenization boundaries
    test_patterns = [
        "3 cm",
        "15 cm",
        "100 cm",
        "7 degrees",
        "23 degrees",
        "5cm",  # No space
        "15cm",  # No space
    ]

    for pattern in test_patterns:
        tokens = tokenizer.encode(pattern, add_bos=False, add_eos=False)
        pieces = [tokenizer._tokenizer.id_to_piece(t) for t in tokens]

        print(f"\nPattern: '{pattern}'")
        print(f"  Tokens: {tokens}")
        print(f"  Pieces: {pieces}")

        # Find which token contains the units digit
        for i, piece in enumerate(pieces):
            if any(c.isdigit() for c in piece):
                digits = [c for c in piece if c.isdigit()]
                rightmost = digits[-1]
                print(f"  Token {i} ('{piece}'): rightmost digit = {rightmost}")


def test_digit_token_id_mapping():
    """Create a definitive mapping from digit values (0-9) to token IDs."""
    tokenizer = PaligemmaCoTTokenizer(
        max_len=48,
        prompt_format="pi05",
        tokenizer_type="paligemma",
    )

    print("\n" + "="*80)
    print("DIGIT TOKEN ID MAPPING (for label smoothing kernel)")
    print("="*80)

    digit_to_token_id = {}

    for digit in range(10):
        # Encode the digit in isolation
        tokens = tokenizer.encode(str(digit), add_bos=False, add_eos=False)

        if len(tokens) == 1:
            digit_to_token_id[digit] = int(tokens[0])
            print(f"  digit_to_token_id[{digit}] = {tokens[0]}")
        else:
            print(f"  WARNING: Digit {digit} tokenizes to multiple tokens: {tokens}")
            digit_to_token_id[digit] = int(tokens[-1])  # Use last token as fallback

    print("\nPython dict for use in code:")
    print(f"digit_to_token_id = {digit_to_token_id}")

    return digit_to_token_id


if __name__ == "__main__":
    # Run all tests
    test_single_digit_tokenization()
    test_multi_digit_tokenization()
    test_language_action_tokenization()
    test_units_digit_extraction_logic()
    test_contextual_digit_extraction()
    digit_mapping = test_digit_token_id_mapping()

    print("\n" + "="*80)
    print("TEST COMPLETE")
    print("="*80)
    print("\nKey findings will be used to implement units_number_token_mask")
    print("and digit_values extraction in the tokenizer.")
