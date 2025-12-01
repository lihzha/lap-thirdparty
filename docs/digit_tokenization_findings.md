# Digit Tokenization Analysis - Key Findings

## Summary
This document summarizes the tokenization behavior of the PaliGemma tokenizer for digits in language actions, used to implement label smoothing for units digits.

## Digit Token Mapping

Each single digit (0-9) is tokenized as **exactly one token**:

```python
digit_to_token_id = {
    0: 235276,
    1: 235274,
    2: 235284,
    3: 235304,
    4: 235310,
    5: 235308,
    6: 235318,
    7: 235324,
    8: 235321,
    9: 235315
}
```

## Multi-Digit Number Tokenization

Multi-digit numbers are tokenized as **separate single-digit tokens**:

| Number | Tokens | Token IDs |
|--------|--------|-----------|
| "3"    | ['3'] | [235304] |
| "15"   | ['1', '5'] | [235274, 235308] |
| "23"   | ['2', '3'] | [235284, 235304] |
| "100"  | ['1', '0', '0'] | [235274, 235276, 235276] |

## Language Action Patterns

### Single-digit numbers
```
"move forward 3 cm"
→ ['move', '▁forward', '▁', '3', '▁cm']
                              ↑
                         units digit
```

### Multi-digit numbers
```
"move left 15 cm"
→ ['move', '▁left', '▁', '1', '5', '▁cm']
                              ↑
                         units digit (rightmost)

"rotate right 23 degrees"
→ ['rotate', '▁right', '▁', '2', '3', '▁degrees']
                                   ↑
                              units digit (rightmost)
```

## Units Digit Identification Strategy

**Definition**: The units digit is the **rightmost digit** in a number before a unit word (cm, degrees, etc.).

**Algorithm**:
1. Identify all digit tokens using existing `number_mask`
2. Find sequences of consecutive digit tokens
3. Check if the token immediately after the sequence is a unit word
4. If yes, mark the **last digit in the sequence** as the units digit

**Unit words to check**:
- "cm" (token 3377 with space, 2890 without)
- "degrees" (token 12584)
- Potentially: "mm", "m", "radians", etc.

## Implementation Notes

### For tokenizer.py modifications:

1. After building `number_mask`, iterate through tokens
2. Detect digit sequences followed by unit words
3. Create `units_number_mask` marking only the last digit in each sequence
4. Create `digit_values` array extracting the digit value (0-9) from each token piece

### Example output for "move left 15 cm":
```python
tokens:              [... 235274, 235308, 3377 ...]  # '1', '5', 'cm'
number_mask:         [... True,   True,   False ...]
units_number_mask:   [... False,  True,   False ...]  # Only '5' is marked
digit_values:        [... 1,      5,      -1 ...]     # -1 for non-digits
```

## Verification

Test case from actual tokenization:
```
Prompt: "Robot is at position (0, 0, 0)"
Reasoning: "move forward 3 cm, move left 15 cm, rotate right 7 degrees"

Number tokens detected:
  Position 240: '3' → units digit (before "cm")
  Position 246: '1' → NOT units digit
  Position 247: '5' → units digit (before "cm")
  Position 253: '7' → units digit (before "degrees")

Expected units_number_mask: positions 240, 247, 253 should be True
Expected digit_values: [240]=3, [246]=1, [247]=5, [253]=7
```

## Next Steps

1. ✅ Phase 1: Understand tokenization (COMPLETE)
2. ⏭️ Phase 2: Implement `units_number_mask` and `digit_values` in tokenizer
3. ⏭️ Phase 3: Update data pipeline
4. ⏭️ Phase 4: Create label smoothing kernel
5. ⏭️ Phase 5: Modify loss function
