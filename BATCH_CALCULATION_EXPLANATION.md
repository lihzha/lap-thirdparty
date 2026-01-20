# Explanation: Why 84 Validation XEMB Demo Batches

## Summary
The number of validation xemb demo batches (84) is calculated using a specific formula in `num_val_batches_per_epoch` that uses the first dataset's `num_transitions` count, not the total dataset length.

## Calculation Formula

The calculation is performed in `src/openpi_cot/datasets/dataset_mixer.py` at lines 643-654:

```python
@property
def num_val_batches_per_epoch(self) -> int:
    """Compute number of batches per epoch based on dataset length and batch size."""
    import jax

    num_transitions = next(v.num_transitions for k, v in self.global_statistics.items() if "state" in k)

    return int(
        num_transitions
        // (self.batch_size)  # NOTE: No sharding here since we're using the full dataset
        * self.config.val_fraction
        * 0.8  # empirically estimated ratio for filtering
    )
```

## Key Components

1. **`num_transitions`**: Retrieved from the first dataset's state statistics in `global_statistics` that has "state" in its key
   - From logs: `yam_demo_dataset` state has `num_transitions=9256`
   - This is the value used, NOT the total dataset length

2. **`self.batch_size`**: The batch size configured for the data loader
   - Estimated to be approximately 88 based on the calculation

3. **`self.config.val_fraction`**: Set to `1.0` for the xemb demo data loader (see `scripts/train.py:546`)

4. **`0.8` factor**: An empirically estimated ratio for filtering (hardcoded in the formula)

## Step-by-Step Calculation

Given the log output:
- `yam_demo_dataset` state: `num_transitions=9256`
- `franka_demo_dataset` state: `num_transitions=4384`
- Total `dataset_length`: 13640
- Result: **84 batches**

The calculation works as follows:

1. `num_transitions = 9256` (first dataset with "state" in key)
2. Formula: `int((9256 // batch_size) * 1.0 * 0.8) = 84`
3. Solving: `(9256 // batch_size) * 0.8 = 84`
4. Therefore: `9256 // batch_size = 84 / 0.8 = 105`
5. This means: `105 * batch_size ≤ 9256 < 106 * batch_size`
6. Solving: `batch_size ≈ 88` (since `105 * 88 = 9240 ≤ 9256 < 106 * 88 = 9328`)

## Important Note

**The calculation uses only the first dataset's `num_transitions` (9256), not the total dataset length (13640).**

This explains why:
- Dataset length: 13640 samples
- But calculation uses: 9256 transitions (from yam_demo_dataset only)
- Result: 84 batches

If the calculation used the total dataset length:
- `int((13640 // 88) * 0.8) = int(155 * 0.8) = 124 batches`

But the actual implementation uses only the first dataset's transition count, resulting in 84 batches.

## Code Location

- Calculation method: `src/openpi_cot/datasets/dataset_mixer.py:643-654`
- Called from: `src/openpi_cot/datasets/cot_data_loader.py:507-515`
- Logged at: `scripts/train.py:569`
