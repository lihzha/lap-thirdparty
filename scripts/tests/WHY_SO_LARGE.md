# Why Are Dataloader Checkpoints SO Large?

## TL;DR

**Your checkpoint is 50-100+ GB because it saves the entire shuffle buffer (250,000-400,000 samples with images)!**

**Solution**: Use `test_dataloader_checkpoint_small_buffer.py` which reduces shuffle buffer to 100 samples for testing.

---

## The Problem Explained

### Default Configuration
```python
shuffle_buffer_size: int = 250_000  # 250,000 samples in shuffle buffer!
```

Some configs use even more (400,000).

### What's in Each Sample?
Each training sample contains:
- **Images**: Multiple camera views (e.g., 224×224×3 RGB)
  - Primary camera: ~150 KB (compressed)
  - Wrist camera(s): ~150 KB each
- **State**: Robot joint positions, gripper state (~1-5 KB)
- **Actions**: Target actions (~1-5 KB)
- **Language**: Task instruction embeddings (~10-50 KB)

**Total per sample**: ~300 KB to 1 MB (depending on image compression and number of cameras)

### Checkpoint Size Calculation

#### Shuffle Buffer
```
250,000 samples × 500 KB/sample = 125 GB
```

#### Prefetch Buffer (2 batches × 32 samples/batch)
```
64 samples × 500 KB/sample = 32 MB
```

#### TensorFlow Overhead
During save, TensorFlow:
1. Creates temporary files
2. Writes checkpoint data
3. May create multiple copies

This can temporarily use **1.5-2× the final checkpoint size**.

#### Total Disk Usage During Save
```
125 GB (shuffle buffer)
+ 0.032 GB (prefetch buffer)
+ ~62 GB (temporary files during save)
= ~187 GB needed!
```

Even with "only" 10 GB free, you're **way short** of what's needed.

---

## Why Does TensorFlow Save the Entire Buffer?

When you checkpoint a TensorFlow dataset iterator with:
```python
dataset = dataset.shuffle(250_000)
```

The checkpoint includes:
1. **Iterator position**: Which sample are we on?
2. **Shuffle buffer state**: All 250,000 samples currently in the buffer
3. **Random state**: RNG state for shuffling
4. **Prefetch buffer state**: Pre-loaded batches

This is by design - TensorFlow needs to restore the **exact same state** when you resume, including which samples are in the shuffle buffer and in what order.

---

## Solutions

### Solution 1: Use Small Shuffle Buffer for Testing ✅ RECOMMENDED

```bash
export GCS_TEST_BUCKET='gs://your-bucket/test-checkpoints'
python scripts/tests/test_dataloader_checkpoint_small_buffer.py --config-name=your_config
```

This script:
- Automatically reduces `shuffle_buffer_size` from 250,000 → 100
- Reduces checkpoint size from ~187 GB → ~1 GB
- Still tests the checkpoint functionality correctly
- Shows you the size difference

**Checkpoint size with buffer=100**:
```
100 samples × 500 KB/sample = 50 MB
+ prefetch buffer = 32 MB
+ TensorFlow overhead = ~80 MB
= ~160 MB total (fits easily!)
```

### Solution 2: Reduce Shuffle Buffer in Your Config

If you need checkpointing in production, consider reducing the shuffle buffer:

```python
# config.py or your config file
shuffle_buffer_size: int = 10_000  # Instead of 250_000

# Checkpoint size: ~5 GB instead of ~125 GB
```

**Trade-off**: Smaller shuffle buffer = less randomness in training
- 10,000 is usually still sufficient
- 1,000 might affect training quality
- 100 is only for testing

### Solution 3: Use Massive Disk for Production Checkpoints

If you really need the full 250,000 buffer size checkpointed:

```bash
# You need 200+ GB of space
export TMPDIR=/mnt/massive-disk/tmp
mkdir -p $TMPDIR

export GCS_TEST_BUCKET='gs://your-bucket/checkpoints'
python scripts/tests/test_dataloader_checkpoint.py --config-name=your_config
```

---

## Comparison: Different Buffer Sizes

| shuffle_buffer_size | Checkpoint Size | Disk Space Needed | Training Quality |
|---------------------|-----------------|-------------------|------------------|
| 100 (testing only) | ~160 MB | ~500 MB | Poor (testing only) |
| 1,000 | ~1.5 GB | ~5 GB | Marginal |
| 10,000 | ~15 GB | ~50 GB | Good |
| 50,000 | ~75 GB | ~250 GB | Very Good |
| 250,000 (default) | ~375 GB | ~1 TB | Excellent |
| 400,000 | ~600 GB | ~1.5 TB | Excellent |

---

## Why Such a Large Shuffle Buffer?

The comment in the code explains:
```python
# Reduce this if you are running out of memory, but careful --
# below ~100k shuffling is not sufficiently random.
shuffle_buffer_size: int = 250_000
```

For **training** robotics policies:
- Need diverse, randomized batches
- Robot datasets have temporal correlation
- Larger shuffle buffer = better randomness = better training

For **testing checkpointing**:
- Don't need perfect randomness
- Just need to verify save/restore works
- Can use much smaller buffer

---

## Real Example

Let's say you're using the `droid_cot` config with:
- `shuffle_buffer_size=250,000`
- Images: 2 cameras × 224×224×3
- ~500 KB per sample

**Checkpoint save operation**:
```
1. TensorFlow allocates temp files in $TMPDIR
2. Writes shuffle buffer: 250,000 × 500 KB = 125 GB
3. Writes prefetch buffer: 64 × 500 KB = 32 MB
4. Writes iterator metadata: ~1 MB
5. Creates temporary copies during atomic write: +62 GB
6. Uploads to GCS (if saving to gs://)
7. Deletes temp files

Peak disk usage: ~187 GB
```

If your `/tmp` has only 10 GB free → **FAILS** ❌

With `shuffle_buffer_size=100`:
```
Peak disk usage: ~160 MB → SUCCEEDS ✅
```

---

## Action Plan

### For Testing Checkpoint Functionality
```bash
# Use the small buffer test
export GCS_TEST_BUCKET='gs://your-bucket/test-checkpoints'
python scripts/tests/test_dataloader_checkpoint_small_buffer.py --config-name=your_config
```

### For Production Checkpointing

**Option A**: Reduce buffer in config
```python
shuffle_buffer_size: int = 10_000  # Good compromise
```

**Option B**: Use massive disk
```bash
# Attach a 500GB+ persistent disk
export TMPDIR=/mnt/large-disk/tmp
```

**Option C**: Save to local disk first, then copy to GCS
```bash
export LOCAL_STAGING_DIR=/mnt/large-disk/checkpoints
python scripts/tests/test_dataloader_checkpoint_local_staging.py --config-name=your_config
```

---

## Questions?

**Q: Can I just disable the shuffle buffer?**
A: Not recommended for training - you need shuffling for good training. But for testing checkpoints, yes, use buffer=100.

**Q: Why not use a different checkpointing method?**
A: TensorFlow's iterator checkpoint is the standard way. Alternative: checkpoint only the batch counter and reset the iterator (but then you lose exact position).

**Q: Can I compress the checkpoints?**
A: The images are already compressed. TensorFlow checkpoints use efficient serialization. There's not much room for additional compression.

**Q: Will this affect my training?**
A: The test script only modifies the config for testing. Your production training config is unchanged.
