# Dataloader Checkpoint Testing - Size Issue Fix

## ⚠️ THE PROBLEM

You have **10+ GB free** but still getting:
```
INTERNAL: Could not append to the internal temporary file.
```

**Why?** Your shuffle buffer has **250,000-400,000 samples** with images. The checkpoint is **50-100+ GB**, not 10 GB!

## ✅ THE SOLUTION

Use the test with a **tiny shuffle buffer** (reduces checkpoint from 100 GB → 160 MB):

```bash
export GCS_TEST_BUCKET='gs://your-bucket/test-checkpoints'
python scripts/tests/test_dataloader_checkpoint_small_buffer.py --config-name=your_config
```

This will:
1. Show you the original estimated checkpoint size (~100+ GB)
2. Reduce shuffle_buffer_size from 250,000 → 100
3. Show you the new estimated size (~160 MB)
4. Run the test successfully

## 📊 Why So Large?

Read the full explanation: [WHY_SO_LARGE.md](WHY_SO_LARGE.md)

**Quick summary**:
- Default: `shuffle_buffer_size = 250,000`
- Each sample: ~500 KB (images + state + actions)
- Checkpoint saves **entire buffer**: 250,000 × 500 KB = **125 GB**
- Plus TensorFlow overhead: **~187 GB total needed**

## 📁 Test Scripts Available

### 1. `test_dataloader_checkpoint_small_buffer.py` ⭐ RECOMMENDED
- **Best for**: Testing checkpoint functionality
- **Checkpoint size**: ~160 MB
- **Disk needed**: ~500 MB
- Automatically reduces shuffle buffer for testing

```bash
export GCS_TEST_BUCKET='gs://your-bucket/test'
python scripts/tests/test_dataloader_checkpoint_small_buffer.py --config-name=your_config
```

### 2. `test_dataloader_checkpoint_local_staging.py`
- **Best for**: When you have a larger local disk but limited /tmp
- **Checkpoint size**: Depends on buffer size
- **Disk needed**: Configurable
- Saves to local disk first, then copies to GCS

```bash
export LOCAL_STAGING_DIR=/mnt/large-disk/checkpoints
export GCS_TEST_BUCKET='gs://your-bucket/test'  # Optional
python scripts/tests/test_dataloader_checkpoint_local_staging.py --config-name=your_config
```

### 3. `test_dataloader_checkpoint.py` (Original)
- **Best for**: When you have 200+ GB free space
- **Checkpoint size**: ~100+ GB with default config
- **Disk needed**: 200+ GB
- Uses full production shuffle buffer size

```bash
export TMPDIR=/mnt/massive-disk/tmp  # Must have 200+ GB
export GCS_TEST_BUCKET='gs://your-bucket/test'
python scripts/tests/test_dataloader_checkpoint.py --config-name=your_config
```

## 🔧 Diagnostic Tools

### Check your disk space:
```bash
bash scripts/tests/check_disk_space.sh
```

### Estimate checkpoint size for your config:
The small buffer test will show you:
- Original size: `ORIGINAL CONFIG CHECKPOINT SIZE ESTIMATE: 187.5 GB`
- Reduced size: `NEW CONFIG CHECKPOINT SIZE ESTIMATE: 0.16 GB`

## 🚀 Quick Start

**Just want to test if checkpointing works?**

```bash
# Set your GCS bucket
export GCS_TEST_BUCKET='gs://your-bucket/test-checkpoints'

# Run the small buffer test (works even with only 1-2 GB free)
python scripts/tests/test_dataloader_checkpoint_small_buffer.py --config-name=your_config

# You should see:
# ORIGINAL CONFIG CHECKPOINT SIZE ESTIMATE: 187.5 GB ← Would fail
# NEW CONFIG CHECKPOINT SIZE ESTIMATE: 0.16 GB ← Will succeed
# ✓ ALL TESTS PASSED
```

## 📖 Documentation

- **[WHY_SO_LARGE.md](WHY_SO_LARGE.md)**: Detailed explanation of checkpoint sizes
- **[QUICK_FIX.md](QUICK_FIX.md)**: Step-by-step troubleshooting guide
- **[DISK_SPACE_SOLUTIONS.md](DISK_SPACE_SOLUTIONS.md)**: Multiple approaches to fix disk space issues

## ❓ FAQ

**Q: I have 10 GB free, why isn't that enough?**
A: With 250,000 samples in the shuffle buffer, your checkpoint is actually **~187 GB**, not 10 GB.

**Q: Will the small buffer test affect my training?**
A: No. It only modifies the config temporarily for testing. Your production training is unchanged.

**Q: What if I need to checkpoint with the full shuffle buffer in production?**
A: You need either:
1. A disk with 200+ GB free space
2. Or reduce `shuffle_buffer_size` to 10,000 in your config (checkpoint becomes ~15 GB)

**Q: Can I use the original test script?**
A: Yes, but you need to:
```bash
# Set TMPDIR to a disk with 200+ GB
export TMPDIR=/mnt/large-disk/tmp
export GCS_TEST_BUCKET='gs://your-bucket/test'
python scripts/tests/test_dataloader_checkpoint.py --config-name=your_config
```

**Q: Which script should I use?**
- **For testing**: `test_dataloader_checkpoint_small_buffer.py` (easiest)
- **For production with limited disk**: Reduce `shuffle_buffer_size` in your config to 10,000
- **For production with large disk**: Use original script with large TMPDIR

## 💡 Recommendations

### For Testing Checkpoint Functionality ✅
```bash
# Use small buffer test - always works
python scripts/tests/test_dataloader_checkpoint_small_buffer.py --config-name=your_config
```

### For Production Checkpointing

**Option 1**: Reduce shuffle buffer in config (recommended)
```python
# In your config.py or YAML
shuffle_buffer_size: 10_000  # Down from 250_000
# Checkpoint: ~15 GB instead of ~187 GB
# Training quality: Still good
```

**Option 2**: Use large disk
```bash
# Need 200+ GB free
export TMPDIR=/mnt/large-disk/tmp
```

**Option 3**: Save locally first
```bash
export LOCAL_STAGING_DIR=/mnt/large-disk/checkpoints
python scripts/tests/test_dataloader_checkpoint_local_staging.py
```

---

## Summary

| Script | Checkpoint Size | Disk Needed | Best For |
|--------|----------------|-------------|----------|
| `small_buffer.py` ⭐ | ~160 MB | ~500 MB | **Testing** |
| `local_staging.py` | ~15-100 GB | Configurable | **Limited /tmp** |
| Original | ~187 GB | 200+ GB | **Production** |

**Start here**: `test_dataloader_checkpoint_small_buffer.py`
