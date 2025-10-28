# QUICK FIX: "No Space Left on Device" Error

## The Problem

You're getting:
```
INTERNAL: Could not append to the internal temporary file.
```

**Root cause**: TensorFlow needs **local disk space** even when saving to GCS. It creates temporary staging files locally before uploading to GCS.

## Immediate Solution: Use Local Staging Version

Use the **new test script** that handles this correctly:

### Step 1: Find a disk with space (10+ GB recommended)

Check your disks:
```bash
df -h
```

Look for a mount point with plenty of free space, like:
- `/mnt/data`
- `/home`
- An attached persistent disk

### Step 2: Set LOCAL_STAGING_DIR

```bash
# Use a path on a disk with sufficient space
export LOCAL_STAGING_DIR=/mnt/large-disk/checkpoints

# Create the directory
mkdir -p $LOCAL_STAGING_DIR

# Verify it has space
df -h $LOCAL_STAGING_DIR
```

### Step 3: Run the new test

```bash
# Run the local staging version (saves locally, optionally copies to GCS)
python scripts/tests/test_dataloader_checkpoint_local_staging.py --config-name=your_config
```

**Optional**: If you also want to copy to GCS:
```bash
export GCS_TEST_BUCKET='gs://your-bucket/test-checkpoints'
python scripts/tests/test_dataloader_checkpoint_local_staging.py --config-name=your_config
```

## Alternative: Fix /tmp Space

If you want to keep using the original test:

### Option A: Point TMPDIR to a larger disk

```bash
export TMPDIR=/mnt/large-disk/tmp
mkdir -p $TMPDIR
df -h $TMPDIR  # Verify it has 10+ GB

export GCS_TEST_BUCKET='gs://your-bucket/test-checkpoints'
python scripts/tests/test_dataloader_checkpoint.py --config-name=your_config
```

### Option B: Increase /tmp size (if using tmpfs)

```bash
# Check if /tmp is tmpfs
mount | grep /tmp

# If it's tmpfs, increase its size
sudo mount -o remount,size=20G /tmp

# Verify
df -h /tmp
```

## Quick Diagnostic

Run this to check your setup:
```bash
bash scripts/tests/check_disk_space.sh
```

## Comparison of Test Scripts

### test_dataloader_checkpoint.py (Original)
- Saves directly to GCS
- Still needs local temp space for TensorFlow staging
- **Use if**: You have 10+ GB free in /tmp or can set TMPDIR

### test_dataloader_checkpoint_local_staging.py (New - Recommended)
- Saves to local disk first
- Optionally copies to GCS afterward
- Checks disk space before running
- Cleans up after test
- **Use if**: You have limited /tmp space but have other disks with space

## Example: Complete Working Setup

```bash
# 1. Find a disk with space
df -h
# Let's say /home has 50GB free

# 2. Set staging directory
export LOCAL_STAGING_DIR=/home/$USER/dataloader_test_staging
mkdir -p $LOCAL_STAGING_DIR

# 3. (Optional) Set GCS bucket
export GCS_TEST_BUCKET='gs://my-bucket/test-checkpoints'

# 4. Run test
python scripts/tests/test_dataloader_checkpoint_local_staging.py --config-name=your_config

# The test will:
# - Check that /home/$USER/dataloader_test_staging has enough space
# - Save checkpoints locally to that directory
# - Copy to GCS if GCS_TEST_BUCKET is set
# - Clean up local files after completion
```

## Still Having Issues?

1. **Check actual disk usage during test**:
   ```bash
   # In another terminal while test runs:
   watch -n 1 'df -h'
   ```

2. **Check TensorFlow is using your TMPDIR**:
   ```bash
   echo $TMPDIR
   # Should show your custom path
   ```

3. **Reduce checkpoint size** (edit test):
   - Reduce `num_batches_before_save` from 10 to 3
   - This creates smaller checkpoints

4. **Check for permission issues**:
   ```bash
   ls -ld $LOCAL_STAGING_DIR
   # Should be writable by your user
   ```
