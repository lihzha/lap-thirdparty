# Solutions for "No Space Left on Device" Error

When running the dataloader checkpoint tests, you may encounter:
```
INTERNAL: Could not append to the internal temporary file.
```

This happens because **TensorFlow needs local temporary storage even when saving to GCS**. The checkpoint files are first written locally, then uploaded to GCS.

## Solutions

### Solution 1: Set TMPDIR to a Location with More Space

Point TensorFlow to use a directory on a disk with more available space:

```bash
# Create a temporary directory on a larger disk
mkdir -p /mnt/large-disk/tmp

# Set TMPDIR environment variable
export TMPDIR=/mnt/large-disk/tmp

# Run the test
python scripts/tests/test_dataloader_checkpoint.py --config-name=your_config
```

### Solution 2: Increase tmpfs Size (if using /tmp)

If `/tmp` is mounted as tmpfs (RAM-backed), you can increase its size:

```bash
# Check current size
df -h /tmp

# Remount with larger size (e.g., 20GB)
sudo mount -o remount,size=20G /tmp

# Verify new size
df -h /tmp

# Run the test
export GCS_TEST_BUCKET='gs://your-bucket/test-checkpoints'
python scripts/tests/test_dataloader_checkpoint.py --config-name=your_config
```

### Solution 3: Clean Up Disk Space

If you're running low on disk space, clean up unnecessary files:

```bash
# Check disk usage
df -h

# Find large directories
du -sh /* 2>/dev/null | sort -h

# Clean up common temporary files
sudo apt-get clean
sudo apt-get autoclean
docker system prune -a  # If using Docker
```

### Solution 4: Mount a Larger Volume

If you're on a cloud instance (e.g., GCP VM), attach and mount a larger persistent disk:

```bash
# Format and mount a new disk (example for /dev/sdb)
sudo mkfs.ext4 /dev/sdb
sudo mkdir -p /mnt/data
sudo mount /dev/sdb /mnt/data

# Create temp directory
sudo mkdir -p /mnt/data/tmp
sudo chmod 1777 /mnt/data/tmp

# Use it
export TMPDIR=/mnt/data/tmp
python scripts/tests/test_dataloader_checkpoint.py --config-name=your_config
```

## How Much Space Do You Need?

The checkpoint size depends on:
- Dataset buffer size (shuffle buffer, prefetch buffer)
- Number of batches iterated
- Batch size

**Estimate**: Expect 5-10 GB of temporary space for typical tests.

## Verification

The test script will automatically check available disk space and warn you if it's below 5 GB:

```
Using temporary directory: /tmp
Available space: 2.34 GB
WARNING: Low disk space in temporary directory!
```

## Example: Complete Setup

```bash
# 1. Set up temporary directory with sufficient space
export TMPDIR=/mnt/large-disk/tmp
mkdir -p $TMPDIR

# 2. Set GCS bucket for checkpoint storage
export GCS_TEST_BUCKET='gs://my-bucket/test-checkpoints'

# 3. Run the test
python scripts/tests/test_dataloader_checkpoint.py --config-name=your_config

# Output will show:
# Using temporary directory: /mnt/large-disk/tmp
# Available space: 50.00 GB
# Using GCS bucket: gs://my-bucket/test-checkpoints
```

## Why Does This Happen?

TensorFlow's checkpoint system:
1. Creates temporary files locally
2. Writes checkpoint data to temp files
3. Uploads completed files to GCS
4. Deletes local temp files

Even though the **final destination is GCS**, TensorFlow still needs **local staging space**.

## Troubleshooting

If you still get errors after trying these solutions:

1. **Check actual available space**:
   ```bash
   df -h $TMPDIR
   ```

2. **Monitor disk usage during test**:
   ```bash
   watch -n 1 'df -h $TMPDIR'
   ```

3. **Check TensorFlow temp files**:
   ```bash
   ls -lah $TMPDIR/tmp* 2>/dev/null | head
   ```

4. **Reduce checkpoint frequency**: Modify the test to save fewer checkpoints or iterate fewer batches before saving.
