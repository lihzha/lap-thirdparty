# GCS Class B Operations During Training

This document identifies all locations where Google Cloud Storage (GCS) Class B operations occur during training. Class B operations include:
- **List operations**: `list_blobs`, `list_objects`, `glob`, `listdir`
- **Get operations**: `exists`, `stat`, `get_metadata`, `isdir`

These operations incur costs and can slow down training if called frequently.

## Primary Sources

### 1. Dataset Statistics Checking (`normalize_adapter.py`)

**Location**: `src/openpi_cot/shared/adapters/normalize_adapter.py`

**Function**: `check_dataset_statistics(save_dir)`

**Lines**: 62-86

**Operations**:
- `tf.io.gfile.exists(preferred_dir)` - Line 72 (Class B: GET)
- `tf.io.gfile.exists(local_dir)` - Line 79 (Class B: GET)

**Impact**: Called for **EVERY dataset** during initialization in `base_dataset.py:99`. If you have 20 datasets, this results in 40+ Class B operations just for statistics checking.

**Code Flow**:
```
base_dataset.py:99 → check_dataset_statistics() → tf.io.gfile.exists() × 2 per dataset
```

### 2. Bounding Box Dataset Initialization (`bbox_common.py`)

**Location**: `src/openpi_cot/datasets/vqa/bbox_common.py`

**Operations**:
- `tf.io.gfile.glob(os.path.join(bbox_annotations_dir, "*.jsonl"))` - Lines 652, 978, 1176, 1310, 1491, 1567

**Impact**: Called during initialization of VQA datasets (droid_bbox, bridge_bbox, molmoact_bbox, etc.). Each `glob()` call is a Class B LIST operation that can be expensive if there are many JSONL files.

**Affected Datasets**:
- `DroidBoundingBoxDataset`
- `BridgeBoundingBoxDataset`
- `MolmoActBoundingBoxDataset`
- Other bbox datasets

### 3. Bounding Box Dataset Initialization (`oxe_bbox_dataset.py`)

**Location**: `src/openpi_cot/datasets/vqa/oxe_bbox_dataset.py`

**Operations**:
- `tf.io.gfile.glob(os.path.join(self.bbox_annotations_dir, "*.jsonl"))` - Lines 695, 743

**Impact**: Called during dataset initialization and debug methods.

### 4. Dataset Download/Validation (`download.py`)

**Location**: `src/openpi_cot/shared/download.py`

**Operations**:
- `tf.io.gfile.isdir(url)` - Line 89 (Class B: GET)
- `tf.io.gfile.exists(url)` - Line 89 (Class B: GET)
- `tf.io.gfile.exists(p)` - Line 108 (Class B: GET)
- `tf.io.gfile.isdir(p)` - Line 113 (Class B: GET)
- `tf.io.gfile.exists(_join(p, "_METADATA"))` - Line 115 (Class B: GET)
- `tf.io.gfile.exists(_join(p, "COMMIT_SUCCESS"))` - Line 115 (Class B: GET)
- `tf.io.gfile.isdir(cache_path)` - Line 123 (Class B: GET)
- `tf.io.gfile.isdir(scratch_path)` - Line 144 (Class B: GET)
- `tf.io.gfile.isdir(cache_path)` - Line 174 (Class B: GET)
- `tf.io.gfile.exists(cache_path)` - Line 176 (Class B: GET)
- `tf.io.gfile.isdir(scratch_path)` - Line 194 (Class B: GET)
- `tf.io.gfile.isdir(url)` - Line 281 (Class B: GET)
- `tf.io.gfile.exists(marker)` - Line 343 (Class B: GET)

**Impact**: Called during dataset downloading and validation. Multiple checks per dataset download.

### 5. DataLoader Checkpoint Operations (`cot_data_loader.py`)

**Location**: `src/openpi_cot/datasets/cot_data_loader.py`

**Operations**:
- `tf.io.gfile.exists(checkpoint_dir)` - Line 414 (Class B: GET)
- `tf.io.gfile.exists(checkpoint_path)` - Line 468 (Class B: GET)
- `tf.io.gfile.exists(process_0_checkpoint_path)` - Line 475 (Class B: GET)

**Impact**: Called during checkpoint save/load operations for dataloader state.

### 6. Checkpoint Operations (`checkpoints.py`)

**Location**: `src/openpi_cot/training/checkpoints.py`

**Operations**:
- `tf.io.gfile.exists(str(checkpoint_dir))` - Line 81 (Class B: GET)
- `tf.io.gfile.exists(dataloader_dir)` - Line 391 (Class B: GET)

**Impact**: Called during checkpoint initialization and restoration.

## Summary by Frequency

### High Frequency (Called per dataset during initialization):
1. **`check_dataset_statistics()`** - 2 Class B operations per dataset
   - If you have 20 datasets: **40 Class B operations** just for stats checking
   
2. **`tf.io.gfile.glob()` in bbox datasets** - 1 Class B operation per VQA dataset
   - If you have 5 VQA datasets: **5 Class B operations**

### Medium Frequency (Called during checkpointing):
3. **DataLoader checkpoint operations** - 1-3 Class B operations per checkpoint save/load
4. **Checkpoint manager operations** - 1-2 Class B operations per checkpoint

### Lower Frequency (Called during dataset download):
5. **Download validation** - Multiple Class B operations, but only during initial setup

## Recommendations

1. **Cache statistics checks**: The `check_dataset_statistics()` function already tries to load from cache, but still performs 2 `exists()` calls. Consider caching the result in memory after first check.

2. **Batch glob operations**: For bbox datasets, consider listing all JSONL files once and caching the result rather than calling `glob()` multiple times.

3. **Reduce checkpoint checks**: Minimize `exists()` calls in checkpoint paths by using try/except instead of checking existence first.

4. **Use local caching**: For frequently accessed paths, consider maintaining a local cache of existence checks.

## Cost Impact

- **Class B operations cost**: ~$0.0004 per 1,000 operations (Regional Standard)
- **During training with 20 datasets**: ~40 operations per initialization
- **If training restarts frequently**: These operations add up quickly

## Files to Modify

Priority order:
1. `src/openpi_cot/shared/adapters/normalize_adapter.py` - `check_dataset_statistics()`
2. `src/openpi_cot/datasets/vqa/bbox_common.py` - `tf.io.gfile.glob()` calls
3. `src/openpi_cot/datasets/vqa/oxe_bbox_dataset.py` - `tf.io.gfile.glob()` calls
4. `src/openpi_cot/datasets/cot_data_loader.py` - Checkpoint existence checks
5. `src/openpi_cot/shared/download.py` - Download validation checks
