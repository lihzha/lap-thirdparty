# Tiger Demo Fine-tuning

This directory contains scripts for fine-tuning pi05-base on Tiger demonstration data.

## Overview

The Tiger dataset contains 53 demonstrations, each in a separate folder with:
- `data.pkl`: Proprioceptive observations and actions
  - State: arm_pos (3) + arm_quat (4) + gripper_pos (1) = 8 dimensions
  - Actions: arm_pos (3) + arm_quat (4) + gripper_pos (1) = 8 dimensions
- `base_image.mp4`: Base camera video (360x640, 10 FPS)
- `wrist_image.mp4`: Wrist camera video (360x640, 10 FPS)

## Step 1: Convert to LeRobot Format

First, convert the Tiger demos to LeRobot format:

```bash
python scripts/tiger/convert_tiger_data_to_lerobot.py --data_dir all_data/data_tiger/demos
```

This will:
- Read all 53 demonstration folders
- Extract frames from videos and align them with proprioception/action timestamps
- Create a LeRobot dataset at `$HF_LEROBOT_HOME/your_hf_username/tiger_demos`


## Step 2: Compute Normalization Statistics

Before training, compute normalization statistics for the dataset. These statistics (mean, std, min/max, quantiles) are used to normalize state and action inputs during training, which improves training stability and performance.

```bash
python scripts/compute_norm_stats.py --config-name pi05_tiger_finetune_local
```

This will:
- Load the Tiger dataset from LeRobot
- Compute statistics over all state and action data
- Save the statistics to `/n/fs/robot-data/pi0-cot/assets/tiger/norm_stats.json`

**Note**: This step takes a few minutes depending on the dataset size. For Tiger's 53 demos, it should complete in under 5 minutes.

## Step 3: Fine-tune Pi05-Base

Once the dataset is converted and normalization stats are computed, run the training script:

```bash
python scripts/train.py pi05_tiger_finetune_local --exp-name finetune
```

This will:
- Load the pi05-base checkpoint from `gs://openpi-assets/checkpoints/pi05_base/params`
- Load normalization statistics from `/n/fs/robot-data/pi0-cot/assets/tiger/norm_stats.json`
- Fine-tune on the Tiger demos
- Save checkpoints to `/n/fs/robot-data/pi0-cot/checkpoints/pi05_tiger_finetune_local/<exp_name>`

**Training Hyperparameters**:
- Batch size: 4
- Training steps: 10,000
- Save interval: 500 steps
- Log interval: 50 steps