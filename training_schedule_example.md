# Training Schedule Example: Language Action + Prediction Loss

This guide shows how to use the `build_langact_prediction_schedule()` function to train with both language action loss and prediction loss.

## Schedule Configuration

The schedule is configured with:
- **Language action loss weight**: 1.0 (always applied)
- **Prediction loss weight**: 0.2
- **Prediction loss probability**: 0.2 (applied stochastically to 20% of samples)

## Usage in Python Code

```python
from openpi_cot.training.config import (
    get_config,
    build_langact_prediction_schedule,
)
import dataclasses

# Load base config
base_config = get_config("pi_droid_cot_v4")

# Create training schedule
schedule = build_langact_prediction_schedule(
    num_train_steps=100_000,
    langact_weight=1.0,
    prediction_weight=0.2,
    prediction_prob=0.2,
)

# Apply schedule to config
config_with_schedule = dataclasses.replace(
    base_config,
    training_schedule=schedule,
)
```

## Usage from Command Line

### Method 1: Using tyro's nested syntax

```bash
python -m openpi_cot.training.train \
    pi_droid_cot_v4 \
    --training_schedule.stages '[
        {
            "start_step": 0,
            "end_step": null,
            "enable_langact_training": true,
            "enable_prediction_training": true,
            "language_loss_weight": 1.0,
            "prediction_loss_weight": 0.2,
            "langact_prob": 1.0,
            "prediction_prob": 0.2
        }
    ]'
```

### Method 2: Create a custom config

Add to `config.py` in the `_CONFIGS` list:

```python
ConfigBuilder("pi_droid_cot")
    .with_model(build_picot_model())
    .with_data(
        RLDSCoTDataConfig,
        repo_id="droid",
        asset_id="droid",
        dataset_type="droid",
        droid_dataset_name="droid",
    )
    .with_training(
        weight_loader=weight_loaders.WeightLoaderChoice(kind="paligemma"),
        training_schedule=build_langact_prediction_schedule(
            num_train_steps=100_000,
            langact_weight=1.0,
            prediction_weight=0.2,
            prediction_prob=0.2,
        ),
    )
    .build(name="pi_droid_cot_langact_pred_v4", fsdp_devices=4, batch_size=256,
           checkpoint_base_dir="gs://pi0-cot/checkpoints"),
```

Then run:

```bash
python -m openpi_cot.training.train pi_droid_cot_langact_pred_v4
```

### Method 3: Override at runtime with tyro

```bash
python -m openpi_cot.training.train \
    pi_droid_cot_v4 \
    --training_schedule.stages.0.enable_prediction_training true \
    --training_schedule.stages.0.prediction_loss_weight 0.2 \
    --training_schedule.stages.0.prediction_prob 0.2
```

## Customizing the Schedule

### Different weights and probabilities

```python
# Higher prediction weight, lower probability
schedule = build_langact_prediction_schedule(
    num_train_steps=100_000,
    langact_weight=1.0,
    prediction_weight=0.5,    # Increased from 0.2
    prediction_prob=0.1,      # Decreased from 0.2
)
```

### Multi-stage curriculum

For more complex training curriculums with multiple stages:

```python
schedule = TrainingSchedule(
    stages=(
        # Stage 1: Language action only (0-20k steps)
        TrainingStage(
            start_step=0,
            end_step=20_000,
            enable_langact_training=True,
            enable_prediction_training=False,
            language_loss_weight=1.0,
            langact_prob=1.0,
        ),
        # Stage 2: Add prediction loss (20k-50k steps)
        TrainingStage(
            start_step=20_000,
            end_step=50_000,
            enable_langact_training=True,
            enable_prediction_training=True,
            language_loss_weight=1.0,
            prediction_loss_weight=0.2,
            langact_prob=1.0,
            prediction_prob=0.2,
        ),
        # Stage 3: Increase prediction probability (50k+ steps)
        TrainingStage(
            start_step=50_000,
            end_step=None,
            enable_langact_training=True,
            enable_prediction_training=True,
            language_loss_weight=1.0,
            prediction_loss_weight=0.2,
            langact_prob=1.0,
            prediction_prob=0.5,  # Increased to 50%
        ),
    )
)
```

## Verification

To verify the schedule is correctly configured:

```python
from openpi_cot.training.config import build_langact_prediction_schedule

schedule = build_langact_prediction_schedule()

# Check stage at step 0
stage = schedule.get_stage_for_step(0)
print(f"Language action enabled: {stage.enable_langact_training}")
print(f"Prediction enabled: {stage.enable_prediction_training}")
print(f"Language weight: {stage.language_loss_weight}")
print(f"Prediction weight: {stage.prediction_loss_weight}")
print(f"Prediction prob: {stage.prediction_prob}")

# Validate for training
schedule.validate_for_training(num_train_steps=100_000)
print("Schedule is valid!")
```

Expected output:
```
Language action enabled: True
Prediction enabled: True
Language weight: 1.0
Prediction weight: 0.2
Prediction prob: 0.2
Schedule is valid!
```
