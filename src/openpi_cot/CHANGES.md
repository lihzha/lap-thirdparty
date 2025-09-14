### Models

* Add: `helpers.py`, `pi0_cot_config.py`, `pi0_cot.py`, `pi0_cot_test.py`.
* Update `model.py`:

  * Add `ModelType.PI0CoT`.
  * Extend `Observation` dataclass.
  * Update `preprocess_observation` for CoT.
* Update `gemma.py`:

  * `Embedder`: add `decode`.
  * `Module`: add `decode`.

### Policies

* Add: `policies/droid_cot_policy.py`.
* Update: `policies/policy_config.py`, `policies/policy.py` to integrate CoT policy.

### Serving

* Update: `serving/WebsocketPolicyServer` to `infer_reasoning`.

### Shared

* Update: `shared/download.py` with `gs://` (GCS) support.
* Update: `shared/normalize.py` save/load with `gs://` support.
* Add: multi-host sharding support in `shared`.

### Training

* Update: `training/checkpoints.py` with `gs://` support.
* Update: base data class in `training/config.py` for openpi-cot args; add standalone configs.
* Update: `training/data_loader.py`:

  * Add adaptive CoT data loader (`cot=True`).
  * Add CoT data loader entry points.
* Add: CoT dataset in `training/droid_rlds_dataset.py`.
* Add: `training/eval_helper.py`.
* Update: `training/sharding.py`:

  * Add standalone logging helpers.
  * Update `make_mesh` and global axis naming for multi-process sharding.
* Add: standalone helpers to `training/utils.py`.
* Update: `training/weight_loaders.py` with `gs://` support.

### Transforms

* Add: CoT tokenization transform in `transforms.py`.