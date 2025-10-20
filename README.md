# openpi-cot

Add language CoT to $\pi$-sery models published by the [Physical Intelligence team](https://www.physicalintelligence.company/).

## Installation

When cloning this repo, make sure to update submodules:

```bash
git clone --recurse-submodules git@github.com:lihzha/openpi-cot.git

# Or if you already cloned the repo:
git submodule update --init --recursive
```

We use [uv](https://docs.astral.sh/uv/) to manage Python dependencies. See the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/) to set it up. Once uv is installed, run the following to set up the environment:

```bash
uv sync
```


## Running Real Robot Experiments

We provide detailed scripts for running inference of CoT checkpoints on [DROID](scripts/real_robot/droid_main.py) and [IRoM-Franka](scripts/real_robot/franka_main.py) robots.


### Step 1: Start a policy server

Since the DROID control laptop does not have a powerful GPU, we will start a remote policy server on a different machine with a more powerful GPU and then query it from the DROID control laptop during inference.

1. On a machine with a powerful GPU (~NVIDIA 4090), clone and install the `openpi` repository following the instructions in the [README](https://github.com/Physical-Intelligence/openpi).
2. Start the OpenPI server via the following command:

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=pi_droid_cot_local --policy.dir=<YOUR_POLICY_DIR>
```

### Step 2: Run the DROID robot

1. Make sure you have the most recent version of the DROID package installed on both the DROID control laptop and the NUC.
2. On the control laptop, activate your DROID conda environment.
3. Clone the openpi repo and install the openpi client, which we will use to connect to the policy server (this has very few dependencies and should be very fast to install): with the DROID conda environment activated, run `cd $OPENPI_ROOT/packages/openpi-client && pip install -e .`.
4. Install `tyro`, which we will use for command line parsing: `pip install tyro`.

```bash
python3 scripts/real_robot/droid_main.py --external_camera="right" --no-in-camera-frame --use-wrist-camera --no-run-upstream
```


## Training on DROID using language actions

On TPU, a sample command is :

```bash
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi_droid_cot_v6 --exp-name=my_experiment --fsdp-devices=64 --batch-size=2048 --exp-name v6_bs2048_pi05_0f --weight-loader.kind=checkpoint --weight-loader.params-path=gs://openpi-assets/checkpoints/pi05_base/params --data.use-wrist-image --resume --model.pi05 --model.discrete_state_input ---model.max_token_len=130
```

The command will log training progress to the console and save checkpoints to the `gs://<BUCKET_NAME>/checkpoints` directory. You can also monitor training progress on the Weights & Biases dashboard. For maximally using the TPU memory, set `XLA_PYTHON_CLIENT_MEM_FRACTION=0.9` before running training -- this enables JAX to use up to 90% of the TPU memory (vs. the default of 75%).

## Running eval

uv run scripts/eval.py \
      pi_droid_cot_v4 \
      --exp-name=test \
      --eval-checkpoint-step=26500 \
      --num-eval-batches=1000 \
      --eval-mode=token_accuracy

## Running VQA
uv run scripts/vqa.py policy:checkpoint --policy.config=paligemma2_vqa_v4 --policy.dir=None

## New Features Summary

* **Chain-of-Thought (CoT) Training Path**

  * New `PI0CoT` model type, CoT-aware `Observation`, and CoT tokenization transform.
  * CoT policy (`droid_cot_policy.py`) and serving support for **reasoning inference** via `WebsocketPolicyServer.infer_reasoning`.

* **CoT Datasets & Data Loading**

  * CoT RLDS dataset for DROID with language-action integration.
  * Adaptive data loader: enable with `cot=True`; works with multi-host setups.

* **Multi-Host / Multi-Process Sharding**

  * Improved `make_mesh`, axis naming, and logging.
  * End-to-end per-host batching and sharded device placement.

* **GCS (`gs://`) IO Support**

  * Unified read/write for downloads, normalization stats, checkpoints, and weight loading.

* **Model & Embedder Enhancements**

  * `gemma.Embedder.decode` and `Module.decode` for text decoding workflows.

* **Config & Utilities**

  * New OpenPI-CoT configs and training arguments.
  * Additional utilities and evaluation helpers.
  * Added tests for CoT model components (`pi0_cot_test.py`).