# LAP: Language Action Pre-training Enables Zero-Shot Cross-Embodiment Transfer

## Installation

When cloning this repo, make sure to update submodules:

```bash
git clone --branch third_party --recurse-submodules git@github.com:lihzha/openpi-cot.git

# Or if you already cloned the repo:
git submodule update --init --recursive
```

We use [uv](https://docs.astral.sh/uv/) to manage Python dependencies. See the [uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/) to set it up. Once uv is installed, run the following to set up the environment:

```bash
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```


## Running Real Robot Experiments

We provide detailed scripts for running inference of CoT checkpoints on [DROID](scripts/real_robot/droid_main.py) and [IRoM-Franka](scripts/real_robot/franka_main.py) robots.


### Step 1: Start a policy server

Start the policy server via the following command:

```bash
uv run scripts/serve_policy.py policy:checkpoint --policy.config=lap --policy.dir=checkpoints/lap/15000
```

### Step 2: Run the robot

1. Modify the init_env() and _extract_observation() function under `scripts/real_robot/aloha_main.py` to make sure it matches your robot environment API. Notably, you should rotate rotate wrist image to match the base frame. For example, when the robot is at its initial pose, if looking from the robot base, the object is on the left, while in the wrist image, the object is on the right, then you should rotate the wrist image by 180 degrees, i.e. wrist_image[::-1, ::-1, ::-1].
2. Under your robot control environment, run `cd third_party/openpi/packages/openpi-client && pip install -e .` Then run `pip install tyro`.
3. Under your robot control environment, Run the robot server script.

```bash

python scripts/real_robot/aloha_main.py
```
When prompted with instructions, make sure the first letter is upper-cased. For example, "Put the apple into the bowl".

I recommend starting with the most basic task, e.g., simple pick-and-place without distractors, and set the object to be picked within the wrist camera's initial view.