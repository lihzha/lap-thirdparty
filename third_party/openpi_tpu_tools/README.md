# openpi-tpu-tools

Unified TPU utilities and watcher for OpenPI-CoT across **v4 / v5 / v6**.

## Installation (no uv required)

Recommended (isolated):

```bash
# Install with pipx
pipx install /Users/lihanzha/code/openpi-cot/third_party/openpi_tpu_tools

# or install for current user with pip
python -m pip install --user /Users/lihanzha/code/openpi-cot/third_party/openpi_tpu_tools

# Ensure ~/.local/bin is on PATH (for --user installs)
export PATH="$HOME/.local/bin:$PATH"

# Verify
tpu-tools --help
```

## Watch & Run

Replaces the old `watch_and_run*.sh` scripts.

```bash
# v6 example (8 workers)
tpu-tools watch v6 -f -n 8 -- <extra args>

# v5 example (16 workers)
tpu-tools watch v5 -f -n 16 -- <extra args>

# v4 example (8 workers, maps 8→2x2x2 topology)
tpu-tools watch v4 -f -n 8 -- <extra args>
```

Use `--` to separate TPU arguments from training script arguments:

```bash
tpu-tools watch v6 -f -n 8 -- --config.some_flag=value
```

---

## Utility Commands

Replaces functions from `.tpu_funcs.sh`:

| Command                                              | Description                     |
| ---------------------------------------------------- | ------------------------------- |
| `tpu-tools list v6`                                 | List TPUs in v6                 |
| `tpu-tools delete v6`                               | Delete current TPU              |
| `tpu-tools delete-name v6 NAME`                     | Delete TPU by name              |
| `tpu-tools tmux v6 --session s <cmd>`               | Run command in tmux on TPU      |
| `tpu-tools attach v6 --session s --worker 0`        | Attach to tmux session worker 0 |
| `tpu-tools tmux-ls v6`                              | List tmux sessions              |
| `tpu-tools tail v6 --worker 0`                      | Tail last log for worker 0      |
| `tpu-tools tmux-kill-all v6`                        | Kill all tmux sessions          |
| `tpu-tools kill-jax v6`                             | Kill all JAX processes          |
| `tpu-tools clean-tmp v6`                            | Clean `/tmp` on TPU             |
| `tpu-tools nuke v6`                                 | Kill tmux, JAX, and clean tmp   |

---

## Environment Setup

Match your existing `~/.tpu_env.sh`:

```bash
export TPU_NAME=pi0-cot
export TPU_PROJECT=mae-irom-lab-guided-data
export TPU_ZONE_v4=us-central2-b
export TPU_ZONE_v5=us-central1-a
export TPU_ZONE_v6=us-east1-d
export TPU_BUCKET_v4=gs://pi0-cot
export TPU_BUCKET_v5=gs://v5_central1_a
export TPU_BUCKET_v6=gs://v6_east1d
```

Optional SSH settings:

```
GCLOUD_SSH_KEY_FILE
SSH_CONNECT_TIMEOUT
SSH_ALIVE_INTERVAL
SSH_ALIVE_COUNT_MAX
SSH_TOTAL_TIMEOUT
SSH_KILL_AFTER
DESCRIBE_TIMEOUT
SLEEP_SECS
```

---

## Raw SSH Aliases (no tmux)

If you source your local helpers (e.g. `~/.tpu_funcs.sh`), you can send commands to all workers without tmux:

```bash
# Examples (run on all workers)
v4 "hostname"
v5 "uname -a"
v6 "nvidia-smi || true"
```

These wrappers use `gcloud compute tpus tpu-vm ssh` directly and respect your `TPU_*` env vars.

---

## Package Structure

```
third_party/openpi_tpu_tools/
  pyproject.toml      # Console script: tpu-tools
  src/tpu_tools/
    config.py         # Env loader for v4/v5/v6
    ssh.py            # gcloud SSH wrapper w/ timeouts
    tpu.py            # List/delete/tmux/kill/nuke helpers
    watch.py          # Watch-and-run logic
    cli.py            # CLI dispatcher
    __init__.py
  README.md
  LICENSE
```

---

## Help

```bash
tpu-tools --help
```