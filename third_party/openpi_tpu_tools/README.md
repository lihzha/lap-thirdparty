# openpi-tpu-tools

Unified TPU utilities and watcher for OpenPI-CoT across **v4 / v5 / v6**.

## Installation & Quick Start

```bash
uv run tpu-tools --help
```

## Watch & Run

Replaces the old `watch_and_run*.sh` scripts.

```bash
# v6 example (8 workers)
uv run tpu-tools watch v6 -f -n 8 -- <extra args>

# v5 example (16 workers)
uv run tpu-tools watch v5 -f -n 16 -- <extra args>

# v4 example (8 workers, maps 8→2x2x2 topology)
uv run tpu-tools watch v4 -f -n 8 -- <extra args>
```

Use `--` to separate TPU arguments from training script arguments:

```bash
uv run tpu-tools watch v6 -f -n 8 -- --config.some_flag=value
```

---

## Utility Commands

Replaces functions from `.tpu_funcs.sh`:

| Command                                              | Description                     |
| ---------------------------------------------------- | ------------------------------- |
| `uv run tpu-tools list v6`                          | List TPUs in v6                 |
| `uv run tpu-tools delete v6`                        | Delete current TPU              |
| `uv run tpu-tools delete-name v6 NAME`              | Delete TPU by name              |
| `uv run tpu-tools tmux v6 --session s`              | Run command in tmux on TPU      |
| `uv run tpu-tools attach v6 --session s --worker 0` | Attach to tmux session worker 0 |
| `uv run tpu-tools tmux-ls v6`                       | List tmux sessions              |
| `uv run tpu-tools tail v6 --worker 0`               | Tail last log for worker 0      |
| `uv run tpu-tools tmux-kill-all v6`                 | Kill all tmux sessions          |
| `uv run tpu-tools kill-jax v6`                      | Kill all JAX processes          |
| `uv run tpu-tools clean-tmp v6`                     | Clean `/tmp` on TPU             |
| `uv run tpu-tools nuke v6`                          | Delete all TPUs in v6           |

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

Root `pyproject.toml` updated to include `packages/openpi-tpu` as a workspace member.

---

## Help

```bash
uv run tpu-tools --help
```