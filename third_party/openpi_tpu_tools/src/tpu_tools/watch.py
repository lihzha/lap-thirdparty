from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
import signal
import sys

from .config import TPUEnvConfig
from .tpu import TPUManager


def _ts() -> str:
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _map_v4_topology(tpu_num: int) -> str:
    mapping = {4: "2x2x1", 8: "2x2x2", 16: "2x2x4", 32: "2x4x4"}
    if tpu_num not in mapping:
        raise SystemExit(f"Error: unsupported TPU_NUM '{tpu_num}' (allowed: 4, 8, 16, 32)")
    return mapping[tpu_num]


@dataclass
class WatchConfig:
    version: str  # v4/v5/v6
    force_run: bool
    tpu_num: int
    extra_args: list[str]


def watch_and_run(cfg: WatchConfig, env: TPUEnvConfig) -> None:
    mgr = TPUManager(env)

    print("Starting TPU auto-launcher with:")
    print(f"  TPU Name: {env.tpu_name}")
    print(f"  Zone: {getattr(env, f'tpu_zone_{cfg.version}')}")
    print(f"  Project: {env.tpu_project}")
    bucket = getattr(env, f"tpu_bucket_{cfg.version}")
    print(f"  Bucket: {bucket}")
    print(f"  TPU Num: {cfg.tpu_num}")
    if cfg.version == "v4":
        print(f"  Topology: {_map_v4_topology(cfg.tpu_num)}")
    print(f"  Force run: {cfg.force_run}")
    if cfg.extra_args:
        print(f"  Extra args: {' '.join(cfg.extra_args)}")
    print()

    def handle_sig(signum, frame):
        print(f"{_ts()} - Caught signal, exiting.")
        raise SystemExit(0)

    signal.signal(signal.SIGINT, handle_sig)
    signal.signal(signal.SIGTERM, handle_sig)

    while True:
        print(f"{_ts()} - Checking TPU state...")
        try:
            state = mgr.describe(cfg.version)  # may raise for invalid zone
        except Exception as exc:
            print(str(exc))
            os.sleep(mgr.sleep_secs)
            continue

        print(f"{_ts()} - TPU {env.tpu_name} state: {state}")

        run_setup_and_training = False

        if state in {"NOT_FOUND", "PREEMPTED", "STOPPED"}:
            print(f"{_ts()} - Need to (re)create TPU...")
            if state != "NOT_FOUND":
                if not mgr.delete(cfg.version):
                    print(f"{_ts()} - Delete failed/timed out.")
                    from time import sleep

                    sleep(mgr.sleep_secs)
                    continue
            print(f"{_ts()} - Creating new TPU...")
            topo = _map_v4_topology(cfg.tpu_num) if cfg.version == "v4" else None
            if not mgr.create(cfg.version, tpu_num=cfg.tpu_num, topology=topo):
                print(f"{_ts()} - Create failed/timed out.")
                from time import sleep

                sleep(mgr.sleep_secs)
                continue
            print(f"{_ts()} - Waiting for TPU to be READY...")
            from time import sleep

            sleep(10)
            run_setup_and_training = True
        elif state == "PERMISSION_DENIED":
            print(f"{_ts()} - PERMISSION_DENIED from describe. Check IAM/API enablement.")
            from time import sleep

            sleep(mgr.sleep_secs)
            continue
        elif state == "READY":
            run_setup_and_training = cfg.force_run
        else:
            print(f"{_ts()} - TPU in state: {state} (not actionable now).")
            from time import sleep

            sleep(mgr.sleep_secs)
            continue

        if run_setup_and_training:
            print(f"{_ts()} - Setting up environment and repository...")
            bucket_env = {
                "v4": env.tpu_bucket_v4,
                "v5": env.tpu_bucket_v5,
                "v6": env.tpu_bucket_v6,
            }[cfg.version]
            openpi_data_home = f"{bucket_env}/cache"
            setup_cmd = (
                "curl -LsSf https://astral.sh/uv/install.sh | sh && "
                "echo 'export WANDB_API_KEY=9d133998a3d44bf5dd2d827a5d8e2710dc91a19b' >> ~/.zshrc && "
                f"echo 'export OPENPI_DATA_HOME=\"{openpi_data_home}\"' >> ~/.zshrc && "
                "source ~/.zshrc && "
                # Configure git identity (safe defaults) and token auth for GitHub if provided
                'git config --global user.name "lihzha" && '
                'git config --global user.email "lihanzha20@gmail.com" && '
                # Set credential helper to store to avoid prompts
                "git config --global credential.helper store && "
                # If GITHUB_TOKEN is set, write credentials for github.com
                'bash -lc \'if [ -n "$GITHUB_TOKEN" ]; then mkdir -p ~/.config/git && printf "https://%s:x-oauth-basic@github.com\\n" "$GITHUB_TOKEN" > ~/.config/git/credentials; git config --global credential.helper "store --file=~/.config/git/credentials"; fi\' && '
                # Clone using token URL if available, else HTTPS public URL
                'bash -lc \'if [ -n "$GITHUB_TOKEN" ]; then REPO_URL="https://$GITHUB_TOKEN:x-oauth-basic@github.com/lihzha/openpi-cot.git"; else REPO_URL="https://github.com/lihzha/openpi-cot.git"; fi; '
                'git clone --recurse-submodules "$REPO_URL" || true\' && '
                "cd openpi-cot && "
                "uv sync"
            )
            if not mgr.tmux(cfg.version, cmd=setup_cmd, session="setup"):
                print(f"{_ts()} - Setup failed/SSH timed out. Back to state check.")
                from time import sleep

                sleep(mgr.sleep_secs)
                continue

            print(f"{_ts()} - Starting training...")
            extra = " ".join(cfg.extra_args) if cfg.extra_args else ""
            target = {"v4": "pi_droid_cot_v4", "v5": "pi_droid_cot_v5", "v6": "pi_droid_cot_v6"}[cfg.version]
            train_cmd = (
                "source ~/.zshrc && cd openpi-cot && "
                "git pull origin main && "
                "XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 "
                f"uv run --group tpu scripts/train.py {target} {extra}"
            )
            if not mgr.tmux(cfg.version, cmd=train_cmd, session="tpu"):
                print(f"{_ts()} - Launch failed/SSH timed out. Back to state check.")
                from time import sleep

                sleep(mgr.sleep_secs)
                continue

            print(f"{_ts()} - Training started successfully!")
            if cfg.force_run:
                print(f"{_ts()} - Force run requested; exiting.")
                return

        from time import sleep

        sleep(mgr.sleep_secs)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="tpu-tools watch")
    p.add_argument("version", choices=["v4", "v5", "v6"], help="TPU version to target")
    p.add_argument("--force", "-f", action="store_true", help="Force setup and training even if READY")
    p.add_argument("--tpu-num", "-n", type=int, default=8, help="TPU chips (v4: 4/8/16/32; v5:16/32/64; v6:any)")
    p.add_argument("extra", nargs=argparse.REMAINDER, help="Extra args to pass to training script")
    return p


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = build_arg_parser()
    ns = ap.parse_args(argv)
    cfg = WatchConfig(version=ns.version, force_run=ns.force, tpu_num=ns.tpu_num, extra_args=ns.extra)
    env = TPUEnvConfig.from_env()
    watch_and_run(cfg, env)
    return 0
