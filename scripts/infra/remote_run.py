from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from .providers.common import Offer
from .providers.runpod import RunPodProvider
from .providers.vastai import VastAIProvider


DEFAULT_DOCKER_REMOTE_ROOT = "/workspace/dominion"
DEFAULT_DOCKER_CHECKPOINT_ROOT = "/workspace/checkpoints"
DEFAULT_NATIVE_REMOTE_ROOT = "/root/dominion"
DEFAULT_NATIVE_CHECKPOINT_ROOT = "/root/dominion/checkpoints"
DEFAULT_IMAGE_TAG = "dominion-v2-train:latest"
DEFAULT_VAST_PYTORCH_IMAGE = "pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime"
DEFAULT_RUNPOD_BASE_IMAGE = "nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04"


@dataclass
class CommandRunner:
    dry_run: bool = False

    def run(self, command: str) -> None:
        if self.dry_run:
            print(f"+ {command}")
            return
        subprocess.run(command, shell=True, check=True)


def quote(value: str | Path) -> str:
    return shlex.quote(str(value))


def provider_for(name: str):
    if name == "vastai":
        return VastAIProvider()
    if name == "runpod":
        return RunPodProvider()
    raise ValueError(f"unknown provider: {name}")


def default_mode(provider: str) -> str:
    return "native" if provider == "vastai" else "docker"


def mode_for(args: argparse.Namespace) -> str:
    return args.mode or default_mode(args.provider)


def remote_root_for(args: argparse.Namespace) -> str:
    if args.remote_root:
        return args.remote_root
    return DEFAULT_NATIVE_REMOTE_ROOT if mode_for(args) == "native" else DEFAULT_DOCKER_REMOTE_ROOT


def checkpoint_root_for(args: argparse.Namespace) -> str:
    if args.remote_checkpoint_root:
        return args.remote_checkpoint_root
    return DEFAULT_NATIVE_CHECKPOINT_ROOT if mode_for(args) == "native" else DEFAULT_DOCKER_CHECKPOINT_ROOT


def base_image_for(args: argparse.Namespace) -> str:
    if args.base_image:
        return args.base_image
    if mode_for(args) == "native" and args.provider == "vastai":
        return DEFAULT_VAST_PYTORCH_IMAGE
    return DEFAULT_RUNPOD_BASE_IMAGE


def print_offer_table(offers: Sequence[Offer]) -> None:
    print("id,gpu,vcpus,price_per_hour,upload_mbps")
    for offer in offers:
        print(f"{offer.id},{offer.gpu},{offer.vcpus},{offer.price_per_hour:.4f},{offer.upload_mbps:.1f}")


def resolve_ssh_target(args: argparse.Namespace) -> str:
    if args.ssh_target:
        return args.ssh_target
    if args.dry_run:
        return "root@example.invalid"
    if not args.instance_id:
        raise SystemExit("--instance-id or --ssh-target is required")
    return provider_for(args.provider).ssh_target(args.instance_id)


def remote_container_name(run: str) -> str:
    return f"dominion-train-{run}"


def remote_run_dir(run: str, checkpoint_root: str) -> str:
    return f"{checkpoint_root.rstrip('/')}/{run}"


def run_ssh_script(runner: CommandRunner, ssh_target: str, script: str) -> None:
    runner.run(f"ssh {quote(ssh_target)} 'bash -s' <<'REMOTE_SCRIPT'\n{script.rstrip()}\nREMOTE_SCRIPT")


def archive_to_remote(args: argparse.Namespace, runner: CommandRunner, ssh_target: str) -> Path:
    with tempfile.NamedTemporaryFile(prefix=f"dominion-{args.run}-", suffix=".tar.gz", delete=False) as handle:
        archive = Path(handle.name)
    runner.run(f"git archive --format=tar HEAD | gzip -9 > {quote(archive)}")
    runner.run(f"scp {quote(archive)} {quote(ssh_target)}:/tmp/dominion-src.tar.gz")
    return archive


def archive_and_build_docker(args: argparse.Namespace, runner: CommandRunner, ssh_target: str) -> None:
    remote_root = remote_root_for(args)
    archive_to_remote(args, runner, ssh_target)
    runner.run(f"ssh {quote(ssh_target)} {quote('mkdir -p ' + quote(remote_root))}")
    remote = (
        f"rm -rf {quote(remote_root)}/* && "
        f"tar -xzf /tmp/dominion-src.tar.gz -C {quote(remote_root)} && "
        f"cd {quote(remote_root)} && "
        f"docker build -f Dockerfile.train -t {quote(args.image_tag)} ."
    )
    runner.run(f"ssh {quote(ssh_target)} {quote(remote)}")


def native_torch_check_script(auto_fix_torch: bool) -> str:
    remedy = "1" if auto_fix_torch else "0"
    return f"""
check_torch_cuda() {{
python3 - <<'PY'
import subprocess
import sys
import torch

cuda = torch.version.cuda or ""
available = bool(torch.cuda.is_available())
try:
    gpu = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        text=True,
    ).strip().splitlines()[0]
except Exception:
    gpu = ""
print(f"torch={{torch.__version__}} cuda={{cuda}} cuda_available={{available}} gpu={{gpu}}")
if not available:
    print("ERROR: torch.cuda.is_available() is false; use a GPU PyTorch template or install a CUDA torch wheel", file=sys.stderr)
    raise SystemExit(20)
blackwell_markers = ("RTX 50", "5090", "5080", "5070", "B200", "B100", "GB200")
is_blackwell = any(marker in gpu for marker in blackwell_markers)
major_minor = tuple(int(part) for part in cuda.split(".")[:2] if part.isdigit())
if is_blackwell and major_minor < (12, 8):
    print("ERROR: Blackwell GPU detected but torch CUDA is older than cu128. Remedy: python3 -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128", file=sys.stderr)
    raise SystemExit(21)
PY
}}
set +e
check_torch_cuda
torch_status=$?
set -e
if [ "$torch_status" -eq 21 ] && [ "{remedy}" = "1" ]; then
  echo "Attempting automatic cu128 torch remedy"
  python3 -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128
  check_torch_cuda
elif [ "$torch_status" -ne 0 ]; then
  exit "$torch_status"
fi
"""


def bootstrap_native(args: argparse.Namespace, runner: CommandRunner, ssh_target: str) -> None:
    remote_root = remote_root_for(args)
    archive_to_remote(args, runner, ssh_target)
    script = f"""
set -euo pipefail
ROOT={quote(remote_root)}
rm -rf "$ROOT"
mkdir -p "$ROOT"
tar -xzf /tmp/dominion-src.tar.gz -C "$ROOT"
cd "$ROOT"
if ! command -v cmake >/dev/null 2>&1; then
  apt-get update
  apt-get install -y cmake build-essential
fi
python3 -m pip install --upgrade pybind11 numpy
{native_torch_check_script(args.auto_fix_torch)}
PYBIND11_DIR="$(python3 -m pybind11 --cmakedir)"
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON=ON -Dpybind11_DIR="$PYBIND11_DIR"
cmake --build build --target dominion_v2_py -j"$(nproc)"
"""
    run_ssh_script(runner, ssh_target, script)


def install_watchdog_docker(args: argparse.Namespace, runner: CommandRunner, ssh_target: str) -> None:
    run_dir = remote_run_dir(args.run, checkpoint_root_for(args))
    container = remote_container_name(args.run)
    script = (
        "while sleep 600; do "
        f"f={quote(run_dir + '/metrics.csv')}; "
        "if [ -f \"$f\" ]; then "
        "age=$(( $(date +%s) - $(stat -c %Y \"$f\") )); "
        "if [ \"$age\" -gt 3600 ]; then "
        f"echo \"WATCHDOG: metrics stale for ${{age}}s; stopping {container}\" "
        f"| tee -a {quote(run_dir + '/watchdog.log')}; "
        f"docker stop {quote(container)}; exit 0; "
        "fi; fi; done"
    )
    remote = (
        f"mkdir -p {quote(run_dir)} && "
        f"nohup bash -lc {quote(script)} > {quote(run_dir + '/watchdog.out')} 2>&1 &"
    )
    runner.run(f"ssh {quote(ssh_target)} {quote(remote)}")


def install_watchdog_native(args: argparse.Namespace, runner: CommandRunner, ssh_target: str) -> None:
    run_dir = remote_run_dir(args.run, checkpoint_root_for(args))
    pidfile = f"{run_dir}/train.pid"
    script = (
        "while sleep 600; do "
        f"f={quote(run_dir + '/metrics.csv')}; "
        f"p={quote(pidfile)}; "
        "if [ -f \"$f\" ]; then "
        "age=$(( $(date +%s) - $(stat -c %Y \"$f\") )); "
        "if [ \"$age\" -gt 3600 ]; then "
        f"echo \"WATCHDOG: metrics stale for ${{age}}s; killing training PID\" "
        f"| tee -a {quote(run_dir + '/watchdog.log')} {quote(run_dir + '/watchdog.marker')}; "
        "if [ -f \"$p\" ]; then kill $(cat \"$p\") || true; fi; exit 0; "
        "fi; fi; done"
    )
    remote = (
        f"mkdir -p {quote(run_dir)} && "
        f"nohup bash -lc {quote(script)} > {quote(run_dir + '/watchdog.out')} 2>&1 &"
    )
    runner.run(f"ssh {quote(ssh_target)} {quote(remote)}")


def launch_container(args: argparse.Namespace, runner: CommandRunner, ssh_target: str) -> None:
    run_dir = remote_run_dir(args.run, checkpoint_root_for(args))
    remote_config = f"/workspace/config-{args.run}.json"
    container = remote_container_name(args.run)
    extra = " --resume latest" if args.resume_latest else ""
    runner.run(f"scp {quote(args.config)} {quote(ssh_target)}:{quote(remote_config)}")
    remote = (
        f"mkdir -p {quote(run_dir)} && "
        f"docker rm -f {quote(container)} >/dev/null 2>&1 || true && "
        "docker run -d --restart on-failure --gpus all "
        f"--name {quote(container)} "
        f"-v {quote(run_dir)}:/workspace/checkpoints "
        f"-v {quote(remote_config)}:/workspace/config.json:ro "
        f"{quote(args.image_tag)}{extra}"
    )
    runner.run(f"ssh {quote(ssh_target)} {quote(remote)}")
    install_watchdog_docker(args, runner, ssh_target)


def launch_native(args: argparse.Namespace, runner: CommandRunner, ssh_target: str) -> None:
    remote_root = remote_root_for(args)
    run_dir = remote_run_dir(args.run, checkpoint_root_for(args))
    remote_config = f"{remote_root}/config-{args.run}.json"
    extra = " --resume latest" if args.resume_latest else ""
    runner.run(f"scp {quote(args.config)} {quote(ssh_target)}:{quote(remote_config)}")
    script = f"""
set -euo pipefail
ROOT={quote(remote_root)}
RUN_DIR={quote(run_dir)}
CONFIG={quote(remote_config)}
mkdir -p "$RUN_DIR"
cd "$ROOT"
if [ -f "$RUN_DIR/train.pid" ] && kill -0 "$(cat "$RUN_DIR/train.pid")" 2>/dev/null; then
  echo "training already running with PID $(cat "$RUN_DIR/train.pid")"
  exit 0
fi
nohup bash -lc 'cd "$0" && PYTHONPATH=build python3 -m src.v2.train.train --config "$1" --device auto --checkpoint-dir "$2"{extra}' "$ROOT" "$CONFIG" "$RUN_DIR" >> "$RUN_DIR/console.log" 2>&1 &
echo $! > "$RUN_DIR/train.pid"
echo "started native training PID $(cat "$RUN_DIR/train.pid")"
"""
    run_ssh_script(runner, ssh_target, script)
    install_watchdog_native(args, runner, ssh_target)


def smoke_docker(args: argparse.Namespace, runner: CommandRunner, ssh_target: str) -> None:
    command = (
        f"docker run --rm --gpus all {quote(args.image_tag)} --smoke --device auto "
        "| tee /tmp/dominion-train-smoke.log && "
        "grep '\"device\": \"cuda\"' /tmp/dominion-train-smoke.log"
    )
    runner.run(f"ssh {quote(ssh_target)} {quote(command)}")


def smoke_native(args: argparse.Namespace, runner: CommandRunner, ssh_target: str) -> None:
    remote_root = remote_root_for(args)
    command = (
        f"cd {quote(remote_root)} && "
        "PYTHONPATH=build python3 -m src.v2.train.train "
        "--config src/v2/train/configs/smoke.json --smoke --device auto "
        "| tee /tmp/dominion-train-smoke.log && "
        "grep '\"device\": \"cuda\"' /tmp/dominion-train-smoke.log"
    )
    runner.run(f"ssh {quote(ssh_target)} {quote(command)}")


def sync_run(args: argparse.Namespace, runner: CommandRunner, ssh_target: str) -> None:
    local = Path(args.local_sync_root) / args.run
    local.mkdir(parents=True, exist_ok=True)
    run_dir = remote_run_dir(args.run, checkpoint_root_for(args))
    runner.run(f"rsync -az --partial {quote(ssh_target)}:{quote(run_dir)}/ {quote(local)}/")


def status_run(args: argparse.Namespace, runner: CommandRunner, ssh_target: str) -> None:
    run_dir = remote_run_dir(args.run, checkpoint_root_for(args))
    if mode_for(args) == "docker":
        container = remote_container_name(args.run)
        command = (
            "set +e; "
            "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader; "
            f"docker ps --filter name={quote(container)} --format 'container={{.Names}} status={{.Status}}'; "
            f"test -f {quote(run_dir + '/watchdog.log')} && tail -5 {quote(run_dir + '/watchdog.log')} || true; "
            f"test -f {quote(run_dir + '/metrics.csv')} && tail -5 {quote(run_dir + '/metrics.csv')} || echo 'metrics.csv not found'"
        )
    else:
        command = (
            "set +e; "
            "nvidia-smi --query-gpu=name,utilization.gpu,memory.used,memory.total --format=csv,noheader; "
            f"if [ -f {quote(run_dir + '/train.pid')} ] && kill -0 $(cat {quote(run_dir + '/train.pid')}) 2>/dev/null; "
            f"then echo native_pid=$(cat {quote(run_dir + '/train.pid')}) status=running; "
            "else echo native_status=not_running; fi; "
            f"test -f {quote(run_dir + '/watchdog.marker')} && cat {quote(run_dir + '/watchdog.marker')} || true; "
            f"test -f {quote(run_dir + '/watchdog.log')} && tail -5 {quote(run_dir + '/watchdog.log')} || true; "
            f"test -f {quote(run_dir + '/metrics.csv')} && tail -5 {quote(run_dir + '/metrics.csv')} || echo 'metrics.csv not found'"
        )
    runner.run(f"ssh {quote(ssh_target)} {quote(command)}")


def logs_run(args: argparse.Namespace, runner: CommandRunner, ssh_target: str) -> None:
    if mode_for(args) == "docker":
        container = remote_container_name(args.run)
        command = f"docker logs -f --tail {int(args.lines)} {quote(container)}"
    else:
        run_dir = remote_run_dir(args.run, checkpoint_root_for(args))
        command = f"tail -f -n {int(args.lines)} {quote(run_dir + '/console.log')}"
    runner.run(f"ssh {quote(ssh_target)} {quote(command)}")


def destroy_after_sync(args: argparse.Namespace, runner: CommandRunner, ssh_target: str) -> None:
    sync_run(args, runner, ssh_target)
    if args.dry_run:
        print(f"+ destroy {args.provider} instance {args.instance_id or '<none>'}")
        return
    if not args.instance_id:
        raise SystemExit("--instance-id is required for down")
    provider = provider_for(args.provider)
    try:
        status = provider.instance_status(args.instance_id)
        if status.cost_per_hour > 0.0:
            print(f"last known price: ${status.cost_per_hour:.4f}/hr")
    except Exception as exc:  # pragma: no cover - best-effort provider metadata
        print(f"warning: could not read final instance cost: {exc}", file=sys.stderr)
    provider.destroy_instance(args.instance_id)
    print(f"destroyed {args.provider} instance {args.instance_id}")


def latest_metrics_rows(path: Path, count: int = 3) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    return rows[-count:]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Remote Dominion v2 training orchestrator")
    parser.add_argument("--provider", choices=["vastai", "runpod"], default="vastai")
    parser.add_argument("--mode", choices=["docker", "native"], default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--instance-id", default="")
    parser.add_argument("--ssh-target", default="")
    parser.add_argument("--run", default="remote")
    parser.add_argument("--remote-root", default="")
    parser.add_argument("--remote-checkpoint-root", default="")
    parser.add_argument("--local-sync-root", default="checkpoints/remote")
    parser.add_argument("--image-tag", default=DEFAULT_IMAGE_TAG)
    parser.add_argument("--auto-fix-torch", action=argparse.BooleanOptionalAction, default=True)
    sub = parser.add_subparsers(dest="command", required=True)

    offers = sub.add_parser("offers")
    offers.add_argument("--gpu", default="RTX 4090")
    offers.add_argument("--min-vcpus", type=int, default=16)
    offers.add_argument("--max-price", type=float, default=1.0)

    up = sub.add_parser("up")
    up.add_argument("--offer-id", required=True)
    up.add_argument("--base-image", default="")

    launch = sub.add_parser("launch")
    launch.add_argument("--config", default="src/v2/train/configs/run_remote.json")
    launch.add_argument("--resume-latest", action="store_true")

    sub.add_parser("smoke")
    sub.add_parser("status")

    logs = sub.add_parser("logs")
    logs.add_argument("--lines", type=int, default=100)

    sub.add_parser("sync")
    sub.add_parser("down")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    runner = CommandRunner(args.dry_run)

    if args.command == "offers":
        if args.dry_run:
            print(f"+ query {args.provider} offers gpu={args.gpu!r} min_vcpus={args.min_vcpus} max_price={args.max_price}")
            return 0
        print_offer_table(provider_for(args.provider).search_offers(args.gpu, args.min_vcpus, args.max_price))
        return 0

    if args.command == "up":
        ssh_target = args.ssh_target
        if args.dry_run:
            print(
                f"+ create {args.provider} instance offer={args.offer_id} "
                f"image={base_image_for(args)} mode={mode_for(args)}"
            )
        else:
            instance = provider_for(args.provider).create_instance(args.offer_id, base_image_for(args))
            args.instance_id = instance.id
            print(f"created instance {instance.id}; waiting for SSH")
            for _ in range(60):
                status = provider_for(args.provider).instance_status(instance.id)
                if status.ssh_host:
                    ssh_target = status.ssh_target()
                    break
                time.sleep(10)
        if not ssh_target:
            ssh_target = resolve_ssh_target(args)
        if mode_for(args) == "native":
            bootstrap_native(args, runner, ssh_target)
        else:
            archive_and_build_docker(args, runner, ssh_target)
        return 0

    ssh_target = resolve_ssh_target(args)
    if args.command == "launch":
        if mode_for(args) == "native":
            launch_native(args, runner, ssh_target)
        else:
            launch_container(args, runner, ssh_target)
    elif args.command == "smoke":
        if mode_for(args) == "native":
            smoke_native(args, runner, ssh_target)
        else:
            smoke_docker(args, runner, ssh_target)
    elif args.command == "status":
        status_run(args, runner, ssh_target)
    elif args.command == "logs":
        logs_run(args, runner, ssh_target)
    elif args.command == "sync":
        sync_run(args, runner, ssh_target)
        local_metrics = Path(args.local_sync_root) / args.run / "metrics.csv"
        for row in latest_metrics_rows(local_metrics):
            print(row)
    elif args.command == "down":
        destroy_after_sync(args, runner, ssh_target)
    else:
        raise SystemExit(f"unknown command: {args.command}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
