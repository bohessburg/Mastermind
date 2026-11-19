# Dominion v2 Remote Training

This tooling launches a disposable GPU worker for Phase T training. The local
machine remains the orchestrator; API keys stay in environment variables and are
never written to the repo.

## Keys

```bash
export VAST_API_KEY=...
export RUNPOD_API_KEY=...
```

Only the provider you use needs a key. Dry-run commands do not require keys.

## Modes

`remote_run.py` has two execution modes:

- `native`: default for Vast.ai. Vast instances are already containers created
  from the selected template, so the runner builds `dominion_v2_py` directly in
  the rented container and starts training with `nohup` plus a PID file.
- `docker`: default for RunPod-style VM providers. The runner builds
  `Dockerfile.train` on the worker and launches the training container with
  `--gpus all`.

You can override either default with `--mode native` or `--mode docker`.

## Vast.ai Native Happy Path

Use the Vast.ai PyTorch template. The default native image is a CUDA 12.8
PyTorch image suitable for current RTX 4090-class rentals.

Search offers:

```bash
python -m scripts.infra.remote_run --provider vastai offers \
  --gpu "RTX 4090" --min-vcpus 16 --max-price 1.00
```

Provision and bootstrap. This uploads a `git archive` of `HEAD`, extracts it to
root's `~/dominion` (`/root/dominion`), installs build tools if `cmake` is
missing, verifies CUDA torch, and builds `dominion_v2_py`.

```bash
python -m scripts.infra.remote_run --provider vastai --run run_remote up \
  --offer-id OFFER_ID
```

Smoke before launch. This must report `"device": "cuda"`.

```bash
python -m scripts.infra.remote_run --provider vastai --instance-id INSTANCE_ID \
  --run run_remote smoke
```

Launch training under `nohup`; output goes to
`/root/dominion/checkpoints/run_remote/console.log`, with a PID file next to it.

```bash
python -m scripts.infra.remote_run --provider vastai --instance-id INSTANCE_ID \
  --run run_remote launch --config src/v2/train/configs/run_remote.json
```

Resume the newest checkpoint:

```bash
python -m scripts.infra.remote_run --provider vastai --instance-id INSTANCE_ID \
  --run run_remote launch --config src/v2/train/configs/run_remote.json --resume-latest
```

Check health, tail logs, sync artifacts, then destroy safely:

```bash
python -m scripts.infra.remote_run --provider vastai --instance-id INSTANCE_ID --run run_remote status
python -m scripts.infra.remote_run --provider vastai --instance-id INSTANCE_ID --run run_remote logs
python -m scripts.infra.remote_run --provider vastai --instance-id INSTANCE_ID --run run_remote sync
python -m scripts.infra.remote_run --provider vastai --instance-id INSTANCE_ID --run run_remote down
```

`down` performs a final `rsync` first and refuses to destroy the provider
instance if that sync command fails.

## RunPod Docker Happy Path

RunPod workers are treated as VM-like Docker hosts.

```bash
python -m scripts.infra.remote_run --provider runpod --mode docker offers \
  --gpu "RTX 4090" --min-vcpus 16 --max-price 1.00
python -m scripts.infra.remote_run --provider runpod --mode docker --run run_remote up \
  --offer-id OFFER_ID
python -m scripts.infra.remote_run --provider runpod --mode docker --instance-id INSTANCE_ID \
  --run run_remote smoke
python -m scripts.infra.remote_run --provider runpod --mode docker --instance-id INSTANCE_ID \
  --run run_remote launch --config src/v2/train/configs/run_remote.json
python -m scripts.infra.remote_run --provider runpod --mode docker --instance-id INSTANCE_ID \
  --run run_remote status
python -m scripts.infra.remote_run --provider runpod --mode docker --instance-id INSTANCE_ID \
  --run run_remote sync
python -m scripts.infra.remote_run --provider runpod --mode docker --instance-id INSTANCE_ID \
  --run run_remote down
```

## Blackwell / cu128 Note

Native bootstrap prints `torch.__version__`, `torch.version.cuda`, CUDA
availability, and the GPU name. If a Blackwell GPU is detected with a torch
build older than cu128, bootstrap fails with a clear remedy:

```bash
python3 -m pip install --upgrade torch --index-url https://download.pytorch.org/whl/cu128
```

`--auto-fix-torch` is on by default and attempts that remedy automatically.
Use `--no-auto-fix-torch` to inspect the failure manually.

## Idle Guard

`launch` installs a watchdog loop on the worker. If the run's `metrics.csv`
exists and has not grown for 60 minutes, the watchdog stops the Docker
container in docker mode or kills the native training PID in native mode. Native
mode also writes `watchdog.marker`; `status` surfaces the marker and watchdog
log.

## Cost Notes

For a single RTX 4090 with 16+ vCPUs, expected spot/on-demand pricing varies by
provider and region. Use `offers` immediately before launching and prefer hosts
with good upload bandwidth so artifact syncs do not dominate teardown time.
