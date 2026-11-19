# c35 box scripts

These are copies of the scripts that ran c35 on vast box 85.165.238.33:40973. That box went down at 20:30 UTC on 2026-11-18.

| File | Purpose |
|---|---|
| `bootstrap.sh` | Sets up a new box: clones the repo from `/root/ship/repo.bundle`, copies data from `/root/ship/`, does a Release build with the bindings and runs a smoke test. |
| `c35c_driver.sh` | Supervisor program for the control continuation (`run_c35_control_long.json`). It retries up to 6 times. |
| `dominion-c35c.conf` | Its supervisor config. Goes in `/etc/supervisor/conf.d/`; the driver goes at `/root/c35c_driver.sh`. |
| `c35t_driver.sh`, `dominion-c35t.conf` | The same pair for the truncate continuation, which was stopped at gen 28. |
| `c35c_monitor.sh`, `c35c_monitor.py` | Mac-side watcher. It reports each finished phase and gate over ssh and keeps the Mac mirror in sync. Box host and port are hard-coded. |
| `deploy_gate_server.sh`, `deploy_gate_bf16.sh` | How the gate changes were deployed: stop the trainer, copy the files, check md5, run the tests, restart. |
| `h2h_merge.py` | Merges board-sharded head-to-head JSONs from `scripts/c29_search_ladder.py`. |
| `gate_tune.py` | Mac gate-worker tuning sweep. It was paused and never run. |

## Resuming c35 control on a new box

1. Ship the repo bundle, `exports/tuples_dgames_level1` and `checkpoints/remote/bc_level1_R40B/` to `/root/ship/`, then run `bootstrap.sh`.
2. Copy the Mac mirror `checkpoints/remote/campaign35_control/` to **`checkpoints/campaign35_control/`** on the box. That is the config's `checkpoint_dir`, not `remote/`. Bring all of it: `gen_0001-0031.pt`, `replay_state.npz` (1M rows, end of gen 31), `metrics.csv` and `league/`. Check the md5s.
3. Install the driver and conf, then run `supervisorctl reread && supervisorctl update`. `resume.sh` picks up `gen_0031.pt`, and `generations` 60 means 29 more generations.
4. Expect a 6-16 min human dataset load before gen 32's self-play starts.

How resume works:
- `generations` is the total generation count, not extra generations.
- The optimizer, the lr (constant at 7e-5) and the anchor schedule all come from the checkpoint and cannot be changed on resume. Changing them needs a code change in `src/v2/train/train.py`, around the block that copies `requested.*` into the config.
- Throughput and gate knobs (`gate_server`, `gate_bf16`, `gate_workers`, `server_*`) do follow the requested config.
