# Play Against the Trained Net

Step-by-step setup for a fresh machine to play Dominion against the current
best checkpoint (campaign 15, generation 45 — card-token transformer, the
first checkpoint to beat a human) in the browser.

> **You need the weights file separately.** Trained checkpoints are not in
> git. Get `gen_0045.pt` (~18 MB) from Jack and place it as shown in step 4.

> **Docker alternative:** to run on a server/VPS instead of a dev setup, use
> the `deploy/` stack (one image, client + server, optional TLS) — see
> `deploy/README.md`. The steps below are the local dev-mode path.

## 1. Prerequisites

- git, CMake >= 3.24, a C++20 compiler (Xcode CLT / gcc 12+)
- Python 3.11+, Node 18+

## 2. Clone and checkout

```sh
git clone <repo-url> Mastermind && cd Mastermind
git checkout v2-phase1
```

## 3. Python env + engine bindings

```sh
python3 -m venv .venv
./.venv/bin/pip install torch numpy pybind11 pytest
./.venv/bin/pip install -r src/v2/web/server/requirements.txt
PYBIND11_DIR="$(./.venv/bin/python -m pybind11 --cmakedir)"
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON=ON -Dpybind11_DIR="$PYBIND11_DIR"
cmake --build build --target dominion_v2_py -j
```

Sanity check: `PYTHONPATH=build ./.venv/bin/python src/v2/py/test_smoke.py`

## 4. Install the weights

```sh
mkdir -p checkpoints/remote/campaign15
# copy the provided file so this path exists:
ls checkpoints/remote/campaign15/gen_0045.pt
```

## 5. Start the game server (repo root)

```sh
DOMINION_NN_CHECKPOINT=checkpoints/remote/campaign15/gen_0045.pt \
PYTHONPATH=build ./.venv/bin/python -m uvicorn src.v2.web.server.main:app --port 8000
```

## 6. Start the web client (second terminal)

```sh
cd src/v2/web/client && npm ci && npm run dev
```

Open the URL Vite prints (typically `http://localhost:5173`).

## 7. Play

- New game -> seats: **human** vs **bot:nnmcts** (full strength: policy +
  value + 400-sim search). `bot:nn` is the weaker policy-only variant.
- Kingdom: `random` is the honest test.
- Any other checkpoint works via the seat kind `bot:nnmcts:<path>`.
- Scripted opponents are also available: the "Human vs EngineBot" mode
  (seat `bot:engine3`) plays the strongest scripted bot.
- Leave `NN_MCTS_SIMS` at the default 400. Raising it does NOT strengthen
  the bot — a measured sims sweep shows play is flat from 200-1000 sims and
  degrades beyond that (see `docs/training-log.md`).

## Troubleshooting

- Bot seat errors on game creation -> the checkpoint path (steps 4/5) is
  wrong or unreadable.
- `ModuleNotFoundError: dominion_v2_py` -> run the server from the repo
  root with `PYTHONPATH=build`.
- Client shows no defs -> the server isn't running on port 8000, or the
  client dev proxy isn't pointing at it.
