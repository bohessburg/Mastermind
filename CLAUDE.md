# DominionZero Project Guide

DominionZero is an AlphaZero-style Dominion engine and training harness. The
current runtime is v2 only: the engine, drivers, Python bindings, web server,
browser client, fuzz harness, and TUI all use the `src/v2/` implementation.

## Current Status

- v2 implements the full 26-card 2nd-edition base kingdom plus the supported
  basic cards.
- The state model is POD-oriented and memcpy-cloneable for MCTS, golden
  replays, fuzzing, and deterministic Python batch runners.
- The web server is authoritative. Clients send action ids; the server validates
  every action against the C++ legal mask before stepping.
- Golden replay files carry regression coverage for retired oracle comparisons.

## Reading Order

1. `README.md` - project goals and build/run quickstart.
2. `REFACTOR_PLAN.md` - authoritative architecture and rationale.
3. `IMPLEMENTATION_PLAN.md` - authoritative phased task list and acceptance
   criteria.
4. `docs/how-to-implement-cards.md` - v2 DSL cookbook for cards.
5. `docs/dominion_rules_reference.md` - rules reference for card behavior.
6. `docs/implemented-cards.md` - current runtime card roster.

## Ground Rules

- New engine work belongs under `src/v2/`, `tests/v2/`, docs, or the web client
  and server trees.
- Keep `GameState` and frame/state types trivially copyable.
- No heap allocation in core engine paths or driver game loops.
- Route gains, trashes, and discards through the move choke points.
- The trigger bus enqueues frames; it must not recurse into the interpreter.
- Prefer DSL card definitions. Add a new op only when the behavior has clear
  reuse; use `custom_step` for genuinely stateful cards.
- Build performance-sensitive checks in Release. Debug builds are much slower.
- Commit messages, when requested, are under 5 words with no body and no
  co-author lines.

## Common Commands

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
```

Python bindings:

```sh
PYBIND11_DIR="$(./.venv/bin/python -m pybind11 --cmakedir)"
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON=ON -Dpybind11_DIR="$PYBIND11_DIR"
cmake --build build --target dominion_v2_py
PYTHONPATH=build ./.venv/bin/python src/v2/py/test_smoke.py
```

Web server and client:

```sh
PYTHONPATH=build ./.venv/bin/python -m uvicorn src.v2.web.server.main:app --reload
cd src/v2/web/client && npm ci && npm run dev
```

TUI:

```sh
./build/dominion_v2_play --bot engine
```

