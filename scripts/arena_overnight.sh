#!/bin/sh
# Keep the Mac awake while the external orchestrator supervises this process.
exec caffeinate -is env PYTHONPATH=build ./.venv/bin/python -m src.v2.arena.main --config configs/arena.json "$@"
