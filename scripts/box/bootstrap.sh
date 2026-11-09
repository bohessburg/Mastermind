#!/usr/bin/env bash
# One-shot box bootstrap: repo from bundle, data into place, Release build with
# bindings against /venv/main (torch cu130, Blackwell-ready), smoke checks.
set -euo pipefail
export PYTHONUNBUFFERED=1
cd /root
if [ ! -d Mastermind/.git ]; then
  git clone -q -b v2-phase1 /root/ship/repo.bundle Mastermind
fi
cd /root/Mastermind

for d in exports/tuples_all_40plus data/dominion_games/ratings checkpoints/campaign24 checkpoints/remote/campaign15; do
  mkdir -p "$d"
  cp -a /root/ship/"$d"/. "$d"/
done
ln -sfn /venv/main .venv
echo "== pip"
.venv/bin/python -m pip install -q pybind11 pytest numpy
echo "== cmake"
PYBIND11_DIR="$(.venv/bin/python -m pybind11 --cmakedir)"
cmake -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_PYTHON=ON -Dpybind11_DIR="$PYBIND11_DIR" \
  -DPython3_EXECUTABLE=/venv/main/bin/python >/root/cmake.log 2>&1 || { tail -30 /root/cmake.log; exit 1; }
echo "== build"
cmake --build build -j 64 >/root/build.log 2>&1 || { grep -m20 -E "error|Error" /root/build.log; exit 1; }
echo "== ctest"
ctest --test-dir build -j 32 2>&1 | tail -4
echo "== smoke"
PYTHONPATH=build .venv/bin/python src/v2/py/test_smoke.py 2>&1 | tail -3
PYTHONPATH=build:. .venv/bin/python - <<'EOF'
import torch, dominion_v2_py as dz
print("torch", torch.__version__, "cuda", torch.version.cuda, torch.cuda.get_device_name(0))
x = torch.randn(4096, 4096, device="cuda", dtype=torch.bfloat16); print("matmul ok", float((x @ x).float().abs().mean()))
print("STATE_HASH_GENERATION", dz.STATE_HASH_GENERATION)
EOF
echo BOOTSTRAP_DONE
