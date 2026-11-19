#!/bin/bash
set -e

BUILD_DIR=/tmp/dominion-build

echo "=== Building DominionZero ==="
cmake -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release 2>&1 | grep -v "^--" || true
cmake --build "$BUILD_DIR" -j"$(nproc)" 2>/dev/null | grep -E "^\[|^BUILD|error:"

echo ""
echo "=== Running tests ==="
ctest --test-dir "$BUILD_DIR" -R v2 --output-on-failure

echo ""
echo "=== Ready. Attach with: mclaude ==="

# Keep container alive
exec sleep infinity
