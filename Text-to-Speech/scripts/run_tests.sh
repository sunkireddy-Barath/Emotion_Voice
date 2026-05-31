#!/bin/bash
# Run test suite
set -e

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"

echo "=== Running Tests ==="
python3 -m pytest tests/unit/ -v --tb=short "$@"
echo ""
echo "=== Integration Tests (requires running API) ==="
# python3 -m pytest tests/integration/ -v --tb=short "$@"
echo "(Run 'pytest tests/integration/' with API running)"
