#!/bin/bash
# Run all quality checks (formatting, linting, tests)
# Usage: ./scripts/quality.sh

set -e

cd "$(dirname "$0")/.."

echo "=== Running format check ==="
uv run black --check .
uv run ruff check .

echo ""
echo "=== Running tests ==="
uv run pytest backend/tests -v

echo ""
echo "=== All quality checks passed! ==="