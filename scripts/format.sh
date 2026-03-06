#!/bin/bash
# Format code using black and ruff
# Usage: ./scripts/format.sh [--check]

set -e

cd "$(dirname "$0")/.."

if [ "$1" == "--check" ]; then
    echo "Checking code formatting..."
    uv run black --check .
    uv run ruff check .
    echo "Format check complete!"
else
    echo "Formatting code..."
    uv run black .
    uv run ruff check --fix .
    echo "Formatting complete!"
fi