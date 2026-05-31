#!/bin/bash
# Start the React frontend dev server
set -e

cd "$(dirname "$0")/../frontend"

if [ ! -d "node_modules" ]; then
    echo "Installing dependencies..."
    npm install
fi

echo "=== Emotion Voice Frontend ==="
echo "Starting on http://localhost:3000"
echo ""

npm run dev
