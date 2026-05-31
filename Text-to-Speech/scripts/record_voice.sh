#!/bin/bash
# Interactive voice recording session
set -e

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"

EMOTION=${1:-neutral}
OUTPUT_DIR="data/raw"
METADATA_FILE="data/metadata/recordings.txt"

mkdir -p "$OUTPUT_DIR" "data/metadata"

echo "=== Voice Recording Session ==="
echo "Emotion: $EMOTION"
echo ""

python3 data/collection/recorder.py \
    --output_dir "$OUTPUT_DIR" \
    --metadata_file "$METADATA_FILE" \
    --emotion "$EMOTION" \
    --sample_rate 22050

echo ""
echo "✓ Recording session complete. Files saved to $OUTPUT_DIR"
