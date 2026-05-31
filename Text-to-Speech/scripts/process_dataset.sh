#!/bin/bash
# Process raw audio recordings into training-ready dataset
set -e

cd "$(dirname "$0")/.."
export PYTHONPATH="$(pwd)"

echo "=== Audio Processing Pipeline ==="
python3 - <<'EOF'
import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from data.processing.audio_processor import AudioProcessor
from data.processing.dataset_builder import DatasetBuilder

processor = AudioProcessor(sample_rate=22050)
builder = DatasetBuilder(
    raw_dir="data/raw",
    processed_dir="data/processed",
    metadata_file="data/metadata/recordings.txt",
    processor=processor,
)

result = builder.build()
print(f"\nDataset built:")
print(f"  Total samples: {result['stats']['total_samples']}")
print(f"  Emotions: {result['stats']['emotion_distribution']}")
if result['errors']:
    print(f"  Errors: {len(result['errors'])}")

# Split into train/val/test
builder.split("data/processed/manifest.json")
print("\n✓ Train/val/test splits created in data/processed/")
EOF
