"""Build and validate the training dataset from raw recordings + metadata."""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .audio_processor import AudioProcessor

logger = logging.getLogger(__name__)


class DatasetBuilder:
    def __init__(
        self,
        raw_dir: str | Path,
        processed_dir: str | Path,
        metadata_file: str | Path,
        processor: Optional[AudioProcessor] = None,
    ):
        self.raw_dir = Path(raw_dir)
        self.processed_dir = Path(processed_dir)
        self.metadata_file = Path(metadata_file)
        self.processor = processor or AudioProcessor()

    def load_metadata(self) -> pd.DataFrame:
        records = []
        with open(self.metadata_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split("|")
                if len(parts) >= 2:
                    filename = parts[0].strip()
                    text = parts[1].strip()
                    emotion = parts[2].strip() if len(parts) > 2 else "neutral"
                    records.append({
                        "filename": filename,
                        "text": text,
                        "emotion": emotion,
                        "path": str(self.raw_dir / filename),
                    })
        df = pd.DataFrame(records)
        logger.info(f"Loaded {len(df)} metadata entries")
        return df

    def validate_dataset(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        errors = []
        valid_rows = []

        for _, row in df.iterrows():
            path = Path(row["path"])
            if not path.exists():
                errors.append(f"File not found: {path}")
                continue
            if path.stat().st_size < 1024:
                errors.append(f"File too small: {path}")
                continue
            valid_rows.append(row)

        valid_df = pd.DataFrame(valid_rows).reset_index(drop=True)
        logger.info(f"Validation: {len(valid_df)} valid, {len(errors)} errors")
        return valid_df, errors

    def build(self, force_reprocess: bool = False) -> Dict:
        df = self.load_metadata()
        df, errors = self.validate_dataset(df)

        results = []
        for _, row in df.iterrows():
            stem = Path(row["filename"]).stem
            clean_wav = self.processed_dir / "cleaned_wav" / f"{stem}.wav"

            if clean_wav.exists() and not force_reprocess:
                logger.debug(f"Skipping (already processed): {stem}")
            else:
                result = self.processor.process(row["path"], self.processed_dir)
                results.append(result)

        # Build dataset manifest
        manifest = []
        for _, row in df.iterrows():
            stem = Path(row["filename"]).stem
            manifest.append({
                "id": stem,
                "text": row["text"],
                "emotion": row["emotion"],
                "wav_path": str(self.processed_dir / "cleaned_wav" / f"{stem}.wav"),
                "mel_path": str(self.processed_dir / "prosody_features" / f"{stem}_mel.npy"),
                "f0_path": str(self.processed_dir / "prosody_features" / f"{stem}_f0.npy"),
                "energy_path": str(self.processed_dir / "prosody_features" / f"{stem}_energy.npy"),
            })

        manifest_path = self.processed_dir / "manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)

        # Dataset statistics
        stats = self._compute_stats(df)
        stats_path = self.processed_dir / "stats.json"
        with open(stats_path, "w") as f:
            json.dump(stats, f, indent=2)

        logger.info(f"Dataset built: {len(manifest)} samples, manifest at {manifest_path}")
        return {"manifest": manifest, "stats": stats, "errors": errors}

    def _compute_stats(self, df: pd.DataFrame) -> Dict:
        emotion_counts = df["emotion"].value_counts().to_dict()
        total = len(df)
        return {
            "total_samples": total,
            "emotion_distribution": emotion_counts,
            "emotions": list(emotion_counts.keys()),
            "avg_text_length": float(df["text"].str.len().mean()) if total > 0 else 0,
        }

    def split(
        self,
        manifest_path: str | Path,
        train_ratio: float = 0.9,
        val_ratio: float = 0.05,
        seed: int = 42,
    ) -> Tuple[List, List, List]:
        with open(manifest_path) as f:
            manifest = json.load(f)

        np.random.seed(seed)
        indices = np.random.permutation(len(manifest))
        n_train = int(len(manifest) * train_ratio)
        n_val = int(len(manifest) * val_ratio)

        train = [manifest[i] for i in indices[:n_train]]
        val = [manifest[i] for i in indices[n_train:n_train + n_val]]
        test = [manifest[i] for i in indices[n_train + n_val:]]

        split_dir = Path(manifest_path).parent
        for name, split in [("train", train), ("val", val), ("test", test)]:
            with open(split_dir / f"{name}_manifest.json", "w") as f:
                json.dump(split, f, indent=2)

        logger.info(f"Split: train={len(train)}, val={len(val)}, test={len(test)}")
        return train, val, test
