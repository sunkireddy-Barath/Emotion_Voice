"""Download and prepare emotion training datasets: GoEmotions, MELD, DailyDialog, EmoWOZ.

Usage:
    python training/scripts/download_datasets.py --dataset goemotions --output_dir data/datasets
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

# GoEmotions → our canonical labels
GOEMOTIONS_MAP = {
    "admiration": "happy", "amusement": "happy", "approval": "happy",
    "caring": "calm", "desire": "excited", "excitement": "excited",
    "gratitude": "happy", "joy": "happy", "love": "happy",
    "optimism": "motivational", "pride": "happy", "relief": "calm",
    "anger": "angry", "annoyance": "angry", "disapproval": "serious",
    "disgust": "serious", "embarrassment": "fear", "fear": "fear",
    "grief": "sad", "nervousness": "fear", "remorse": "sad",
    "sadness": "sad", "confusion": "questioning", "curiosity": "questioning",
    "surprise": "surprise", "realization": "surprise",
    "disappointment": "sad", "neutral": "neutral",
}

MELD_MAP = {
    "neutral": "neutral", "surprise": "surprise", "fear": "fear",
    "sadness": "sad", "joy": "happy", "disgust": "serious", "anger": "angry",
}


def download_goemotions(output_dir: Path) -> None:
    try:
        from datasets import load_dataset
        logger.info("Downloading GoEmotions...")
        ds = load_dataset("go_emotions", "simplified")

        for split in ["train", "validation", "test"]:
            subset = ds[split]
            out_path = output_dir / f"goemotions_{split}.jsonl"
            with open(out_path, "w") as f:
                for item in subset:
                    emotions = item.get("labels", [])
                    if not emotions:
                        emotion = "neutral"
                    else:
                        label_names = subset.features["labels"].feature.names
                        raw_emotion = label_names[emotions[0]]
                        emotion = GOEMOTIONS_MAP.get(raw_emotion, "neutral")

                    f.write(json.dumps({"text": item["text"], "emotion": emotion}) + "\n")

            logger.info(f"  Saved {split}: {out_path}")
    except Exception as e:
        logger.error(f"GoEmotions download failed: {e}")


def download_meld(output_dir: Path) -> None:
    try:
        from datasets import load_dataset
        logger.info("Downloading MELD...")
        ds = load_dataset("declare-lab/MELD")

        for split, ds_split in [("train", "train"), ("val", "validation"), ("test", "test")]:
            out_path = output_dir / f"meld_{split}.jsonl"
            with open(out_path, "w") as f:
                for item in ds[ds_split]:
                    emotion = MELD_MAP.get(item.get("Emotion", "neutral").lower(), "neutral")
                    text = item.get("Utterance", "")
                    if text:
                        f.write(json.dumps({"text": text, "emotion": emotion}) + "\n")
            logger.info(f"  Saved {split}: {out_path}")
    except Exception as e:
        logger.error(f"MELD download failed: {e}")


def merge_datasets(output_dir: Path) -> None:
    """Merge all downloaded datasets into train/val splits."""
    all_train = []
    all_val = []

    for f in output_dir.glob("*_train.jsonl"):
        with open(f) as fh:
            all_train.extend(fh.readlines())

    for f in output_dir.glob("*_val*.jsonl"):
        with open(f) as fh:
            all_val.extend(fh.readlines())

    import random
    random.shuffle(all_train)
    random.shuffle(all_val)

    with open(output_dir / "combined_train.jsonl", "w") as f:
        f.writelines(all_train)
    with open(output_dir / "combined_val.jsonl", "w") as f:
        f.writelines(all_val)

    logger.info(f"Merged: {len(all_train)} train, {len(all_val)} val samples")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["goemotions", "meld", "all"], default="all")
    parser.add_argument("--output_dir", default="data/datasets")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("goemotions", "all"):
        download_goemotions(output_dir)
    if args.dataset in ("meld", "all"):
        download_meld(output_dir)

    merge_datasets(output_dir)
    logger.info("Dataset preparation complete.")


if __name__ == "__main__":
    main()
