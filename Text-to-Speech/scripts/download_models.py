#!/usr/bin/env python3
"""Download all required models for the Emotion Voice system.

Usage:
    python scripts/download_models.py                  # Download VITS (default, ~200MB)
    python scripts/download_models.py --model xtts     # Download XTTS v2 (~2.5GB)
    python scripts/download_models.py --model emotion  # Download emotion model only
    python scripts/download_models.py --model all      # Download everything
    python scripts/download_models.py --list           # List available models
"""
from __future__ import annotations

import argparse
import os
import sys
import shutil
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

MODELS = {
    "vits": {
        "description": "VITS multi-speaker English TTS (VCTK) — 200MB, fast, high quality",
        "size": "~200 MB",
        "type": "tts",
        "id": "tts_models/en/vctk/vits",
    },
    "vits-single": {
        "description": "VITS single-speaker English TTS (LJSpeech) — 100MB, fallback",
        "size": "~100 MB",
        "type": "tts",
        "id": "tts_models/en/ljspeech/vits",
    },
    "xtts": {
        "description": "XTTS v2 multilingual + voice cloning — 2.5GB, best quality",
        "size": "~2.5 GB",
        "type": "tts",
        "id": "tts_models/multilingual/multi-dataset/xtts_v2",
    },
    "emotion": {
        "description": "Emotion classifier (DistilRoBERTa) — 300MB",
        "size": "~300 MB",
        "type": "hf",
        "id": "j-hartmann/emotion-english-distilroberta-base",
    },
    "emotion-small": {
        "description": "Emotion classifier (DistilBERT) — 250MB, smaller fallback",
        "size": "~250 MB",
        "type": "hf",
        "id": "bhadresh-savani/distilbert-base-uncased-emotion",
    },
}


def check_disk_space(required_gb: float) -> bool:
    stat = shutil.disk_usage("/")
    free_gb = stat.free / (1024 ** 3)
    print(f"  Disk: {free_gb:.1f} GB free, {required_gb:.1f} GB required")
    if free_gb < required_gb + 0.5:
        print(f"  ⚠ Warning: low disk space. Need at least {required_gb + 0.5:.1f} GB free.")
        return False
    return True


def download_tts_model(model_id: str, models_dir: str) -> bool:
    print(f"  Downloading TTS model: {model_id}")
    try:
        from TTS.api import TTS
        tts = TTS(model_name=model_id, progress_bar=True, gpu=False)
        print(f"  ✓ TTS model ready: {model_id}")
        return True
    except ImportError:
        print("  ✗ TTS package not installed. Run: pip install TTS")
        return False
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def download_hf_model(model_id: str, cache_dir: str) -> bool:
    print(f"  Downloading HuggingFace model: {model_id}")
    try:
        from transformers import pipeline
        os.environ.setdefault("TRANSFORMERS_CACHE", cache_dir)
        pipe = pipeline("text-classification", model=model_id, top_k=None, device=-1)
        # Warm up
        pipe("test")
        print(f"  ✓ Model ready: {model_id}")
        return True
    except ImportError:
        print("  ✗ transformers not installed. Run: pip install transformers")
        return False
    except Exception as e:
        print(f"  ✗ Failed: {e}")
        return False


def list_models() -> None:
    print("\nAvailable models:\n")
    print(f"{'Name':<16} {'Size':<12} {'Description'}")
    print("-" * 70)
    for name, info in MODELS.items():
        print(f"{name:<16} {info['size']:<12} {info['description']}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download models for Emotion Voice",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model", "-m",
        default="vits",
        choices=list(MODELS.keys()) + ["all", "tts-only", "emotion-only"],
        help="Which model(s) to download (default: vits)",
    )
    parser.add_argument("--list", "-l", action="store_true", help="List available models")
    parser.add_argument(
        "--models-dir",
        default="models",
        help="Directory to store models (default: models/)",
    )
    args = parser.parse_args()

    if args.list:
        list_models()
        return

    models_dir = Path(args.models_dir)
    tts_dir = str(models_dir / "tts_model")
    emotion_dir = str(models_dir / "emotion_classifier")
    Path(tts_dir).mkdir(parents=True, exist_ok=True)
    Path(emotion_dir).mkdir(parents=True, exist_ok=True)

    print("\n=== Emotion Voice — Model Download ===\n")

    to_download = []
    if args.model == "all":
        to_download = ["vits", "emotion"]  # Skip xtts by default (large)
        print("Note: Use --model xtts to download XTTS v2 (2.5GB, multilingual)")
    elif args.model == "tts-only":
        to_download = ["vits"]
    elif args.model == "emotion-only":
        to_download = ["emotion"]
    else:
        to_download = [args.model]

    results = {}
    for name in to_download:
        info = MODELS[name]
        print(f"[{name}] {info['description']}")
        check_disk_space(float(info["size"].replace("~", "").replace(" MB", "e-3")
                               .replace(" GB", "").replace("e-3", "")) *
                         (0.001 if "MB" in info["size"] else 1.0))

        if info["type"] == "tts":
            results[name] = download_tts_model(info["id"], tts_dir)
        elif info["type"] == "hf":
            results[name] = download_hf_model(info["id"], emotion_dir)
        print()

    print("=== Summary ===")
    all_ok = True
    for name, ok in results.items():
        status = "✓" if ok else "✗"
        print(f"  {status} {name}")
        if not ok:
            all_ok = False

    if all_ok:
        print("\n✓ All models downloaded. Run ./scripts/start_api.sh to start the server.\n")
    else:
        print("\n⚠ Some models failed. Check the output above.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
