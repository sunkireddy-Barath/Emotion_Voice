"""Train the neural prosody prediction model on processed dataset.

Usage:
    python training/scripts/train_prosody_model.py \
        --manifest data/processed/train_manifest.json \
        --val_manifest data/processed/val_manifest.json \
        --output_dir training/checkpoints/prosody \
        --epochs 100
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from core.prosody.prosody_model import ProsodyModel

logger = logging.getLogger(__name__)

EMOTION_LABELS = ["neutral", "happy", "sad", "angry", "excited", "fear", "surprise", "calm", "serious"]
LABEL2ID = {l: i for i, l in enumerate(EMOTION_LABELS)}


class ProsodyDataset(Dataset):
    def __init__(self, manifest_path: str, max_text_len: int = 256):
        with open(manifest_path) as f:
            self.samples = json.load(f)
        self.max_text_len = max_text_len

    def _text_to_ids(self, text: str) -> List[int]:
        ids = [ord(c) % 256 for c in text[: self.max_text_len]]
        ids += [0] * (self.max_text_len - len(ids))
        return ids

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx) -> Dict:
        sample = self.samples[idx]
        text_ids = self._text_to_ids(sample.get("text", ""))
        emotion_id = LABEL2ID.get(sample.get("emotion", "neutral"), 0)

        # Load prosody features if available
        f0 = np.zeros(100, dtype=np.float32)
        energy = np.zeros(100, dtype=np.float32)
        if sample.get("f0_path") and Path(sample["f0_path"]).exists():
            f0_raw = np.load(sample["f0_path"])
            f0_norm = f0_raw[:100] / 600.0 if len(f0_raw) >= 100 else np.pad(f0_raw / 600.0, (0, 100 - len(f0_raw)))
            f0 = f0_norm.astype(np.float32)
        if sample.get("energy_path") and Path(sample["energy_path"]).exists():
            e_raw = np.load(sample["energy_path"])
            e_norm = e_raw[:100] if len(e_raw) >= 100 else np.pad(e_raw, (0, 100 - len(e_raw)))
            energy = e_norm.astype(np.float32)

        return {
            "text_ids": torch.tensor(text_ids, dtype=torch.long),
            "emotion_id": torch.tensor(emotion_id, dtype=torch.long),
            "intensity": torch.tensor(0.8, dtype=torch.float),
            "f0_target": torch.tensor(f0, dtype=torch.float),
            "energy_target": torch.tensor(energy, dtype=torch.float),
        }


def train_epoch(model, loader, optimizer, device) -> float:
    model.train()
    total_loss = 0.0
    criterion = nn.MSELoss()

    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(batch["text_ids"], batch["emotion_id"], batch["intensity"])

        # Compute loss on pitch/energy curves
        pitch_pred = out["pitch_curve"][:, :100]
        energy_pred = out["energy_curve"][:, :100]

        loss = criterion(pitch_pred, batch["f0_target"]) + \
               criterion(energy_pred, batch["energy_target"])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()

    return total_loss / len(loader)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--val_manifest", required=True)
    parser.add_argument("--output_dir", default="training/checkpoints/prosody")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = ProsodyModel().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    train_ds = ProsodyDataset(args.manifest)
    val_ds = ProsodyDataset(args.val_manifest)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device)

        if epoch % 10 == 0:
            val_loss = train_epoch(model, val_loader, optimizer, device)
            logger.info(f"Epoch {epoch}: train_loss={train_loss:.6f}, val_loss={val_loss:.6f}")

            if val_loss < best_loss:
                best_loss = val_loss
                torch.save(
                    {"epoch": epoch, "model_state_dict": model.state_dict(),
                     "val_loss": val_loss},
                    Path(args.output_dir) / "best_prosody_model.pt",
                )
                logger.info(f"  → Saved best model (val_loss={val_loss:.6f})")
        else:
            logger.info(f"Epoch {epoch}: train_loss={train_loss:.6f}")

    logger.info(f"Training complete. Best val loss: {best_loss:.6f}")


if __name__ == "__main__":
    main()
