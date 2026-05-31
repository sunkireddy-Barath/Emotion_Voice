"""Fine-tune emotion classifier on custom dataset using GoEmotions / MELD / DailyDialog.

Usage:
    python training/scripts/train_emotion_classifier.py \
        --dataset goemotions \
        --output_dir training/checkpoints/emotion \
        --epochs 5 --batch_size 16
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

logger = logging.getLogger(__name__)

EMOTION_LABELS = ["neutral", "happy", "sad", "angry", "excited", "fear", "surprise", "calm", "serious"]
LABEL2ID = {l: i for i, l in enumerate(EMOTION_LABELS)}
ID2LABEL = {i: l for i, l in enumerate(EMOTION_LABELS)}


class EmotionDataset(Dataset):
    def __init__(self, data_path: str, tokenizer, max_length: int = 128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []

        with open(data_path) as f:
            for line in f:
                item = json.loads(line.strip())
                text = item.get("text", "")
                label = item.get("emotion", "neutral")
                if label in LABEL2ID:
                    self.samples.append({"text": text, "label": LABEL2ID[label]})

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        enc = self.tokenizer(
            item["text"],
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "labels": torch.tensor(item["label"], dtype=torch.long),
        }


def train(
    train_path: str,
    val_path: str,
    model_name: str = "distilbert-base-uncased",
    output_dir: str = "training/checkpoints/emotion",
    epochs: int = 5,
    batch_size: int = 16,
    lr: float = 2e-5,
    device: str = "cpu",
):
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=len(EMOTION_LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    ).to(device)

    train_dataset = EmotionDataset(train_path, tokenizer)
    val_dataset = EmotionDataset(val_path, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(train_loader) // 10,
        num_training_steps=len(train_loader) * epochs,
    )

    best_val_acc = 0.0
    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                preds = outputs.logits.argmax(dim=-1)
                correct += (preds == batch["labels"]).sum().item()
                total += len(batch["labels"])

        val_acc = correct / total
        logger.info(f"Epoch {epoch}/{epochs}: loss={avg_loss:.4f}, val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            logger.info(f"  → Saved best model (acc={val_acc:.4f})")

    logger.info(f"Training complete. Best val accuracy: {best_val_acc:.4f}")
    return best_val_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--model", default="distilbert-base-uncased")
    parser.add_argument("--output_dir", default="training/checkpoints/emotion")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    train(args.train, args.val, args.model, args.output_dir,
          args.epochs, args.batch_size, args.lr, args.device)


if __name__ == "__main__":
    main()
