#!/usr/bin/env python3
"""
Train the NCF-style two-tower recall model from synthetic recall pairs.

Usage:
  python3 backend/train_two_tower.py \
    --data-dir /private/tmp/insurance_synth_full \
    --checkpoint-path backend/checkpoints/two_tower_recall.pt
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# Keep these aligned with backend/main.py architecture.
ZIP_CODE_CARDINALITY = 500
PROJECT_TYPE_CARDINALITY = 8
TRADE_CARDINALITY = 10
SUBCONTRACTOR_CARDINALITY = 10_000
CERTIFICATION_CARDINALITY = 8

ZIP_EMBED_DIM = 24
PROJECT_TYPE_EMBED_DIM = 12
TRADE_EMBED_DIM = 16
SUB_ID_EMBED_DIM = 32
PRIMARY_TRADE_EMBED_DIM = 16
CERTIFICATION_EMBED_DIM = 8


def clamp_id(value: int, max_id: int) -> int:
    return max(1, min(max_id, int(value)))


class ProjectTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.zip_embedding = nn.Embedding(ZIP_CODE_CARDINALITY + 1, ZIP_EMBED_DIM)
        self.project_type_embedding = nn.Embedding(PROJECT_TYPE_CARDINALITY + 1, PROJECT_TYPE_EMBED_DIM)
        self.trade_needed_embedding = nn.Embedding(TRADE_CARDINALITY + 1, TRADE_EMBED_DIM)
        concat_dim = ZIP_EMBED_DIM + PROJECT_TYPE_EMBED_DIM + TRADE_EMBED_DIM
        self.net = nn.Sequential(
            nn.Linear(concat_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

    def forward(self, zip_code_id: torch.Tensor, project_type_id: torch.Tensor, trade_needed_id: torch.Tensor) -> torch.Tensor:
        x = torch.cat(
            [
                self.zip_embedding(zip_code_id),
                self.project_type_embedding(project_type_id),
                self.trade_needed_embedding(trade_needed_id),
            ],
            dim=1,
        )
        return self.net(x)


class SubcontractorTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.subcontractor_embedding = nn.Embedding(SUBCONTRACTOR_CARDINALITY + 1, SUB_ID_EMBED_DIM)
        self.primary_trade_embedding = nn.Embedding(TRADE_CARDINALITY + 1, PRIMARY_TRADE_EMBED_DIM)
        self.certification_embedding = nn.Embedding(CERTIFICATION_CARDINALITY + 1, CERTIFICATION_EMBED_DIM)
        concat_dim = SUB_ID_EMBED_DIM + PRIMARY_TRADE_EMBED_DIM + CERTIFICATION_EMBED_DIM
        self.net = nn.Sequential(
            nn.Linear(concat_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
        )

    def forward(
        self, subcontractor_id: torch.Tensor, primary_trade_id: torch.Tensor, certification_id: torch.Tensor
    ) -> torch.Tensor:
        x = torch.cat(
            [
                self.subcontractor_embedding(subcontractor_id),
                self.primary_trade_embedding(primary_trade_id),
                self.certification_embedding(certification_id),
            ],
            dim=1,
        )
        return self.net(x)


class TwoTowerRecallModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.project_tower = ProjectTower()
        self.sub_tower = SubcontractorTower()

    def forward(
        self,
        zip_code_id: torch.Tensor,
        project_type_id: torch.Tensor,
        trade_needed_id: torch.Tensor,
        sub_id: torch.Tensor,
        primary_trade_id: torch.Tensor,
        certification_id: torch.Tensor,
    ) -> torch.Tensor:
        project_emb = self.project_tower(zip_code_id, project_type_id, trade_needed_id)
        sub_emb = self.sub_tower(sub_id, primary_trade_id, certification_id)
        return (project_emb * sub_emb).sum(dim=1)


@dataclass
class Sample:
    zip_code_id: int
    project_type_id: int
    trade_needed_id: int
    sub_id: int
    primary_trade_id: int
    certification_id: int
    label: float


class RecallDataset(Dataset):
    def __init__(self, rows: list[Sample]):
        self.zip_code_id = torch.tensor([r.zip_code_id for r in rows], dtype=torch.long)
        self.project_type_id = torch.tensor([r.project_type_id for r in rows], dtype=torch.long)
        self.trade_needed_id = torch.tensor([r.trade_needed_id for r in rows], dtype=torch.long)
        self.sub_id = torch.tensor([r.sub_id for r in rows], dtype=torch.long)
        self.primary_trade_id = torch.tensor([r.primary_trade_id for r in rows], dtype=torch.long)
        self.certification_id = torch.tensor([r.certification_id for r in rows], dtype=torch.long)
        self.label = torch.tensor([r.label for r in rows], dtype=torch.float32)

    def __len__(self) -> int:
        return self.label.shape[0]

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        return (
            self.zip_code_id[idx],
            self.project_type_id[idx],
            self.trade_needed_id[idx],
            self.sub_id[idx],
            self.primary_trade_id[idx],
            self.certification_id[idx],
            self.label[idx],
        )


def reservoir_sample_append(
    buffer: list[Sample], sample: Sample, seen: int, max_rows: int, rnd: random.Random
) -> tuple[list[Sample], int]:
    seen += 1
    if max_rows <= 0:
        return buffer, seen
    if len(buffer) < max_rows:
        buffer.append(sample)
        return buffer, seen
    replace_idx = rnd.randint(0, seen - 1)
    if replace_idx < max_rows:
        buffer[replace_idx] = sample
    return buffer, seen


def parse_sample_from_row(row: dict[str, Any]) -> Sample | None:
    trade_ids = row.get("trade_ids", [])
    if not isinstance(trade_ids, list) or not trade_ids:
        return None

    primary_trade_id = clamp_id(int(row.get("primary_trade_id", 0)), TRADE_CARDINALITY)
    if primary_trade_id not in trade_ids:
        # Model scores per requested trade; only keep rows aligned to the sub's own trade.
        return None

    label_raw = row.get("label", row.get("recall_target", 0))
    label = float(int(label_raw))
    return Sample(
        zip_code_id=clamp_id(int(row.get("zip_code_id", 1)), ZIP_CODE_CARDINALITY),
        project_type_id=clamp_id(int(row.get("project_type_id", 1)), PROJECT_TYPE_CARDINALITY),
        trade_needed_id=primary_trade_id,
        sub_id=clamp_id(int(row.get("sub_id", 1)), SUBCONTRACTOR_CARDINALITY),
        primary_trade_id=primary_trade_id,
        certification_id=clamp_id(int(row.get("certification_id", 1)), CERTIFICATION_CARDINALITY),
        label=label,
    )


def load_split_samples(
    input_path: Path,
    max_train_rows: int,
    max_val_rows: int,
    max_test_rows: int,
    seed: int,
) -> dict[str, list[Sample]]:
    rnd = random.Random(seed)
    sampled: dict[str, list[Sample]] = {"train": [], "val": [], "test": []}
    seen = {"train": 0, "val": 0, "test": 0}
    max_by_split = {"train": max_train_rows, "val": max_val_rows, "test": max_test_rows}

    with input_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            split = row.get("split")
            if split not in sampled:
                continue
            sample = parse_sample_from_row(row)
            if sample is None:
                continue
            sampled[split], seen[split] = reservoir_sample_append(
                sampled[split],
                sample,
                seen[split],
                max_by_split[split],
                rnd,
            )
            if idx % 1_000_000 == 0:
                print(
                    f"Scanned {idx:,} rows | train={len(sampled['train']):,} "
                    f"val={len(sampled['val']):,} test={len(sampled['test']):,}"
                )

    return sampled


def select_device(device_arg: str) -> torch.device:
    d = device_arg.strip().lower()
    if d == "cpu":
        return torch.device("cpu")
    if d == "cuda":
        return torch.device("cuda")
    if d == "mps":
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def compute_metrics(logits: torch.Tensor, labels: torch.Tensor) -> dict[str, float]:
    probs = torch.sigmoid(logits)
    preds = (probs >= 0.5).float()
    total = labels.numel()
    correct = float((preds == labels).sum().item())
    positives = float((labels == 1).sum().item())
    predicted_positives = float((preds == 1).sum().item())
    true_positives = float(((preds == 1) & (labels == 1)).sum().item())
    precision = true_positives / max(1.0, predicted_positives)
    recall = true_positives / max(1.0, positives)
    return {
        "accuracy": correct / max(1, total),
        "precision": precision,
        "recall": recall,
        "positive_rate": positives / max(1, total),
    }


def evaluate(model: TwoTowerRecallModel, loader: DataLoader, criterion: nn.Module, device: torch.device) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_logits: list[torch.Tensor] = []
    all_labels: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            z, ptype, tneed, sid, ptrade, cert, label = [x.to(device) for x in batch]
            logits = model(z, ptype, tneed, sid, ptrade, cert)
            loss = criterion(logits, label)
            batch_size = label.shape[0]
            total_loss += float(loss.item()) * batch_size
            total_examples += batch_size
            all_logits.append(logits.cpu())
            all_labels.append(label.cpu())
    if total_examples == 0:
        return {"loss": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "positive_rate": 0.0}
    logits_cat = torch.cat(all_logits, dim=0)
    labels_cat = torch.cat(all_labels, dim=0)
    metrics = compute_metrics(logits_cat, labels_cat)
    metrics["loss"] = total_loss / total_examples
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train two-tower recall model from synthetic recall pairs.")
    parser.add_argument("--data-dir", required=True, help="Directory containing recall_training_pairs.jsonl")
    parser.add_argument(
        "--input-file",
        default="recall_training_pairs.jsonl",
        help="Input jsonl filename under --data-dir",
    )
    parser.add_argument(
        "--checkpoint-path",
        default="backend/checkpoints/two_tower_recall.pt",
        help="Path to save best checkpoint",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-train-rows", type=int, default=1_200_000)
    parser.add_argument("--max-val-rows", type=int, default=200_000)
    parser.add_argument("--max-test-rows", type=int, default=200_000)
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    input_path = Path(args.data_dir).expanduser() / args.input_file
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")

    print(f"Loading samples from {input_path} ...")
    sampled = load_split_samples(
        input_path=input_path,
        max_train_rows=args.max_train_rows,
        max_val_rows=args.max_val_rows,
        max_test_rows=args.max_test_rows,
        seed=args.seed,
    )
    print(
        f"Sampled rows | train={len(sampled['train']):,} "
        f"val={len(sampled['val']):,} test={len(sampled['test']):,}"
    )

    if not sampled["train"] or not sampled["val"]:
        raise RuntimeError("Need non-empty train and val samples.")

    train_ds = RecallDataset(sampled["train"])
    val_ds = RecallDataset(sampled["val"])
    test_ds = RecallDataset(sampled["test"]) if sampled["test"] else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = (
        DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers) if test_ds else None
    )

    device = select_device(args.device)
    print(f"Training device: {device}")

    model = TwoTowerRecallModel().to(device)

    n_pos = float((train_ds.label == 1).sum().item())
    n_neg = float((train_ds.label == 0).sum().item())
    pos_weight = max(1.0, n_neg / max(1.0, n_pos))
    criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_loss = float("inf")
    best_snapshot: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for batch in train_loader:
            z, ptype, tneed, sid, ptrade, cert, label = [x.to(device) for x in batch]
            optimizer.zero_grad(set_to_none=True)
            logits = model(z, ptype, tneed, sid, ptrade, cert)
            loss = criterion(logits, label)
            loss.backward()
            optimizer.step()
            batch_size = label.shape[0]
            running_loss += float(loss.item()) * batch_size
            seen += batch_size

        train_loss = running_loss / max(1, seen)
        val_metrics = evaluate(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
        }
        history.append(row)
        print(json.dumps(row))

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            best_snapshot = {
                "epoch": epoch,
                "val_metrics": val_metrics,
                "project_tower_state_dict": model.project_tower.state_dict(),
                "subcontractor_tower_state_dict": model.sub_tower.state_dict(),
            }

    if best_snapshot is None:
        raise RuntimeError("Training completed without a valid best checkpoint.")

    test_metrics = {}
    if test_loader is not None:
        test_metrics = evaluate(model, test_loader, criterion, device)
        print(f"Test metrics: {json.dumps(test_metrics)}")

    checkpoint_path = Path(args.checkpoint_path).expanduser()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "architecture": {
            "zip_embed_dim": ZIP_EMBED_DIM,
            "project_type_embed_dim": PROJECT_TYPE_EMBED_DIM,
            "trade_embed_dim": TRADE_EMBED_DIM,
            "sub_id_embed_dim": SUB_ID_EMBED_DIM,
            "primary_trade_embed_dim": PRIMARY_TRADE_EMBED_DIM,
            "certification_embed_dim": CERTIFICATION_EMBED_DIM,
            "mlp": [512, 128, 64],
        },
        "data": {
            "input_file": input_path.name,
            "max_train_rows": args.max_train_rows,
            "max_val_rows": args.max_val_rows,
            "max_test_rows": args.max_test_rows,
            "sampled_train_rows": len(sampled["train"]),
            "sampled_val_rows": len(sampled["val"]),
            "sampled_test_rows": len(sampled["test"]),
        },
        "training": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "device": str(device),
            "class_pos_weight": pos_weight,
            "history": history,
            "best_epoch": best_snapshot["epoch"],
            "best_val_metrics": best_snapshot["val_metrics"],
            "test_metrics": test_metrics,
        },
        "project_tower_state_dict": best_snapshot["project_tower_state_dict"],
        "subcontractor_tower_state_dict": best_snapshot["subcontractor_tower_state_dict"],
    }
    torch.save(payload, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
