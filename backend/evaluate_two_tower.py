#!/usr/bin/env python3
"""Evaluate a trained two-tower checkpoint on held-out recall pairs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from backend.train_two_tower import (
        RecallDataset,
        TwoTowerRecallModel,
        evaluate,
        load_split_samples,
        select_device,
    )
except ModuleNotFoundError:
    # Supports direct execution: `python backend/evaluate_two_tower.py ...`
    from train_two_tower import (  # type: ignore
        RecallDataset,
        TwoTowerRecallModel,
        evaluate,
        load_split_samples,
        select_device,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate two-tower recall checkpoint.")
    parser.add_argument("--data-dir", required=True, help="Directory containing recall_training_pairs.jsonl")
    parser.add_argument("--checkpoint-path", required=True, help="Checkpoint produced by train_two_tower.py")
    parser.add_argument("--input-file", default="recall_training_pairs.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-test-rows", type=int, default=300_000)
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    args = parser.parse_args()

    input_path = Path(args.data_dir).expanduser() / args.input_file
    checkpoint_path = Path(args.checkpoint_path).expanduser()
    if not input_path.exists():
        raise FileNotFoundError(f"Missing input file: {input_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    sampled = load_split_samples(
        input_path=input_path,
        max_train_rows=0,
        max_val_rows=0,
        max_test_rows=args.max_test_rows,
        seed=args.seed,
    )
    if not sampled["test"]:
        raise RuntimeError("No sampled test rows found.")

    test_ds = RecallDataset(sampled["test"])
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = select_device(args.device)
    model = TwoTowerRecallModel().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model.project_tower.load_state_dict(checkpoint["project_tower_state_dict"])
    model.sub_tower.load_state_dict(checkpoint["subcontractor_tower_state_dict"])

    criterion = nn.BCEWithLogitsLoss()
    metrics = evaluate(model, test_loader, criterion, device)
    result = {
        "checkpoint_path": str(checkpoint_path),
        "input_file": input_path.name,
        "sampled_test_rows": len(sampled["test"]),
        "metrics": metrics,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
