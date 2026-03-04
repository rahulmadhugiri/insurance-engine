#!/usr/bin/env python3
"""Evaluate saved risk ranker checkpoint on held-out ranking labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

try:
    from backend.risk_ranker import RiskRanker
    from backend.train_ranker import (
        RankerDataset,
        evaluate,
        load_split_samples,
        load_subcontractor_index,
        select_device,
    )
except ModuleNotFoundError:
    from risk_ranker import RiskRanker  # type: ignore
    from train_ranker import RankerDataset, evaluate, load_split_samples, load_subcontractor_index, select_device  # type: ignore


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate risk-ranker checkpoint on held-out split.")
    parser.add_argument("--data-dir", required=True, help="Directory containing ranking_labels and subcontractor catalog")
    parser.add_argument("--checkpoint-path", required=True, help="Path to ranker checkpoint")
    parser.add_argument("--ranking-file", default="ranking_labels.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--max-test-rows", type=int, default=250_000)
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    args = parser.parse_args()

    data_dir = Path(args.data_dir).expanduser()
    ranking_path = data_dir / args.ranking_file
    checkpoint_path = Path(args.checkpoint_path).expanduser()
    if not ranking_path.exists():
        raise FileNotFoundError(f"Missing ranking labels file: {ranking_path}")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}")

    sub_index = load_subcontractor_index(data_dir)
    sampled = load_split_samples(
        ranking_labels_path=ranking_path,
        sub_index=sub_index,
        max_train_rows=0,
        max_val_rows=0,
        max_test_rows=args.max_test_rows,
        seed=args.seed,
    )
    if not sampled["test"]:
        raise RuntimeError("No test samples were loaded.")

    ds = RankerDataset(sampled["test"])
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    device = select_device(args.device)
    model = RiskRanker().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
    model.load_state_dict(checkpoint["ranker_state_dict"])

    metrics = evaluate(model, loader, nn.MSELoss(), device)
    result = {
        "checkpoint_path": str(checkpoint_path),
        "input_file": ranking_path.name,
        "sampled_test_rows": len(sampled["test"]),
        "metrics": metrics,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
