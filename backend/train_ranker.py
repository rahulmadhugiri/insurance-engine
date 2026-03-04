#!/usr/bin/env python3
"""
Train second-stage claim-risk ranker and report policy uplift.

This script learns predicted claim risk from `ranking_labels.jsonl` and compares:
1) Baseline policy: choose highest recall score candidate per (project_id, trade_id)
2) Ranker policy: choose lowest predicted claim-risk candidate per (project_id, trade_id)
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

try:
    from backend.risk_ranker import (
        RANKER_CATEGORICAL_FEATURES,
        RANKER_NUMERIC_FEATURES,
        RiskRanker,
        build_ranker_features,
        clamp_id,
        sigmoid_scalar,
    )
except ModuleNotFoundError:
    from risk_ranker import (  # type: ignore
        RANKER_CATEGORICAL_FEATURES,
        RANKER_NUMERIC_FEATURES,
        RiskRanker,
        build_ranker_features,
        clamp_id,
        sigmoid_scalar,
    )

TRADE_CARDINALITY = 10
SUBCONTRACTOR_CARDINALITY = 10_000
CERTIFICATION_CARDINALITY = 8
ZIP_CODE_CARDINALITY = 500


@dataclass
class RankerSample:
    project_id: int
    trade_id: int
    cat: list[int]
    num: list[float]
    target_risk: float
    baseline_recall_prob: float
    ranking_target: int


class RankerDataset(Dataset):
    def __init__(self, rows: list[RankerSample]):
        self.project_id = torch.tensor([r.project_id for r in rows], dtype=torch.long)
        self.trade_id = torch.tensor([r.trade_id for r in rows], dtype=torch.long)
        self.cat = torch.tensor([r.cat for r in rows], dtype=torch.long)
        self.num = torch.tensor([r.num for r in rows], dtype=torch.float32)
        self.target_risk = torch.tensor([r.target_risk for r in rows], dtype=torch.float32)
        self.baseline_recall_prob = torch.tensor([r.baseline_recall_prob for r in rows], dtype=torch.float32)
        self.ranking_target = torch.tensor([r.ranking_target for r in rows], dtype=torch.float32)

    def __len__(self) -> int:
        return int(self.target_risk.shape[0])

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, ...]:
        return (
            self.cat[idx],
            self.num[idx],
            self.target_risk[idx],
            self.baseline_recall_prob[idx],
            self.project_id[idx],
            self.trade_id[idx],
            self.ranking_target[idx],
        )


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


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


def reservoir_sample_append(
    buffer: list[RankerSample], sample: RankerSample, seen: int, max_rows: int, rnd: random.Random
) -> tuple[list[RankerSample], int]:
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


def load_subcontractor_index(data_dir: Path) -> dict[int, dict[str, Any]]:
    jsonl_path = data_dir / "subcontractors.jsonl"
    json_path = data_dir / "subcontractors.json"
    rows: list[dict[str, Any]] = []

    if jsonl_path.exists():
        with jsonl_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if isinstance(item, dict):
                    rows.append(item)
    elif json_path.exists():
        with json_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, list):
            raise ValueError(f"Invalid format in {json_path}; expected list")
        rows = [x for x in payload if isinstance(x, dict)]
    else:
        raise FileNotFoundError(
            f"Could not find subcontractor catalog in {data_dir} "
            f"(expected {jsonl_path.name} or {json_path.name})"
        )

    index: dict[int, dict[str, Any]] = {}
    for raw in rows:
        sub_id = clamp_id(raw.get("sub_id", raw.get("subcontractor_id", 1)), SUBCONTRACTOR_CARDINALITY)
        normalized = {
            "sub_id": sub_id,
            "primary_trade_id": clamp_id(raw.get("primary_trade_id", 1), TRADE_CARDINALITY),
            "certification_id": clamp_id(raw.get("certification_id", 1), CERTIFICATION_CARDINALITY),
            "primary_zip_id": clamp_id(raw.get("primary_zip_id", 1), ZIP_CODE_CARDINALITY),
            "headcount": raw.get("headcount", raw.get("capacity")),
            "years_in_business": raw.get("years_in_business"),
            "capacity": raw.get("capacity"),
            "reliability": raw.get("reliability"),
            "price_level": raw.get("price_level"),
            "cert_strength": raw.get("cert_strength"),
            "skill_by_trade": raw.get("skill_by_trade"),
            "geo_preference": raw.get("geo_preference"),
            "core_trade_ids": raw.get("core_trade_ids"),
            "is_cold_start": raw.get("is_cold_start", False),
        }
        index[sub_id] = normalized
    return index


def parse_ranker_sample(row: dict[str, Any], sub_index: dict[int, dict[str, Any]]) -> RankerSample | None:
    trade_ids = row.get("trade_ids", [])
    if not isinstance(trade_ids, list) or not trade_ids:
        return None

    sub_id = clamp_id(row.get("sub_id", 1), SUBCONTRACTOR_CARDINALITY)
    sub = sub_index.get(sub_id)
    if sub is None:
        return None

    trade_id = clamp_id(sub["primary_trade_id"], TRADE_CARDINALITY)
    if trade_id not in trade_ids:
        # Keep ranker aligned to trade-slot ranking.
        return None

    recall_latent = row.get("recall_score_latent")
    recall_prob = (
        sigmoid_scalar(float(recall_latent))
        if recall_latent is not None
        else clamp01(row.get("claim_probability", 0.5))
    )
    cat, num = build_ranker_features(
        project_zip_id=clamp_id(row.get("zip_code_id", 1), ZIP_CODE_CARDINALITY),
        project_type_id=clamp_id(row.get("project_type_id", 1), 8),
        trade_needed_id=trade_id,
        subcontractor=sub,
        recall_prob=recall_prob,
    )

    target_risk = clamp01(row.get("observed_claim_risk", 0.5))
    ranking_target = int(row.get("ranking_target", 0))
    return RankerSample(
        project_id=int(row.get("project_id", 0)),
        trade_id=trade_id,
        cat=cat,
        num=num,
        target_risk=target_risk,
        baseline_recall_prob=recall_prob,
        ranking_target=ranking_target,
    )


def load_split_samples(
    ranking_labels_path: Path,
    sub_index: dict[int, dict[str, Any]],
    max_train_rows: int,
    max_val_rows: int,
    max_test_rows: int,
    seed: int,
) -> dict[str, list[RankerSample]]:
    rnd = random.Random(seed)
    sampled: dict[str, list[RankerSample]] = {"train": [], "val": [], "test": []}
    seen = {"train": 0, "val": 0, "test": 0}
    max_by_split = {"train": max_train_rows, "val": max_val_rows, "test": max_test_rows}

    with ranking_labels_path.open("r", encoding="utf-8") as f:
        for idx, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            split = row.get("split")
            if split not in sampled:
                continue
            sample = parse_ranker_sample(row, sub_index)
            if sample is None:
                continue
            sampled[split], seen[split] = reservoir_sample_append(
                sampled[split], sample, seen[split], max_by_split[split], rnd
            )
            if idx % 1_000_000 == 0:
                print(
                    f"Scanned {idx:,} rows | train={len(sampled['train']):,} "
                    f"val={len(sampled['val']):,} test={len(sampled['test']):,}"
                )
    return sampled


def policy_metrics_from_vectors(
    project_ids: torch.Tensor,
    trade_ids: torch.Tensor,
    target_risks: torch.Tensor,
    baseline_recall_probs: torch.Tensor,
    predicted_risks: torch.Tensor,
    ranking_targets: torch.Tensor,
    low_risk_threshold: float = 0.18,
) -> dict[str, float]:
    groups: dict[tuple[int, int], dict[str, float]] = {}

    pids = project_ids.tolist()
    tids = trade_ids.tolist()
    targets = target_risks.tolist()
    recalls = baseline_recall_probs.tolist()
    preds = predicted_risks.tolist()
    rtargets = ranking_targets.tolist()

    for pid, tid, tgt, rec, pred, rtar in zip(pids, tids, targets, recalls, preds, rtargets):
        key = (int(pid), int(tid))
        slot = groups.get(key)
        if slot is None:
            groups[key] = {
                "best_recall": rec,
                "baseline_risk": tgt,
                "baseline_rank_target": rtar,
                "best_pred": pred,
                "ranker_risk": tgt,
                "ranker_rank_target": rtar,
            }
            continue
        if rec > slot["best_recall"]:
            slot["best_recall"] = rec
            slot["baseline_risk"] = tgt
            slot["baseline_rank_target"] = rtar
        if pred < slot["best_pred"]:
            slot["best_pred"] = pred
            slot["ranker_risk"] = tgt
            slot["ranker_rank_target"] = rtar

    if not groups:
        return {
            "groups": 0.0,
            "baseline_mean_risk": 0.0,
            "ranker_mean_risk": 0.0,
            "risk_reduction_pct": 0.0,
            "baseline_low_risk_rate": 0.0,
            "ranker_low_risk_rate": 0.0,
            "baseline_ranking_target_rate": 0.0,
            "ranker_ranking_target_rate": 0.0,
        }

    baseline_risks = [v["baseline_risk"] for v in groups.values()]
    ranker_risks = [v["ranker_risk"] for v in groups.values()]
    baseline_target_rate = [v["baseline_rank_target"] for v in groups.values()]
    ranker_target_rate = [v["ranker_rank_target"] for v in groups.values()]

    baseline_mean = sum(baseline_risks) / len(baseline_risks)
    ranker_mean = sum(ranker_risks) / len(ranker_risks)
    reduction_pct = 0.0
    if baseline_mean > 1e-9:
        reduction_pct = ((baseline_mean - ranker_mean) / baseline_mean) * 100.0

    return {
        "groups": float(len(groups)),
        "baseline_mean_risk": float(baseline_mean),
        "ranker_mean_risk": float(ranker_mean),
        "risk_reduction_pct": float(reduction_pct),
        "baseline_low_risk_rate": float(sum(1 for x in baseline_risks if x <= low_risk_threshold) / len(baseline_risks)),
        "ranker_low_risk_rate": float(sum(1 for x in ranker_risks if x <= low_risk_threshold) / len(ranker_risks)),
        "baseline_ranking_target_rate": float(sum(baseline_target_rate) / len(baseline_target_rate)),
        "ranker_ranking_target_rate": float(sum(ranker_target_rate) / len(ranker_target_rate)),
    }


def evaluate(
    model: RiskRanker,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, Any]:
    model.eval()
    total_loss = 0.0
    total_examples = 0
    all_preds: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    all_recall_probs: list[torch.Tensor] = []
    all_project_ids: list[torch.Tensor] = []
    all_trade_ids: list[torch.Tensor] = []
    all_ranking_targets: list[torch.Tensor] = []

    with torch.no_grad():
        for batch in loader:
            cat, num, target, recall_prob, project_id, trade_id, ranking_target = batch
            cat = cat.to(device)
            num = num.to(device)
            target = target.to(device)

            pred = model(cat, num)
            loss = criterion(pred, target)
            bs = target.shape[0]
            total_loss += float(loss.item()) * bs
            total_examples += bs

            all_preds.append(pred.cpu())
            all_targets.append(target.cpu())
            all_recall_probs.append(recall_prob.cpu())
            all_project_ids.append(project_id.cpu())
            all_trade_ids.append(trade_id.cpu())
            all_ranking_targets.append(ranking_target.cpu())

    if total_examples == 0:
        return {
            "loss": 0.0,
            "mae": 0.0,
            "rmse": 0.0,
            "policy": policy_metrics_from_vectors(
                torch.tensor([]),
                torch.tensor([]),
                torch.tensor([]),
                torch.tensor([]),
                torch.tensor([]),
                torch.tensor([]),
            ),
        }

    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    recall_probs = torch.cat(all_recall_probs)
    project_ids = torch.cat(all_project_ids)
    trade_ids = torch.cat(all_trade_ids)
    ranking_targets = torch.cat(all_ranking_targets)

    mse = torch.mean((preds - targets) ** 2)
    mae = torch.mean(torch.abs(preds - targets))
    rmse = torch.sqrt(mse)
    policy = policy_metrics_from_vectors(
        project_ids=project_ids,
        trade_ids=trade_ids,
        target_risks=targets,
        baseline_recall_probs=recall_probs,
        predicted_risks=preds,
        ranking_targets=ranking_targets,
    )
    return {
        "loss": total_loss / total_examples,
        "mae": float(mae.item()),
        "rmse": float(rmse.item()),
        "policy": policy,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train claim-risk ranker from ranking_labels.jsonl.")
    parser.add_argument("--data-dir", required=True, help="Directory containing ranking_labels and subcontractor catalog")
    parser.add_argument("--ranking-file", default="ranking_labels.jsonl", help="Ranking labels filename under --data-dir")
    parser.add_argument(
        "--checkpoint-path",
        default="backend/checkpoints/risk_ranker.pt",
        help="Path to save best ranker checkpoint",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--max-train-rows", type=int, default=700_000)
    parser.add_argument("--max-val-rows", type=int, default=140_000)
    parser.add_argument("--max-test-rows", type=int, default=140_000)
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda|mps")
    parser.add_argument("--num-workers", type=int, default=0)
    args = parser.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir).expanduser()
    ranking_path = data_dir / args.ranking_file
    if not ranking_path.exists():
        raise FileNotFoundError(f"Missing ranking labels file: {ranking_path}")

    print(f"Loading subcontractor catalog from {data_dir} ...")
    sub_index = load_subcontractor_index(data_dir)
    print(f"Loaded subcontractors: {len(sub_index):,}")

    print(f"Loading ranker samples from {ranking_path} ...")
    sampled = load_split_samples(
        ranking_labels_path=ranking_path,
        sub_index=sub_index,
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

    train_ds = RankerDataset(sampled["train"])
    val_ds = RankerDataset(sampled["val"])
    test_ds = RankerDataset(sampled["test"]) if sampled["test"] else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    test_loader = (
        DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
        if test_ds
        else None
    )

    device = select_device(args.device)
    print(f"Training device: {device}")
    model = RiskRanker().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val_rmse = float("inf")
    best_state: dict[str, Any] | None = None
    history: list[dict[str, Any]] = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        seen = 0
        for batch in train_loader:
            cat, num, target, _recall_prob, _pid, _tid, _rt = batch
            cat = cat.to(device)
            num = num.to(device)
            target = target.to(device)

            optimizer.zero_grad(set_to_none=True)
            pred = model(cat, num)
            loss = criterion(pred, target)
            loss.backward()
            optimizer.step()

            bs = target.shape[0]
            running_loss += float(loss.item()) * bs
            seen += bs

        train_loss = running_loss / max(1, seen)
        val_metrics = evaluate(model, val_loader, criterion, device)
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_mae": val_metrics["mae"],
            "val_rmse": val_metrics["rmse"],
            "val_policy_risk_reduction_pct": val_metrics["policy"]["risk_reduction_pct"],
        }
        history.append(row)
        print(json.dumps(row))

        if val_metrics["rmse"] < best_val_rmse:
            best_val_rmse = val_metrics["rmse"]
            best_state = {
                "epoch": epoch,
                "val_metrics": val_metrics,
                "ranker_state_dict": model.state_dict(),
            }

    if best_state is None:
        raise RuntimeError("No valid checkpoint was produced.")

    test_metrics = {}
    if test_loader is not None:
        test_metrics = evaluate(model, test_loader, criterion, device)
        print(f"Test metrics: {json.dumps(test_metrics)}")

    checkpoint_path = Path(args.checkpoint_path).expanduser()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "feature_schema": {
            "categorical": RANKER_CATEGORICAL_FEATURES,
            "numeric": RANKER_NUMERIC_FEATURES,
        },
        "data": {
            "ranking_file": ranking_path.name,
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
            "history": history,
            "best_epoch": best_state["epoch"],
            "best_val_metrics": best_state["val_metrics"],
            "test_metrics": test_metrics,
        },
        "ranker_state_dict": best_state["ranker_state_dict"],
    }
    torch.save(payload, checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")


if __name__ == "__main__":
    main()
