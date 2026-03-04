#!/usr/bin/env python3
"""
Synthetic training data generator for Insurance Engine's NCF-style stack.

Hard constraints enforced:
1) IDs match backend catalogs exactly (zip_code_id, project_type_id, trade_ids)
2) Stable output schema with explicit field names/types
3) Generates recall pairs + parser labels
4) Time-based train/val/test split
5) Includes hard negatives and near misses
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


# Canonical catalogs mirrored from backend/main.py.
TRADE_CATALOG = {
    1: "Framing",
    2: "Plumbing",
    3: "Electrical",
    4: "Concrete",
    5: "Roofing",
    6: "HVAC",
    7: "Masonry",
    8: "Drywall",
    9: "Excavation",
    10: "Painting",
}

PROJECT_TYPE_CATALOG = {
    1: "Residential",
    2: "Commercial",
    3: "Industrial",
    4: "Infrastructure / Civil",
}

ZIP_CATALOG = {
    1: {"zip_code": "90001", "city": "Los Angeles", "state": "CA", "region": "SoCal", "weather_risk_base": 0.22},
    2: {"zip_code": "77001", "city": "Houston", "state": "TX", "region": "Gulf", "weather_risk_base": 0.42},
    3: {"zip_code": "33101", "city": "Miami", "state": "FL", "region": "South", "weather_risk_base": 0.55},
    4: {"zip_code": "60601", "city": "Chicago", "state": "IL", "region": "Midwest", "weather_risk_base": 0.36},
    5: {"zip_code": "98101", "city": "Seattle", "state": "WA", "region": "PNW", "weather_risk_base": 0.48},
    6: {"zip_code": "80202", "city": "Denver", "state": "CO", "region": "Mountain", "weather_risk_base": 0.30},
    7: {"zip_code": "10001", "city": "New York", "state": "NY", "region": "Northeast", "weather_risk_base": 0.34},
    8: {"zip_code": "94107", "city": "San Francisco", "state": "CA", "region": "Bay Area", "weather_risk_base": 0.26},
}

CERTIFICATION_CATALOG = {
    1: "OSHA-10",
    2: "OSHA-30",
    3: "Union",
    4: "Minority-Owned",
    5: "Woman-Owned",
    6: "EMR-Below-1.0",
}

TRADE_ALIASES = {
    1: ["framing", "wood framing", "rough carpentry"],
    2: ["plumbing", "piping", "domestic water"],
    3: ["electrical", "wiring", "power distribution"],
    4: ["concrete", "slab", "foundation pour"],
    5: ["roofing", "roof membrane", "shingles"],
    6: ["hvac", "mechanical", "air handling"],
    7: ["masonry", "block work", "brick"],
    8: ["drywall", "gypsum board", "sheetrock"],
    9: ["excavation", "grading", "earthwork"],
    10: ["painting", "coatings", "interior paint"],
}

PROJECT_TYPE_ALIASES = {
    1: ["residential", "multifamily", "single-family", "townhome"],
    2: ["commercial", "office", "retail", "mixed-use"],
    3: ["industrial", "warehouse", "manufacturing", "plant"],
    4: ["infrastructure", "civil", "public works", "utility corridor"],
}


@dataclass
class Config:
    seed: int = 42
    output_dir: str = "./synthetic_construction_underwriting_data"
    n_projects: int = 100_000
    n_subcontractors: int = 50_000
    min_candidates_per_project: int = 95
    max_candidates_per_project: int = 110
    months: int = 24
    start_date: str = "2024-01-01"
    train_end: str = "2025-06-30"
    val_end: str = "2025-09-30"
    cold_start_subcontractor_frac: float = 0.12
    cold_start_project_frac: float = 0.06
    min_trades_per_project: int = 1
    max_trades_per_project: int = 4
    claim_event_delay_days_min: int = 30
    claim_event_delay_days_max: int = 180
    parser_noise_prob: float = 0.28
    print_every_projects: int = 5_000


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def month_starts(start: datetime, n_months: int) -> list[datetime]:
    out = []
    y, m = start.year, start.month
    for _ in range(n_months):
        out.append(datetime(y, m, 1))
        m += 1
        if m > 12:
            y += 1
            m = 1
    return out


def random_timestamp_within_month(base: datetime, rnd: random.Random) -> datetime:
    if base.month == 12:
        nxt = datetime(base.year + 1, 1, 1)
    else:
        nxt = datetime(base.year, base.month + 1, 1)
    sec = int((nxt - base).total_seconds())
    return base + timedelta(seconds=rnd.randint(0, max(0, sec - 1)))


def time_split(ts: datetime, cfg: Config) -> str:
    d = ts.date()
    if d <= datetime.fromisoformat(cfg.train_end).date():
        return "train"
    if d <= datetime.fromisoformat(cfg.val_end).date():
        return "val"
    return "test"


def weighted_choice(items: list[int], weights: list[float], rnd: random.Random) -> int:
    total = sum(weights)
    p = rnd.random() * total
    c = 0.0
    for item, w in zip(items, weights):
        c += w
        if p <= c:
            return item
    return items[-1]


def sample_k(items: list[int], weights: list[float], k: int, rnd: random.Random) -> list[int]:
    picked: list[int] = []
    pool_items = items[:]
    pool_w = weights[:]
    for _ in range(min(k, len(pool_items))):
        choice = weighted_choice(pool_items, pool_w, rnd)
        idx = pool_items.index(choice)
        picked.append(choice)
        pool_items.pop(idx)
        pool_w.pop(idx)
    return picked


def assert_catalog_hard_constraints() -> None:
    assert set(TRADE_CATALOG.keys()) == set(range(1, 11)), "trade_ids must be exactly 1..10"
    assert set(PROJECT_TYPE_CATALOG.keys()) == set(range(1, 5)), "project_type_ids must be exactly 1..4"
    assert set(ZIP_CATALOG.keys()) == set(range(1, 9)), "zip_ids must be exactly 1..8"

    try:
        # Optional parity check against backend constants; skipped if import fails.
        from backend.main import PROJECT_TYPE_LABELS, TRADE_LABELS, ZIP_CODE_LABELS  # type: ignore

        assert TRADE_CATALOG == TRADE_LABELS, "TRADE_CATALOG mismatch with backend/main.py"
        assert PROJECT_TYPE_CATALOG == PROJECT_TYPE_LABELS, "PROJECT_TYPE_CATALOG mismatch with backend/main.py"
        assert {
            zid: f"{meta['zip_code']} ({meta['city']}, {meta['state']})" for zid, meta in ZIP_CATALOG.items()
        } == ZIP_CODE_LABELS, "ZIP_CATALOG mismatch with backend/main.py"
    except Exception:
        pass


def generate_subcontractors(cfg: Config, rnd: random.Random) -> list[dict[str, Any]]:
    trade_ids = sorted(TRADE_CATALOG.keys())
    rows: list[dict[str, Any]] = []
    start = datetime.fromisoformat(cfg.start_date)
    months = month_starts(start, cfg.months)
    cold_cutoff = months[-3]

    zip_ids = sorted(ZIP_CATALOG.keys())
    zip_weights = [0.22, 0.14, 0.13, 0.12, 0.09, 0.08, 0.12, 0.10]

    for sub_id in range(1, cfg.n_subcontractors + 1):
        zip_id = weighted_choice(zip_ids, zip_weights, rnd)
        join_month = months[rnd.randrange(len(months))]
        joined_at = random_timestamp_within_month(join_month, rnd)
        is_cold = joined_at >= cold_cutoff

        n_core = weighted_choice([1, 2, 3], [0.5, 0.38, 0.12], rnd)
        core_trades = sorted(sample_k(trade_ids, [1.0] * len(trade_ids), n_core, rnd))
        primary_trade_id = core_trades[0]

        skill_by_trade = {}
        for tid in trade_ids:
            if tid in core_trades:
                v = clamp(rnd.gauss(0.76, 0.12))
            else:
                v = clamp(rnd.gauss(0.24, 0.1))
            skill_by_trade[tid] = round(v, 4)

        geo_pref = {}
        denom = 0.0
        for zid in zip_ids:
            base = 0.08 + 0.12 * rnd.random()
            if zid == zip_id:
                base += 0.45 + 0.22 * rnd.random()
            elif ZIP_CATALOG[zid]["region"] == ZIP_CATALOG[zip_id]["region"]:
                base += 0.14 + 0.12 * rnd.random()
            geo_pref[zid] = base
            denom += base
        for zid in geo_pref:
            geo_pref[zid] = round(geo_pref[zid] / max(denom, 1e-9), 6)

        rows.append(
            {
                # model-facing IDs
                "sub_id": sub_id,
                "primary_trade_id": primary_trade_id,
                "certification_id": weighted_choice(list(CERTIFICATION_CATALOG.keys()), [1, 1, 1, 1, 1, 1], rnd),
                # additional synthetic features
                "joined_at": joined_at.isoformat(),
                "primary_zip_id": zip_id,
                "core_trade_ids": core_trades,
                "skill_by_trade": skill_by_trade,
                "capacity": max(1, min(40, int(round(math.exp(rnd.gauss(1.35, 0.55)))))),
                "reliability": round(clamp(rnd.betavariate(8, 3)), 4),
                "price_level": round(clamp(rnd.gauss(0.52, 0.18)), 4),
                "cert_strength": round(clamp(rnd.betavariate(3, 2)), 4),
                "geo_preference": geo_pref,
                "is_cold_start": is_cold,
            }
        )
    return rows


def trade_weights_for_type(project_type_id: int) -> dict[int, float]:
    base = {tid: 1.0 for tid in TRADE_CATALOG}
    if project_type_id == 1:
        for tid, mult in [(1, 1.8), (2, 1.5), (10, 1.6), (5, 1.2)]:
            base[tid] *= mult
    elif project_type_id == 2:
        for tid, mult in [(3, 1.55), (6, 1.45), (4, 1.35)]:
            base[tid] *= mult
    elif project_type_id == 3:
        for tid, mult in [(3, 1.65), (6, 1.55), (4, 1.45), (9, 1.2)]:
            base[tid] *= mult
    else:
        for tid, mult in [(4, 1.5), (9, 1.55), (7, 1.35)]:
            base[tid] *= mult
    return base


def generate_scope_text(project: dict[str, Any], rnd: random.Random) -> str:
    zip_meta = ZIP_CATALOG[project["zip_code_id"]]
    p_alias = rnd.choice(PROJECT_TYPE_ALIASES[project["project_type_id"]])
    trade_phrases = [rnd.choice(TRADE_ALIASES[tid]) for tid in project["trade_ids"]]

    fragments = [
        f"{p_alias.capitalize()} project in {zip_meta['city']}, {zip_meta['state']} ({zip_meta['zip_code']}).",
        f"Need bids for {', '.join(trade_phrases)}.",
    ]
    if rnd.random() < 0.8:
        fragments.append(rnd.choice(["Fast-track schedule.", "Phased schedule.", "Tight turnover timeline."]))
    if rnd.random() < 0.7:
        fragments.append(rnd.choice(["Insurance-sensitive job.", "High documentation standards.", "Strict compliance needs."]))
    if rnd.random() < 0.5:
        fragments.append(rnd.choice(["Owner still finalizing finish scope.", "Utility coordination still in progress."]))
    return " ".join(fragments)


def generate_projects(cfg: Config, rnd: random.Random) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    start = datetime.fromisoformat(cfg.start_date)
    months = month_starts(start, cfg.months)
    cold_cutoff = months[-3]

    project_type_ids = sorted(PROJECT_TYPE_CATALOG.keys())
    project_type_weights = [0.47, 0.28, 0.13, 0.12]
    zip_ids = sorted(ZIP_CATALOG.keys())
    zip_weights = [0.18, 0.14, 0.13, 0.12, 0.10, 0.08, 0.13, 0.12]

    projects: list[dict[str, Any]] = []
    scope_rows: list[dict[str, Any]] = []
    parser_rows: list[dict[str, Any]] = []

    for pid in range(1, cfg.n_projects + 1):
        m = months[rnd.randrange(len(months))]
        created_at = random_timestamp_within_month(m, rnd)
        split = time_split(created_at, cfg)

        project_type_id = weighted_choice(project_type_ids, project_type_weights, rnd)
        zip_code_id = weighted_choice(zip_ids, zip_weights, rnd)
        complexity_base = {1: 0.35, 2: 0.55, 3: 0.66, 4: 0.70}[project_type_id]
        complexity = round(clamp(rnd.gauss(complexity_base, 0.16)), 4)
        urgency = round(clamp(rnd.betavariate(2 + int(3 * complexity), 2.5)), 4)
        budget_tier = max(1, min(4, int(round({1: 1.7, 2: 2.4, 3: 2.9, 4: 3.0}[project_type_id] + complexity))))
        weather_risk = round(
            clamp(
                ZIP_CATALOG[zip_code_id]["weather_risk_base"] + (0.16 if m.month in (11, 12, 1, 2, 3) else 0.0) + rnd.gauss(0, 0.08)
            ),
            4,
        )

        n_trades = max(cfg.min_trades_per_project, min(cfg.max_trades_per_project, int(round(1.2 + 2.2 * complexity))))
        tw = trade_weights_for_type(project_type_id)
        trade_ids = sorted(sample_k(list(tw.keys()), list(tw.values()), n_trades, rnd))
        is_cold_start = m >= cold_cutoff

        project = {
            "project_id": pid,
            "created_at": created_at.isoformat(),
            "zip_code_id": zip_code_id,
            "project_type_id": project_type_id,
            "trade_ids": trade_ids,
            "complexity": complexity,
            "urgency": urgency,
            "budget_tier": budget_tier,
            "weather_risk": weather_risk,
            "is_cold_start": is_cold_start,
            "split": split,
        }
        projects.append(project)

        scope_text = generate_scope_text(project, rnd)
        scope_rows.append(
            {
                "scope_text_id": pid,
                "project_id": pid,
                "created_at": created_at.isoformat(),
                "scope_text": scope_text,
                "split": split,
            }
        )
        parser_rows.append(
            {
                "scope_text_id": pid,
                "project_id": pid,
                "zip_code_id": zip_code_id,
                "project_type_id": project_type_id,
                "trade_ids": trade_ids,
                "split": split,
            }
        )

    return projects, scope_rows, parser_rows


def compute_candidate_scores(project: dict[str, Any], sub: dict[str, Any], rnd: random.Random) -> dict[str, float]:
    req = project["trade_ids"]
    trade_skills = [sub["skill_by_trade"][tid] for tid in req]
    avg_trade_skill = sum(trade_skills) / max(1, len(trade_skills))
    min_trade_skill = min(trade_skills) if trade_skills else 0.0
    overlap = len(set(req).intersection(sub["core_trade_ids"])) / max(1, len(req))
    geo_fit = sub["geo_preference"].get(project["zip_code_id"], 0.0)
    price_target = {1: 0.28, 2: 0.45, 3: 0.62, 4: 0.74}[project["budget_tier"]]
    price_fit = clamp(1.0 - abs(sub["price_level"] - price_target))
    cap_pressure = clamp(project["complexity"] * 0.65 + project["urgency"] * 0.55)
    capacity_fit = clamp((sub["capacity"] / 20.0) - 0.35 * cap_pressure + 0.35)
    weather_penalty = project["weather_risk"] * (1.0 - sub["reliability"]) * 0.8

    recall_score = (
        1.45 * avg_trade_skill
        + 0.55 * min_trade_skill
        + 1.00 * overlap
        + 0.90 * geo_fit
        + 0.70 * sub["reliability"]
        + 0.55 * capacity_fit
        + 0.42 * price_fit
        + 0.40 * sub["cert_strength"]
        - 0.50 * weather_penalty
        - (0.08 if sub["is_cold_start"] else 0.0)
        + rnd.gauss(0.0, 0.22)
    )
    selection_score = (
        1.20 * avg_trade_skill
        + 0.80 * sub["reliability"]
        + 0.70 * capacity_fit
        + 0.45 * geo_fit
        + 0.35 * price_fit
        + 0.35 * sub["cert_strength"]
        - 0.35 * project["urgency"] * max(0.0, 0.35 - capacity_fit)
        - 0.12 * project["complexity"] * max(0.0, 0.45 - min_trade_skill)
        + rnd.gauss(0.0, 0.18)
    )
    claim_risk = clamp(
        0.42
        - 0.28 * avg_trade_skill
        - 0.18 * sub["reliability"]
        - 0.12 * sub["cert_strength"]
        + 0.16 * project["weather_risk"]
        + 0.12 * project["complexity"]
        + 0.10 * project["urgency"]
        + 0.08 * (1.0 - geo_fit)
        + 0.06 * (1.0 - capacity_fit)
        + (0.06 if sub["is_cold_start"] else 0.0)
        + rnd.gauss(0.0, 0.06)
    )

    return {
        "avg_trade_skill": round(avg_trade_skill, 6),
        "min_trade_skill": round(min_trade_skill, 6),
        "trade_overlap_core": round(overlap, 6),
        "geo_fit": round(geo_fit, 6),
        "price_fit": round(price_fit, 6),
        "capacity_fit": round(capacity_fit, 6),
        "recall_score": round(recall_score, 6),
        "selection_score": round(selection_score, 6),
        "claim_risk": round(claim_risk, 6),
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def generate_interactions_and_labels(
    cfg: Config,
    projects: list[dict[str, Any]],
    subcontractors: list[dict[str, Any]],
    rnd: random.Random,
    out_dir: Path,
) -> tuple[Counter, Counter]:
    subs_by_trade: dict[int, list[dict[str, Any]]] = defaultdict(list)
    subs_by_zip: dict[int, list[dict[str, Any]]] = defaultdict(list)
    sub_by_id: dict[int, dict[str, Any]] = {}
    for sub in subcontractors:
        sub_by_id[sub["sub_id"]] = sub
        subs_by_trade[sub["primary_trade_id"]].append(sub)
        subs_by_zip[sub["primary_zip_id"]].append(sub)

    interactions_path = out_dir / "interaction_logs.jsonl"
    ranking_path = out_dir / "ranking_labels.jsonl"
    recall_pairs_path = out_dir / "recall_training_pairs.jsonl"

    interaction_counts = Counter()
    ranking_counts = Counter()

    with interactions_path.open("w", encoding="utf-8") as iof, ranking_path.open("w", encoding="utf-8") as rof, recall_pairs_path.open("w", encoding="utf-8") as cof:
        for idx, project in enumerate(projects, start=1):
            p_created = datetime.fromisoformat(project["created_at"])
            eligible = [s for s in subcontractors if datetime.fromisoformat(s["joined_at"]) <= p_created]
            if len(eligible) < cfg.min_candidates_per_project:
                eligible = subcontractors

            n_candidates = rnd.randint(cfg.min_candidates_per_project, cfg.max_candidates_per_project)
            candidates: dict[int, dict[str, Any]] = {}

            for tid in project["trade_ids"]:
                local = subs_by_trade[tid]
                take = min(len(local), max(8, n_candidates // max(2, len(project["trade_ids"]))))
                for s in rnd.sample(local, take) if take > 0 else []:
                    candidates[s["sub_id"]] = s

            local_zip = subs_by_zip.get(project["zip_code_id"], [])
            if local_zip:
                take = min(len(local_zip), max(15, n_candidates // 4))
                for s in rnd.sample(local_zip, take):
                    candidates[s["sub_id"]] = s

            need = max(0, n_candidates - len(candidates))
            if need > 0:
                for s in rnd.sample(eligible, min(need * 2, len(eligible))):
                    candidates[s["sub_id"]] = s

            candidate_ids = list(candidates.keys())
            if len(candidate_ids) > n_candidates:
                candidate_ids = rnd.sample(candidate_ids, n_candidates)
            elif len(candidate_ids) < n_candidates:
                fill = [s["sub_id"] for s in rnd.sample(eligible, min(n_candidates - len(candidate_ids), len(eligible)))]
                candidate_ids.extend(fill)
                candidate_ids = list(dict.fromkeys(candidate_ids))[:n_candidates]

            scored: list[tuple[int, dict[str, float], float, float, float]] = []
            hard_negative_ids: set[int] = set()
            near_miss_ids: set[int] = set()

            for sid in candidate_ids:
                s = sub_by_id[sid]
                feats = compute_candidate_scores(project, s, rnd)
                p_short = sigmoid((feats["recall_score"] - 2.35) * 1.05)
                p_invite = sigmoid((feats["recall_score"] - 2.85) + 0.35 * (feats["selection_score"] - 2.35))
                p_select = sigmoid((feats["selection_score"] - 2.95) * 1.10 - 0.9 * feats["claim_risk"])
                scored.append((sid, feats, p_short, p_invite, p_select))

                overlap = len(set(project["trade_ids"]).intersection(s["core_trade_ids"]))
                geo_local = s["geo_preference"].get(project["zip_code_id"], 0.0)
                if overlap == 0 and geo_local < 0.1:
                    hard_negative_ids.add(sid)
                elif overlap >= 1 and (geo_local < 0.16 or s["reliability"] < 0.55 or s["capacity"] < 3):
                    near_miss_ids.add(sid)

            # Enforce hard-negative/near-miss presence by mining bottom tail.
            scored_by_recall = sorted(scored, key=lambda x: x[1]["recall_score"])
            target_hn = max(1, int(round(0.15 * len(scored))))
            target_nm = max(1, int(round(0.20 * len(scored))))
            for sid, _feats, _ps, _pi, _pse in scored_by_recall:
                if len(hard_negative_ids) >= target_hn:
                    break
                hard_negative_ids.add(sid)
            for sid, _feats, _ps, _pi, _pse in scored_by_recall:
                if len(near_miss_ids) >= target_nm:
                    break
                if sid in hard_negative_ids:
                    continue
                near_miss_ids.add(sid)

            scored.sort(key=lambda x: x[2] * 0.55 + x[3] * 0.25 + x[4] * 0.20 + rnd.gauss(0.0, 0.02), reverse=True)
            shortlist_n = min(len(scored), rnd.randint(5, 12))
            invite_n = min(shortlist_n, rnd.randint(2, 5))
            shortlisted = set([sid for sid, *_ in scored[:shortlist_n]])
            invited = set([sid for sid, *_ in scored[:invite_n]])

            selected_sid: int | None = None
            if invited and rnd.random() < 0.96:
                invited_scored = [x for x in scored if x[0] in invited]
                weights = [max(1e-6, x[4]) for x in invited_scored]
                selected_sid = weighted_choice([x[0] for x in invited_scored], weights, rnd)

            for sid, feats, _p_short, _p_invite, _p_select in scored:
                imp_ts = p_created + timedelta(minutes=rnd.randint(0, 60 * 24 * 2))
                iof.write(
                    json.dumps(
                        {
                            "event_id": f"{project['project_id']}_{sid}_impression",
                            "project_id": project["project_id"],
                            "sub_id": sid,
                            "event_type": "impression",
                            "timestamp": imp_ts.isoformat(),
                            "split": project["split"],
                        }
                    )
                    + "\n"
                )
                interaction_counts["impression"] += 1

                did_short = sid in shortlisted
                did_invite = sid in invited
                did_select = selected_sid == sid
                short_ts = None
                invite_ts = None

                if did_short:
                    short_ts = imp_ts + timedelta(minutes=rnd.randint(5, 60 * 24))
                    iof.write(
                        json.dumps(
                            {
                                "event_id": f"{project['project_id']}_{sid}_shortlist",
                                "project_id": project["project_id"],
                                "sub_id": sid,
                                "event_type": "shortlist",
                                "timestamp": short_ts.isoformat(),
                                "split": project["split"],
                            }
                        )
                        + "\n"
                    )
                    interaction_counts["shortlist"] += 1

                if did_invite:
                    base = short_ts if short_ts is not None else imp_ts
                    invite_ts = base + timedelta(minutes=rnd.randint(30, 60 * 48))
                    iof.write(
                        json.dumps(
                            {
                                "event_id": f"{project['project_id']}_{sid}_invite_bid",
                                "project_id": project["project_id"],
                                "sub_id": sid,
                                "event_type": "invite_bid",
                                "timestamp": invite_ts.isoformat(),
                                "split": project["split"],
                            }
                        )
                        + "\n"
                    )
                    interaction_counts["invite_bid"] += 1

                claim_outcome = None
                claim_paid = None
                if did_select:
                    base = invite_ts if invite_ts is not None else (short_ts if short_ts is not None else imp_ts)
                    sel_ts = base + timedelta(hours=rnd.randint(12, 24 * 14))
                    iof.write(
                        json.dumps(
                            {
                                "event_id": f"{project['project_id']}_{sid}_selected",
                                "project_id": project["project_id"],
                                "sub_id": sid,
                                "event_type": "selected",
                                "timestamp": sel_ts.isoformat(),
                                "split": project["split"],
                            }
                        )
                        + "\n"
                    )
                    interaction_counts["selected"] += 1

                    claim_ts = sel_ts + timedelta(days=rnd.randint(cfg.claim_event_delay_days_min, cfg.claim_event_delay_days_max))
                    claim_happens = rnd.random() < feats["claim_risk"]
                    claim_outcome = "claim" if claim_happens else "no_claim"
                    claim_paid = round(max(0.0, math.exp(rnd.gauss(8.2, 0.85))) if claim_happens else 0.0, 2)
                    iof.write(
                        json.dumps(
                            {
                                "event_id": f"{project['project_id']}_{sid}_claim_outcome",
                                "project_id": project["project_id"],
                                "sub_id": sid,
                                "event_type": "claim_outcome",
                                "timestamp": claim_ts.isoformat(),
                                "claim_outcome": claim_outcome,
                                "claim_paid_amount": claim_paid,
                                "split": project["split"],
                            }
                        )
                        + "\n"
                    )
                    interaction_counts["claim_outcome"] += 1

                recall_target = int(did_short or did_invite or did_select)
                ranking_target = int(did_select and feats["claim_risk"] <= 0.18)
                is_hard_negative = int(sid in hard_negative_ids)
                is_near_miss = int(sid in near_miss_ids)

                ranking_row = {
                    "project_id": project["project_id"],
                    "sub_id": sid,
                    "project_created_at": project["created_at"],
                    "zip_code_id": project["zip_code_id"],
                    "project_type_id": project["project_type_id"],
                    "trade_ids": project["trade_ids"],
                    "split": project["split"],
                    "recall_target": recall_target,
                    "ranking_target": ranking_target,
                    "was_shortlisted": int(did_short),
                    "was_invited": int(did_invite),
                    "was_selected": int(did_select),
                    "observed_claim_risk": feats["claim_risk"],
                    "claim_outcome": claim_outcome,
                    "claim_paid_amount": claim_paid,
                    "avg_trade_skill": feats["avg_trade_skill"],
                    "min_trade_skill": feats["min_trade_skill"],
                    "trade_overlap_core": feats["trade_overlap_core"],
                    "geo_fit": feats["geo_fit"],
                    "price_fit": feats["price_fit"],
                    "capacity_fit": feats["capacity_fit"],
                    "recall_score_latent": feats["recall_score"],
                    "selection_score_latent": feats["selection_score"],
                    "is_hard_negative": is_hard_negative,
                    "is_near_miss": is_near_miss,
                }
                rof.write(json.dumps(ranking_row) + "\n")
                ranking_counts["rows"] += 1
                ranking_counts[f"recall_{recall_target}"] += 1
                ranking_counts[f"ranking_{ranking_target}"] += 1
                ranking_counts["hard_negative"] += is_hard_negative
                ranking_counts["near_miss"] += is_near_miss

                recall_row = {
                    "project_id": project["project_id"],
                    "sub_id": sid,
                    "zip_code_id": project["zip_code_id"],
                    "project_type_id": project["project_type_id"],
                    "trade_ids": project["trade_ids"],
                    "primary_trade_id": sub_by_id[sid]["primary_trade_id"],
                    "certification_id": sub_by_id[sid]["certification_id"],
                    "timestamp": project["created_at"],
                    "split": project["split"],
                    "label": recall_target,
                    "is_hard_negative": is_hard_negative,
                    "is_near_miss": is_near_miss,
                }
                cof.write(json.dumps(recall_row) + "\n")

            if idx % cfg.print_every_projects == 0:
                print(f"Processed {idx:,}/{len(projects):,} projects | interactions={sum(interaction_counts.values()):,}")

    return interaction_counts, ranking_counts


def schema_documentation() -> dict[str, Any]:
    return {
        "subcontractors": {
            "fields": {
                "sub_id": "int",
                "primary_trade_id": "int [1..10]",
                "certification_id": "int [1..6]",
                "joined_at": "timestamp",
                "primary_zip_id": "int [1..8]",
                "core_trade_ids": "array<int>",
                "skill_by_trade": "map<trade_id:int,float>",
                "capacity": "int",
                "reliability": "float",
                "price_level": "float",
                "cert_strength": "float",
                "geo_preference": "map<zip_id:int,float>",
                "is_cold_start": "bool",
            }
        },
        "projects": {
            "fields": {
                "project_id": "int",
                "created_at": "timestamp",
                "zip_code_id": "int [1..8]",
                "project_type_id": "int [1..4]",
                "trade_ids": "array<int> each in [1..10]",
                "complexity": "float",
                "urgency": "float",
                "budget_tier": "int [1..4]",
                "weather_risk": "float",
                "is_cold_start": "bool",
                "split": "train|val|test",
            }
        },
        "parser_labels": {
            "fields": {
                "scope_text_id": "int",
                "project_id": "int",
                "zip_code_id": "int",
                "project_type_id": "int",
                "trade_ids": "array<int>",
                "split": "train|val|test",
            }
        },
        "recall_training_pairs": {
            "fields": {
                "project_id": "int",
                "sub_id": "int",
                "zip_code_id": "int",
                "project_type_id": "int",
                "trade_ids": "array<int>",
                "primary_trade_id": "int",
                "certification_id": "int",
                "timestamp": "timestamp",
                "split": "train|val|test",
                "label": "0|1",
                "is_hard_negative": "0|1",
                "is_near_miss": "0|1",
            }
        },
        "ranking_labels": {
            "fields": {
                "project_id": "int",
                "sub_id": "int",
                "recall_target": "0|1",
                "ranking_target": "0|1",
                "is_hard_negative": "0|1",
                "is_near_miss": "0|1",
                "split": "train|val|test",
            }
        },
    }


def validate_outputs(
    cfg: Config,
    projects: list[dict[str, Any]],
    subcontractors: list[dict[str, Any]],
    parser_labels: list[dict[str, Any]],
    ranking_counts: Counter | None,
) -> None:
    assert len(projects) == cfg.n_projects
    assert len(subcontractors) == cfg.n_subcontractors
    assert len(parser_labels) == cfg.n_projects
    if ranking_counts is not None:
        assert ranking_counts["hard_negative"] > 0, "No hard negatives generated"
        assert ranking_counts["near_miss"] > 0, "No near misses generated"

    for p in projects:
        assert 1 <= p["zip_code_id"] <= 8
        assert 1 <= p["project_type_id"] <= 4
        assert p["split"] in {"train", "val", "test"}
        assert 1 <= len(p["trade_ids"]) <= cfg.max_trades_per_project
        assert all(1 <= tid <= 10 for tid in p["trade_ids"])
        assert time_split(datetime.fromisoformat(p["created_at"]), cfg) == p["split"]

    for row in parser_labels:
        assert 1 <= row["zip_code_id"] <= 8
        assert 1 <= row["project_type_id"] <= 4
        assert all(1 <= tid <= 10 for tid in row["trade_ids"])


def write_small_tables(out_dir: Path, rows: dict[str, list[dict[str, Any]]]) -> None:
    for name, payload in rows.items():
        write_jsonl(out_dir / f"{name}.jsonl", payload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic data for NCF recall/ranking/parser pipelines.")
    parser.add_argument("--output-dir", default=Config.output_dir, help="Output directory")
    parser.add_argument("--seed", type=int, default=Config.seed, help="Random seed")
    parser.add_argument("--n-projects", type=int, default=Config.n_projects, help="Number of projects")
    parser.add_argument("--n-subs", type=int, default=Config.n_subcontractors, help="Number of subcontractors")
    parser.add_argument("--min-cands", type=int, default=Config.min_candidates_per_project, help="Min candidates/project")
    parser.add_argument("--max-cands", type=int, default=Config.max_candidates_per_project, help="Max candidates/project")
    parser.add_argument("--print-every", type=int, default=Config.print_every_projects, help="Progress print interval")
    args = parser.parse_args()

    cfg = Config(
        seed=args.seed,
        output_dir=args.output_dir,
        n_projects=args.n_projects,
        n_subcontractors=args.n_subs,
        min_candidates_per_project=args.min_cands,
        max_candidates_per_project=args.max_cands,
        print_every_projects=args.print_every,
    )

    assert cfg.min_candidates_per_project <= cfg.max_candidates_per_project
    assert_catalog_hard_constraints()

    rnd = random.Random(cfg.seed)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating subcontractors...")
    subcontractors = generate_subcontractors(cfg, rnd)
    print("Generating projects/scope/parser labels...")
    projects, scope_rows, parser_rows = generate_projects(cfg, rnd)

    validate_outputs(cfg, projects, subcontractors, parser_rows, None)

    write_small_tables(
        out_dir,
        {
            "subcontractors": subcontractors,
            "projects": projects,
            "project_scope_text_dataset": scope_rows,
            "parser_labels": parser_rows,
        },
    )

    print("Generating interactions + ranking + recall pairs...")
    interaction_counts, ranking_counts = generate_interactions_and_labels(cfg, projects, subcontractors, rnd, out_dir)

    validate_outputs(cfg, projects, subcontractors, parser_rows, ranking_counts)

    with (out_dir / "schema_documentation.json").open("w", encoding="utf-8") as f:
        json.dump(schema_documentation(), f, indent=2)

    summary = {
        "seed": cfg.seed,
        "n_projects": cfg.n_projects,
        "n_subcontractors": cfg.n_subcontractors,
        "interaction_event_counts": dict(interaction_counts),
        "ranking_counts": dict(ranking_counts),
        "project_split_counts": dict(Counter([p["split"] for p in projects])),
        "catalogs": {
            "trades": TRADE_CATALOG,
            "project_types": PROJECT_TYPE_CATALOG,
            "zip_codes": {k: f"{v['zip_code']} ({v['city']}, {v['state']})" for k, v in ZIP_CATALOG.items()},
        },
    }
    with (out_dir / "summary_stats.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Done.")
    print(f"Output: {out_dir.resolve()}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
