"""Shared risk-ranker model and feature engineering for train/serve parity."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn

ZIP_CODE_CARDINALITY = 500
PROJECT_TYPE_CARDINALITY = 8
TRADE_CARDINALITY = 10
SUBCONTRACTOR_CARDINALITY = 10_000
CERTIFICATION_CARDINALITY = 8

RANKER_NUMERIC_FEATURES = [
    "recall_prob",
    "headcount_norm",
    "years_norm",
    "capacity_norm",
    "reliability",
    "price_level",
    "cert_strength",
    "trade_skill",
    "geo_fit",
    "core_trade_match",
    "is_cold_start",
    "same_zip",
    "trade_match",
]

RANKER_CATEGORICAL_FEATURES = [
    "zip_code_id",
    "project_type_id",
    "trade_needed_id",
    "sub_id",
    "primary_trade_id",
    "certification_id",
    "primary_zip_id",
]


def clamp_id(value: int, max_id: int) -> int:
    return max(1, min(max_id, int(value)))


def sigmoid_scalar(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _norm_headcount(value: Any) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(1.0, float(value) / 1000.0))


def _norm_years(value: Any) -> float:
    if value is None:
        return 0.0
    return max(0.0, min(1.0, float(value) / 50.0))


def _norm_capacity(value: Any, fallback_headcount: Any) -> float:
    if value is not None:
        return max(0.0, min(1.0, float(value) / 40.0))
    if fallback_headcount is not None:
        return max(0.0, min(1.0, float(fallback_headcount) / 400.0))
    return 0.0


def _bounded_float(value: Any, default: float = 0.5) -> float:
    if value is None:
        return default
    return max(0.0, min(1.0, float(value)))


def _lookup_map_float(mapping: Any, key: int, default: float = 0.0) -> float:
    if not isinstance(mapping, dict):
        return default
    if key in mapping:
        return _bounded_float(mapping.get(key), default)
    key_str = str(key)
    if key_str in mapping:
        return _bounded_float(mapping.get(key_str), default)
    return default


def build_ranker_features(
    project_zip_id: int,
    project_type_id: int,
    trade_needed_id: int,
    subcontractor: dict[str, Any],
    recall_prob: float,
) -> tuple[list[int], list[float]]:
    """
    Build categorical and numeric features used by the claim-risk ranker.

    The output ordering is fixed by:
      - RANKER_CATEGORICAL_FEATURES
      - RANKER_NUMERIC_FEATURES
    """
    sub_id = clamp_id(subcontractor.get("sub_id", 1), SUBCONTRACTOR_CARDINALITY)
    primary_trade_id = clamp_id(subcontractor.get("primary_trade_id", 1), TRADE_CARDINALITY)
    certification_id = clamp_id(subcontractor.get("certification_id", 1), CERTIFICATION_CARDINALITY)
    primary_zip_id = clamp_id(subcontractor.get("primary_zip_id", 1), ZIP_CODE_CARDINALITY)

    headcount = subcontractor.get("headcount")
    years = subcontractor.get("years_in_business")
    capacity = subcontractor.get("capacity")
    reliability = subcontractor.get("reliability")
    price_level = subcontractor.get("price_level")
    cert_strength = subcontractor.get("cert_strength")
    skill_by_trade = subcontractor.get("skill_by_trade")
    geo_preference = subcontractor.get("geo_preference")
    core_trade_ids = subcontractor.get("core_trade_ids") or []
    is_cold_start = subcontractor.get("is_cold_start", False)

    cat = [
        clamp_id(project_zip_id, ZIP_CODE_CARDINALITY),
        clamp_id(project_type_id, PROJECT_TYPE_CARDINALITY),
        clamp_id(trade_needed_id, TRADE_CARDINALITY),
        sub_id,
        primary_trade_id,
        certification_id,
        primary_zip_id,
    ]

    num = [
        _bounded_float(recall_prob, 0.0),
        _norm_headcount(headcount),
        _norm_years(years),
        _norm_capacity(capacity, headcount),
        _bounded_float(reliability, 0.5),
        _bounded_float(price_level, 0.5),
        _bounded_float(cert_strength, 0.5),
        _lookup_map_float(skill_by_trade, clamp_id(trade_needed_id, TRADE_CARDINALITY), 0.5),
        _lookup_map_float(geo_preference, clamp_id(project_zip_id, ZIP_CODE_CARDINALITY), 0.5),
        1.0 if clamp_id(trade_needed_id, TRADE_CARDINALITY) in set(int(x) for x in core_trade_ids) else 0.0,
        1.0 if bool(is_cold_start) else 0.0,
        1.0 if primary_zip_id == clamp_id(project_zip_id, ZIP_CODE_CARDINALITY) else 0.0,
        1.0 if primary_trade_id == clamp_id(trade_needed_id, TRADE_CARDINALITY) else 0.0,
    ]
    return cat, num


class RiskRanker(nn.Module):
    """Second-stage ranker that predicts claim risk for candidate reranking."""

    def __init__(self):
        super().__init__()
        self.zip_embedding = nn.Embedding(ZIP_CODE_CARDINALITY + 1, 16)
        self.project_type_embedding = nn.Embedding(PROJECT_TYPE_CARDINALITY + 1, 6)
        self.trade_embedding = nn.Embedding(TRADE_CARDINALITY + 1, 8)
        self.sub_embedding = nn.Embedding(SUBCONTRACTOR_CARDINALITY + 1, 24)
        self.primary_trade_embedding = nn.Embedding(TRADE_CARDINALITY + 1, 8)
        self.cert_embedding = nn.Embedding(CERTIFICATION_CARDINALITY + 1, 4)
        self.sub_zip_embedding = nn.Embedding(ZIP_CODE_CARDINALITY + 1, 8)

        cat_dim = 16 + 6 + 8 + 24 + 8 + 4 + 8
        input_dim = cat_dim + len(RANKER_NUMERIC_FEATURES)
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, cat_features: torch.Tensor, num_features: torch.Tensor) -> torch.Tensor:
        """
        cat_features: [B, 7] in RANKER_CATEGORICAL_FEATURES order.
        num_features: [B, 9] in RANKER_NUMERIC_FEATURES order.
        Returns predicted claim risk in [0, 1].
        """
        zip_emb = self.zip_embedding(cat_features[:, 0])
        ptype_emb = self.project_type_embedding(cat_features[:, 1])
        trade_emb = self.trade_embedding(cat_features[:, 2])
        sub_emb = self.sub_embedding(cat_features[:, 3])
        primary_trade_emb = self.primary_trade_embedding(cat_features[:, 4])
        cert_emb = self.cert_embedding(cat_features[:, 5])
        sub_zip_emb = self.sub_zip_embedding(cat_features[:, 6])
        x = torch.cat(
            [
                zip_emb,
                ptype_emb,
                trade_emb,
                sub_emb,
                primary_trade_emb,
                cert_emb,
                sub_zip_emb,
                num_features,
            ],
            dim=1,
        )
        logits = self.net(x).squeeze(1)
        return torch.sigmoid(logits)
