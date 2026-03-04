"""FastAPI server for NCF-style embedding two-tower subcontractor recall."""
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

try:
    from backend.risk_ranker import RiskRanker, build_ranker_features
except ModuleNotFoundError:
    from risk_ranker import RiskRanker, build_ranker_features  # type: ignore

app = FastAPI(title="Insurance Engine API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Keep deterministic outputs across backend restarts.
torch.manual_seed(42)

TRADE_LABELS = {
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

PROJECT_TYPE_LABELS = {
    1: "Residential",
    2: "Commercial",
    3: "Industrial",
    4: "Infrastructure / Civil",
}

ZIP_CODE_LABELS = {
    1: "90001 (Los Angeles, CA)",
    2: "77001 (Houston, TX)",
    3: "33101 (Miami, FL)",
    4: "60601 (Chicago, IL)",
    5: "98101 (Seattle, WA)",
    6: "80202 (Denver, CO)",
    7: "10001 (New York, NY)",
    8: "94107 (San Francisco, CA)",
}

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


class ProjectTower(nn.Module):
    """Project-side tower with categorical embeddings and MLP head."""

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

    def forward(
        self,
        zip_code_id: torch.Tensor,
        project_type_id: torch.Tensor,
        trade_needed_id: torch.Tensor,
    ) -> torch.Tensor:
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
    """Subcontractor-side tower with categorical embeddings and MLP head."""

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
        self,
        subcontractor_id: torch.Tensor,
        primary_trade_id: torch.Tensor,
        certification_id: torch.Tensor,
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


tower_project = ProjectTower()
tower_subcontractor = SubcontractorTower()
risk_ranker = RiskRanker()

DEFAULT_DATA_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_PATH = DEFAULT_DATA_DIR / "checkpoints" / "two_tower_recall.pt"
DEFAULT_RANKER_CHECKPOINT_PATH = DEFAULT_DATA_DIR / "checkpoints" / "risk_ranker.pt"
DEFAULT_RANKER_EVAL_PATH = DEFAULT_DATA_DIR / "checkpoints" / "risk_ranker_eval.json"
RECALL_MODEL_STATUS = {"loaded": False, "checkpoint_path": None, "error": None, "best_metrics": None}
RANKER_MODEL_STATUS = {
    "loaded": False,
    "checkpoint_path": None,
    "error": None,
    "best_metrics": None,
    "latest_eval": None,
}


def clamp_id(value: int, max_id: int) -> int:
    return max(1, min(max_id, int(value)))


def resolve_checkpoint_path() -> Path:
    configured = os.getenv("INSURANCE_MODEL_PATH", "").strip()
    if not configured:
        return DEFAULT_CHECKPOINT_PATH
    return Path(configured).expanduser()


def resolve_ranker_checkpoint_path() -> Path:
    configured = os.getenv("INSURANCE_RANKER_PATH", "").strip()
    if not configured:
        return DEFAULT_RANKER_CHECKPOINT_PATH
    return Path(configured).expanduser()


def resolve_ranker_eval_path() -> Path:
    configured = os.getenv("INSURANCE_RANKER_EVAL_PATH", "").strip()
    if not configured:
        return DEFAULT_RANKER_EVAL_PATH
    return Path(configured).expanduser()


def load_model_checkpoint_if_available() -> None:
    """
    Load trained tower weights if checkpoint exists.
    Falls back to deterministic seeded random weights when missing.
    """
    checkpoint_path = resolve_checkpoint_path()
    RECALL_MODEL_STATUS["checkpoint_path"] = str(checkpoint_path)

    if not checkpoint_path.exists():
        RECALL_MODEL_STATUS["loaded"] = False
        RECALL_MODEL_STATUS["error"] = f"Checkpoint not found at {checkpoint_path}; using seeded baseline weights."
        tower_project.eval()
        tower_subcontractor.eval()
        return

    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        project_state = checkpoint.get("project_tower_state_dict")
        sub_state = checkpoint.get("subcontractor_tower_state_dict")
        if not project_state or not sub_state:
            raise ValueError("Checkpoint missing required state dict keys")

        tower_project.load_state_dict(project_state)
        tower_subcontractor.load_state_dict(sub_state)
        tower_project.eval()
        tower_subcontractor.eval()
        RECALL_MODEL_STATUS["loaded"] = True
        RECALL_MODEL_STATUS["error"] = None
        RECALL_MODEL_STATUS["best_metrics"] = checkpoint.get("training", {}).get("best_val_metrics")
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        RECALL_MODEL_STATUS["loaded"] = False
        RECALL_MODEL_STATUS["error"] = f"Failed loading checkpoint: {exc}"
        tower_project.eval()
        tower_subcontractor.eval()


def load_ranker_checkpoint_if_available() -> None:
    """Load second-stage risk ranker checkpoint if available."""
    checkpoint_path = resolve_ranker_checkpoint_path()
    RANKER_MODEL_STATUS["checkpoint_path"] = str(checkpoint_path)

    if not checkpoint_path.exists():
        RANKER_MODEL_STATUS["loaded"] = False
        RANKER_MODEL_STATUS["error"] = f"Ranker checkpoint not found at {checkpoint_path}; using recall-only selection."
        RANKER_MODEL_STATUS["latest_eval"] = None
        risk_ranker.eval()
        return

    try:
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))
        state = checkpoint.get("ranker_state_dict")
        if not state:
            raise ValueError("Checkpoint missing ranker_state_dict")
        risk_ranker.load_state_dict(state)
        risk_ranker.eval()
        RANKER_MODEL_STATUS["loaded"] = True
        RANKER_MODEL_STATUS["error"] = None
        RANKER_MODEL_STATUS["best_metrics"] = checkpoint.get("training", {}).get("best_val_metrics")
        eval_path = resolve_ranker_eval_path()
        if eval_path.exists():
            try:
                RANKER_MODEL_STATUS["latest_eval"] = load_json_with_log_prefix(eval_path)
            except Exception:
                RANKER_MODEL_STATUS["latest_eval"] = None
        else:
            RANKER_MODEL_STATUS["latest_eval"] = None
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        RANKER_MODEL_STATUS["loaded"] = False
        RANKER_MODEL_STATUS["error"] = f"Failed loading ranker checkpoint: {exc}"
        RANKER_MODEL_STATUS["latest_eval"] = None
        risk_ranker.eval()


def resolve_data_dir() -> Path:
    configured = os.getenv("INSURANCE_DATA_DIR", "").strip()
    if not configured:
        return DEFAULT_DATA_DIR
    return Path(configured).expanduser()


def resolve_subcontractor_source_path() -> Path:
    """
    Resolve runtime subcontractor source.
    Priority:
    1) INSURANCE_SUBCONTRACTOR_PATH (file or directory)
    2) backend/subcontractors.json (default serving catalog)
    """
    explicit = os.getenv("INSURANCE_SUBCONTRACTOR_PATH", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    return DEFAULT_DATA_DIR / "subcontractors.json"


def load_json_with_log_prefix(path: Path) -> dict | None:
    """Load JSON even if the file contains log lines before the first '{'."""
    raw = path.read_text(encoding="utf-8").strip()
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        idx = raw.find("{")
        if idx == -1:
            return None
        try:
            return json.loads(raw[idx:])
        except Exception:
            return None


def normalize_subcontractor(raw: dict) -> dict:
    """Normalize raw rows from json/jsonl into runtime fields the API expects."""
    sub_id = clamp_id(raw.get("sub_id", raw.get("subcontractor_id", 1)), SUBCONTRACTOR_CARDINALITY)
    primary_trade_id = clamp_id(raw.get("primary_trade_id", 1), TRADE_CARDINALITY)
    certification_id = clamp_id(raw.get("certification_id", 1), CERTIFICATION_CARDINALITY)
    return {
        "sub_id": sub_id,
        "name": raw.get("name") or f"Synthetic Subcontractor {sub_id}",
        "primary_trade_id": primary_trade_id,
        "certification_id": certification_id,
        "primary_zip_id": clamp_id(raw.get("primary_zip_id", 1), ZIP_CODE_CARDINALITY),
        # Synthetic generator may not include these exact fields; provide graceful fallbacks.
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


def _load_subcontractors_from_json(json_path: Path) -> tuple[list[dict], str | None]:
    try:
        with open(json_path, encoding="utf-8") as f:
            payload = json.load(f)
        if not isinstance(payload, list):
            return [], f"Invalid format in {json_path}: expected a JSON array"
        return [normalize_subcontractor(row) for row in payload if isinstance(row, dict)], None
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        return [], f"Failed reading {json_path}: {exc}"


def _load_subcontractors_from_jsonl(jsonl_path: Path) -> tuple[list[dict], str | None]:
    rows: list[dict] = []
    try:
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                if isinstance(row, dict):
                    rows.append(normalize_subcontractor(row))
        return rows, None
    except Exception as exc:  # pragma: no cover - defensive runtime guard
        return [], f"Failed reading {jsonl_path}: {exc}"


def load_subcontractors() -> tuple[list[dict], str | None, str | None]:
    """
    Load subcontractor catalog from either a file path or a directory.
    Returns: (rows, error, source_path_string)
    """
    source = resolve_subcontractor_source_path()

    if source.is_file():
        if source.suffix.lower() == ".json":
            rows, err = _load_subcontractors_from_json(source)
            return rows, err, str(source)
        if source.suffix.lower() == ".jsonl":
            rows, err = _load_subcontractors_from_jsonl(source)
            return rows, err, str(source)
        return [], f"Unsupported subcontractor file type: {source}", str(source)

    if source.is_dir():
        json_path = source / "subcontractors.json"
        jsonl_path = source / "subcontractors.jsonl"

        if json_path.exists():
            rows, err = _load_subcontractors_from_json(json_path)
            return rows, err, str(json_path)

        if jsonl_path.exists():
            rows, err = _load_subcontractors_from_jsonl(jsonl_path)
            return rows, err, str(jsonl_path)

        return [], (
            f"Missing subcontractor dataset in {source}. Expected "
            f"{json_path.name} or {jsonl_path.name}. "
            "Set INSURANCE_SUBCONTRACTOR_PATH to a real catalog file/path."
        ), str(source)

    return [], f"Subcontractor source path does not exist: {source}", str(source)


def build_project_inputs(zip_code_id: int, project_type_id: int, trade_needed_id: int) -> tuple[torch.Tensor, ...]:
    return (
        torch.tensor([clamp_id(zip_code_id, ZIP_CODE_CARDINALITY)], dtype=torch.long),
        torch.tensor([clamp_id(project_type_id, PROJECT_TYPE_CARDINALITY)], dtype=torch.long),
        torch.tensor([clamp_id(trade_needed_id, TRADE_CARDINALITY)], dtype=torch.long),
    )


def build_subcontractor_batch(subcontractors: list[dict]) -> tuple[torch.Tensor, ...]:
    sub_ids = [clamp_id(sub["sub_id"], SUBCONTRACTOR_CARDINALITY) for sub in subcontractors]
    trade_ids = [clamp_id(sub["primary_trade_id"], TRADE_CARDINALITY) for sub in subcontractors]
    cert_ids = [clamp_id(sub["certification_id"], CERTIFICATION_CARDINALITY) for sub in subcontractors]
    return (
        torch.tensor(sub_ids, dtype=torch.long),
        torch.tensor(trade_ids, dtype=torch.long),
        torch.tensor(cert_ids, dtype=torch.long),
    )


def score_subcontractors_for_trade(
    zip_code_id: int,
    project_type_id: int,
    trade_needed_id: int,
    sub_batch: tuple[torch.Tensor, ...],
) -> list[float]:
    with torch.no_grad():
        project_emb = tower_project(*build_project_inputs(zip_code_id, project_type_id, trade_needed_id))  # (1, 64)
        sub_embs = tower_subcontractor(*sub_batch)  # (N, 64)
        dot = (sub_embs * project_emb).sum(dim=1)
        probs = torch.sigmoid(dot)
    return [round(float(p), 4) for p in probs]


class ProjectRequest(BaseModel):
    zip_code_id: int = 1
    project_type_id: int = 1
    trade_ids: list[int] = Field(default_factory=list, min_length=1)


@app.get("/api/trades")
def get_trades():
    """Return the canonical trade catalog used by the model."""
    return [{"id": trade_id, "name": name} for trade_id, name in sorted(TRADE_LABELS.items())]


@app.get("/api/project-types")
def get_project_types():
    """Return the canonical project type catalog used by the model."""
    return [{"id": type_id, "name": name} for type_id, name in sorted(PROJECT_TYPE_LABELS.items())]


@app.get("/api/zip-codes")
def get_zip_codes():
    """Return the canonical zip-code catalog used by the model/UI mapping."""
    return [{"id": zip_id, "name": name} for zip_id, name in sorted(ZIP_CODE_LABELS.items())]


@app.get("/api/model-status")
def get_model_status():
    """Expose whether a trained checkpoint is loaded."""
    _rows, _err, source_path = load_subcontractors()
    return {
        "recall_model": RECALL_MODEL_STATUS,
        "ranker_model": RANKER_MODEL_STATUS,
        "subcontractor_source_path": source_path,
    }


@app.post("/api/recommend")
def get_recommendations(payload: ProjectRequest):
    """
    NCF recall endpoint.
    Scores all subcontractors per requested trade and returns the top candidate per trade.
    """
    subcontractors, load_error, source_path = load_subcontractors()
    if load_error:
        return {"project_team": [], "error": load_error, "subcontractor_source_path": source_path}

    if not subcontractors:
        return {"project_team": [], "error": "No subcontractors available"}

    normalized_trade_ids: list[int] = []
    seen = set()
    for trade_id in payload.trade_ids:
        tid = clamp_id(trade_id, TRADE_CARDINALITY)
        if tid not in seen:
            seen.add(tid)
            normalized_trade_ids.append(tid)

    sub_batch = build_subcontractor_batch(subcontractors)
    project_team = []

    for trade_id in normalized_trade_ids:
        scores = score_subcontractors_for_trade(
            zip_code_id=payload.zip_code_id,
            project_type_id=payload.project_type_id,
            trade_needed_id=trade_id,
            sub_batch=sub_batch,
        )

        candidates_for_trade: list[tuple[dict, float]] = []
        for sub, score in zip(subcontractors, scores):
            sub_trade_id = clamp_id(sub["primary_trade_id"], TRADE_CARDINALITY)
            if sub_trade_id != trade_id:
                continue
            candidates_for_trade.append((sub, score))

        if not candidates_for_trade:
            project_team.append(
                {
                    "trade_id": trade_id,
                    "trade_name": TRADE_LABELS.get(trade_id, f"Trade {trade_id}"),
                    "claim_probability": None,
                    "predicted_claim_risk": None,
                    "baseline_claim_probability": None,
                    "baseline_predicted_claim_risk": None,
                    "risk_reduction_vs_baseline": None,
                    "baseline_subcontractor": None,
                    "selection_strategy": "none",
                    "subcontractor": None,
                }
            )
            continue

        baseline_idx = max(range(len(candidates_for_trade)), key=lambda idx: candidates_for_trade[idx][1])
        baseline_sub, baseline_score = candidates_for_trade[baseline_idx]
        baseline_risk = None
        selected_idx = baseline_idx
        best_sub = baseline_sub
        best_score = baseline_score
        best_risk = None
        selection_strategy = "recall"

        if RANKER_MODEL_STATUS["loaded"]:
            cat_rows: list[list[int]] = []
            num_rows: list[list[float]] = []
            for sub, score in candidates_for_trade:
                cat, num = build_ranker_features(
                    project_zip_id=payload.zip_code_id,
                    project_type_id=payload.project_type_id,
                    trade_needed_id=trade_id,
                    subcontractor=sub,
                    recall_prob=score,
                )
                cat_rows.append(cat)
                num_rows.append(num)

            with torch.no_grad():
                cat_tensor = torch.tensor(cat_rows, dtype=torch.long)
                num_tensor = torch.tensor(num_rows, dtype=torch.float32)
                risk_scores = risk_ranker(cat_tensor, num_tensor).tolist()

            baseline_risk = round(float(risk_scores[baseline_idx]), 4)
            selected_idx = min(range(len(risk_scores)), key=lambda idx: risk_scores[idx])
            best_sub, best_score = candidates_for_trade[selected_idx]
            best_risk = round(float(risk_scores[selected_idx]), 4)
            selection_strategy = "ranker"
        risk_reduction_vs_baseline = None
        if baseline_risk is not None and best_risk is not None:
            risk_reduction_vs_baseline = round(baseline_risk - best_risk, 4)

        project_team.append(
            {
                "trade_id": trade_id,
                "trade_name": TRADE_LABELS.get(trade_id, f"Trade {trade_id}"),
                "claim_probability": round(best_score, 4),
                "predicted_claim_risk": best_risk,
                "baseline_claim_probability": round(float(baseline_score), 4),
                "baseline_predicted_claim_risk": baseline_risk,
                "risk_reduction_vs_baseline": risk_reduction_vs_baseline,
                "baseline_subcontractor": {
                    "sub_id": baseline_sub["sub_id"],
                    "name": baseline_sub["name"],
                },
                "selection_strategy": selection_strategy,
                "subcontractor": {
                    "sub_id": best_sub["sub_id"],
                    "name": best_sub["name"],
                    "primary_trade_id": best_sub["primary_trade_id"],
                    "certification_id": best_sub["certification_id"],
                    "headcount": best_sub.get("headcount"),
                    "years_in_business": best_sub.get("years_in_business"),
                },
            }
        )

    return {
        "zip_code_id": clamp_id(payload.zip_code_id, ZIP_CODE_CARDINALITY),
        "project_type_id": clamp_id(payload.project_type_id, PROJECT_TYPE_CARDINALITY),
        "requested_trade_ids": normalized_trade_ids,
        "subcontractor_source_path": source_path,
        "project_team": project_team,
    }


load_model_checkpoint_if_available()
load_ranker_checkpoint_if_available()
