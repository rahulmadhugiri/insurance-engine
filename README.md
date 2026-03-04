# Insurance Engine

Two-stage subcontractor recommendation demo for construction insurance underwriting:
1. Recall stage (NCF two-tower) finds strong candidates per required trade.
2. Ranking stage predicts claim risk and re-orders candidates to reduce expected loss.

The UI supports an underwriter-style project scope text box, auto-extracts structured IDs from backend catalogs, and returns a recommended project team by trade.

## What This Project Does

- Accepts project intake as free text plus optional advanced overrides.
- Converts intake into structured IDs: `zip_code_id`, `project_type_id`, `trade_ids`.
- Scores subcontractors with a two-tower PyTorch model.
- Optionally applies a second-stage ranker to minimize predicted claim risk.
- Displays projected claim risk per selected trade partner.
- Displays risk reduction vs recall-only baseline.
- Displays project-level team summary.

## Current Architecture

- Backend: FastAPI + PyTorch
- Frontend: Next.js (App Router) + React
- `backend/main.py`: serving API + recall inference + ranker inference
- `backend/train_two_tower.py`: recall training
- `backend/train_ranker.py`: second-stage claim-risk training
- `backend/evaluate_two_tower.py`: recall held-out eval
- `backend/evaluate_ranker.py`: ranker held-out eval
- `backend/synthetic_data_generator.py`: synthetic data generation entrypoint
- Synthetic generator output includes catalogs, projects, parser labels, interaction logs, recall pairs, and ranking labels

## Repo Layout

- `backend/` API, model code, training/eval scripts, checkpoints, sample subcontractor catalog
- `frontend/` Next.js UI
- `README.md` project runbook

## Prerequisites

- Python 3.11+ recommended
- Node.js 20+ recommended
- `venv` available

## 1) Backend Setup

```bash
python3 -m venv backend/venv
source backend/venv/bin/activate
pip install -r backend/requirements.txt
```

## 2) Frontend Setup

```bash
cd frontend
npm install
cd ..
```

## 3) Generate Synthetic Training Data

```bash
backend/venv/bin/python -m backend.synthetic_data_generator \
  --output-dir /private/tmp/insurance_synth_full \
  --n-projects 100000 \
  --n-subs 50000
```

Main outputs in `/private/tmp/insurance_synth_full`:
- `subcontractors.jsonl`
- `projects.jsonl`
- `project_scope_text_dataset.jsonl`
- `parser_labels.jsonl`
- `interaction_logs.jsonl`
- `recall_training_pairs.jsonl`
- `ranking_labels.jsonl`
- `schema_documentation.json`
- `summary_stats.json`

## 4) Train Recall Model (Stage 1)

```bash
backend/venv/bin/python -m backend.train_two_tower \
  --data-dir /private/tmp/insurance_synth_full \
  --checkpoint-path backend/checkpoints/two_tower_recall.pt \
  --epochs 5 \
  --batch-size 4096
```

## 5) Evaluate Recall on Held-out Test

```bash
backend/venv/bin/python -m backend.evaluate_two_tower \
  --data-dir /private/tmp/insurance_synth_full \
  --checkpoint-path backend/checkpoints/two_tower_recall.pt \
  --max-test-rows 300000
```

## 6) Train Risk Ranker (Stage 2)

```bash
backend/venv/bin/python -m backend.train_ranker \
  --data-dir /private/tmp/insurance_synth_full \
  --checkpoint-path backend/checkpoints/risk_ranker.pt \
  --epochs 6 \
  --batch-size 4096
```

## 7) Evaluate Ranker on Held-out Test

```bash
backend/venv/bin/python -m backend.evaluate_ranker \
  --data-dir /private/tmp/insurance_synth_full \
  --checkpoint-path backend/checkpoints/risk_ranker.pt \
  --max-test-rows 250000 \
  > backend/checkpoints/risk_ranker_eval.json
```

## 8) Run API + UI

Start backend with trained checkpoints and real serving catalog:

```bash
INSURANCE_MODEL_PATH=backend/checkpoints/two_tower_recall.pt \
INSURANCE_RANKER_PATH=backend/checkpoints/risk_ranker.pt \
INSURANCE_RANKER_EVAL_PATH=backend/checkpoints/risk_ranker_eval.json \
INSURANCE_SUBCONTRACTOR_PATH=backend/subcontractors.json \
uvicorn backend.main:app --reload
```

In another shell:

```bash
npm run dev
```

Open `http://localhost:3000`.

## Environment Variables

- `INSURANCE_MODEL_PATH`: recall checkpoint path
- `INSURANCE_RANKER_PATH`: ranker checkpoint path
- `INSURANCE_RANKER_EVAL_PATH`: ranker eval artifact used by impact panel
- `INSURANCE_SUBCONTRACTOR_PATH`: serving subcontractor source (`.json`, `.jsonl`, or directory)
- `INSURANCE_DATA_DIR`: optional default data dir for training/eval artifacts

## Important Data Flow Clarification

- Synthetic data is used to train and evaluate model weights.
- Serving candidates come from `INSURANCE_SUBCONTRACTOR_PATH`.
- If `INSURANCE_SUBCONTRACTOR_PATH` points at a real catalog, recommendations use those real subcontractors.
- If it points at synthetic catalog data, recommendations use synthetic subcontractors.

## API Endpoints

- `GET /api/trades`
- `GET /api/project-types`
- `GET /api/zip-codes`
- `GET /api/model-status`
- `POST /api/recommend`

Request body for `/api/recommend`:

```json
{
  "zip_code_id": 1,
  "project_type_id": 2,
  "trade_ids": [3, 4, 6]
}
```

## Demo Goal

Show that a two-stage retrieval + ranking stack can produce trade-by-trade recommendations that reduce expected claim risk versus recall-only selection, using realistic synthetic interaction data and a backend-driven intake catalog.
