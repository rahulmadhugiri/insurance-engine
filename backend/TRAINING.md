# Recall + Ranker Training Runbook

This project supports:
- Recall model training (`two_tower_recall.pt`)
- Second-stage risk ranker training (`risk_ranker.pt`)
- Serving both models in `/api/recommend`

## 1) Train recall model (stage 1)

```bash
backend/venv/bin/python -m backend.train_two_tower \
  --data-dir <DATA_DIR> \
  --checkpoint-path backend/checkpoints/two_tower_recall.pt \
  --epochs 5 \
  --batch-size 4096
```

Notes:
- Uses `recall_training_pairs.jsonl` with the existing `split` field.
- Keeps `train` for optimization, `val` for model selection, `test` untouched during training.
- Saves the best (lowest validation loss) checkpoint.

## 2) Evaluate recall model on held-out test split

```bash
backend/venv/bin/python -m backend.evaluate_two_tower \
  --data-dir <DATA_DIR> \
  --checkpoint-path backend/checkpoints/two_tower_recall.pt \
  --max-test-rows 300000
```

## 3) Train risk ranker (stage 2)

```bash
backend/venv/bin/python -m backend.train_ranker \
  --data-dir <DATA_DIR> \
  --checkpoint-path backend/checkpoints/risk_ranker.pt \
  --epochs 6 \
  --batch-size 4096
```

What this adds:
- Predicts claim risk for candidates
- Computes policy uplift vs recall-only on held-out split

## 4) Evaluate risk ranker on held-out test split

```bash
backend/venv/bin/python -m backend.evaluate_ranker \
  --data-dir <DATA_DIR> \
  --checkpoint-path backend/checkpoints/risk_ranker.pt \
  --max-test-rows 250000
```

Optional: persist latest eval artifact for dashboard impact panel:

```bash
backend/venv/bin/python -m backend.evaluate_ranker \
  --data-dir <DATA_DIR> \
  --checkpoint-path backend/checkpoints/risk_ranker.pt \
  --max-test-rows 250000 \
  > backend/checkpoints/risk_ranker_eval.json
```

## 5) Serve recommendations with recall + ranker + real subcontractor catalog

```bash
INSURANCE_MODEL_PATH=backend/checkpoints/two_tower_recall.pt \
INSURANCE_RANKER_PATH=backend/checkpoints/risk_ranker.pt \
INSURANCE_RANKER_EVAL_PATH=backend/checkpoints/risk_ranker_eval.json \
INSURANCE_SUBCONTRACTOR_PATH=<REAL_SUBCONTRACTOR_FILE_OR_DIR> \
uvicorn backend.main:app --reload
```

Examples for `INSURANCE_SUBCONTRACTOR_PATH`:
- File: `/path/to/subcontractors.json`
- File: `/path/to/subcontractors.jsonl`
- Directory containing either file: `/path/to/catalog_dir`

`INSURANCE_DATA_DIR` can still point at synthetic training/eval data, but serving candidates should come from `INSURANCE_SUBCONTRACTOR_PATH`.

Optional status check:
- `GET /api/model-status` returns recall and ranker load status + best validation metrics.

## 6) Frontend

Start frontend normally:

```bash
npm run dev
```

The frontend sends structured IDs (`zip_code_id`, `project_type_id`, `trade_ids`) and now shows:
- model impact panel from `/api/model-status`
- per-trade predicted claim risk when ranker is loaded

## Common issue: `ModuleNotFoundError: No module named 'torch'`

If you see that error, you're likely using system Python instead of the project venv.
Use either:

```bash
backend/venv/bin/python ...
```

or activate once per terminal session:

```bash
source backend/venv/bin/activate
```
