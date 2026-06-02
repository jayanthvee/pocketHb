# Independent Reproduction Record

A blinded, code-and-data-only reproduction of the global model's headline numbers.
This document records **what was reproduced and how** — no interpretation.

- Date: 2026-06-01
- Repo branch/commit at time of run: `main` @ `9108a94`
- Environment: Python 3.12.6 · torch 2.12.0+cpu · timm 1.0.27 · scikit-learn 1.8.0 (CPU)

## Blinding protocol

The reproduction was performed with zero influence from any existing analysis or
writeup in the repo.

**Deliberately NOT read** (contain conclusions being verified):
- `docs/` directory
- `notebooks/` directory
- `README.md`, `CITATION.cff`, and any `.md` file
- `weights/pockethb_base_metrics.json` (the stored metrics being reproduced)
- `user_data/_results.json`, `user_data/_deaf_test_results.json`, and the
  corresponding `.png` figures
- The `metrics` key inside `weights/pockethb_base.pt` (not printed/read)
- The analysis scripts `scripts/deaf_model_test.py`, `scripts/deaf_model_followup.py`
  were not opened. (One line was incidentally surfaced by a `grep` for `bag`,
  algebraically implying a prior true-Hb-std value; this was disclosed before
  proceeding. Result #4 is a deterministic statistic of the label column with no
  analytical freedom, so it was computed independently regardless.)

**Read** (code + data + live model bundle only):
- `src/pockethb/` — `__init__.py`, `embed.py`, `preprocess.py`, `regressor.py`,
  `data.py`, `calibration.py`, `inference.py`
- `data/extracted/metadata.csv` and `data/extracted/photo/*.jpg`
- `weights/pockethb_base.pkl` (config fields + the fitted `FittedBlender`; metrics
  file was not read)
- `pyproject.toml`, `requirements.txt`, `scripts/run_chunk7.py` (bag-size constant)

## Method

1. Loaded the 250-patient metadata via `pockethb.data.load_metadata` (Hb converted
   g/L → g/dL on load).
2. Extracted nail crops via `pockethb.data.iter_crops(region="nail")`.
3. Embedded each crop with the live pipeline: Shades-of-Gray (p=6) → resize 224 →
   ImageNet normalize → frozen **ConvNeXt-Tiny** (`convnext_tiny.fb_in22k_ft_in1k`,
   `pockethb.embed.load_backbone` / `embed_crops`), giving 768-d features.
4. Aggregated to one vector per patient by element-wise **mean+std** over that
   patient's crops (`aggregate_per_patient`) → X shape (250, 1536). This is the
   model's normal aggregation path; bag size = **3** (all 250 patients list 3 nail
   bboxes; `scripts/run_chunk7.py` sets `BAG_SIZE = 3 # matches training`).
5. **Result #1** — ran the repo's own `pockethb.regressor.stratified_kfold_cv`
   (5 folds, 5 Hb-stratified bins, seed 42 — matching the bundle's stored seed),
   refitting the full standardize→PLS+SVR→isotonic→blend pipeline per fold, and
   scored the concatenated out-of-fold predictions.
6. **Results #2/#3** — applied the single shipped global blender
   (`weights/pockethb_base.pkl`) to each patient's bag-of-3 aggregate (in-sample),
   then fit `numpy.polyfit` of predicted Hb (y) on true Hb (x).
7. **Result #4** — standard deviation of the 250 true Hb labels.

## Four raw results

| # | Quantity | Value |
|---|----------|-------|
| 1 | Out-of-fold 5-fold CV **MAE** (n=250) | **2.0855 g/dL** |
| 1 | Out-of-fold 5-fold CV **R²** (n=250) | **−0.0531** |
| 3 | Linear fit pred~true — **slope** | **0.4693** |
| 3 | Linear fit pred~true — **intercept** | 6.7711 |
| 3 | Linear fit pred~true — **R²** | 0.5242 (Pearson r = 0.7240) |
| 4 | **Std of true Hb labels** (sample, ddof=1) | **2.6710 g/dL** |
| 4 | Std of true Hb labels (population, ddof=0) | 2.6656 g/dL |

**Result #2** (per-patient global predictions, bag-of-3) were generated for all 250
patients; distribution: mean 12.759, std 1.728, min 7.720, max 15.256 g/dL.

## Factual notes

- Metadata lists 3 nail bboxes for all 250 patients (750 nominal), but **719** crops
  survived `iter_crops` after in-bounds clipping (some bboxes clip to zero area).
  Every patient retained ≥1 crop, so X is (250, 1536).
- **Result #1 is out-of-fold** (pipeline refit per fold; patient held out of its own
  prediction). **Result #3 is in-sample** (the single global blender applied to the
  same 250 patients it was fit on). They are not the same quantity.
- The live global model is `weights/pockethb_base.pkl` — a ConvNeXt-Tiny feature
  extractor + fitted `FittedBlender` (PLS n_components=2, SVR C=0.5 gamma=scale,
  blend weight_pls=0.2, scaler n_features=1536).
- `weights/pockethb_base.pt` is an unused **resnet18** leftover (`config.backbone =
  'resnet18'`, state_dict keys prefixed `backbone.conv1...`) from the deleted PyTorch
  pipeline; it was not used in this reproduction.

## Reproduce

```
PYTHONPATH=src python _repro_independent.py
```

(`_repro_independent.py` at repo root; embeddings cache to `_repro_emb_cache.npz`
for instant re-runs. Both are untracked.)
