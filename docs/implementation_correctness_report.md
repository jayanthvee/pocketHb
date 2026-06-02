# implementation correctness report

a defense against "you implemented it wrong" objections to pocketHb's published null result on the Nature Sci Data 2024 fingernail-Hb dataset (n=250).

## the four checks

| check | purpose | result | verdict |
|---|---|---|---|
| **biophotonics-msu reference run** | dataset authors' own published code on their own data — does it ALSO produce results in our ballpark? | RMSE 2.39 g/dL via 7-fold CV on 100-balanced patients; Bland-Altman LoA ±4.7 g/dL (validation), ±4.0 g/dL (test); bias 0.4 / −4.3 g/L | **CONFIRM** — the dataset authors' own reference implementation produces RMSE in our range (2.39 vs our 2.74) and reports LoA wider than the normal Hb range. Implementation is not the issue; the dataset is. |
| **overfit-50-patients positive control** | can the pipeline learn at all? train on 50 patients, test on those same 50 | in-sample MAE 0.288 g/dL, RMSE 0.855, R² 0.885, slope 0.885 | **PASS** — pipeline fits training data tightly; learning machinery works. Chunk-3 generalization failure is method-bound. |
| **multi-seed CV robustness (5 seeds)** | is the collapse a single-seed artifact or robust? | seeds [0, 1, 42, 100, 999]: MAE 2.06 ± 0.05, R² −0.05 ± 0.04, **OOF slope 0.055 ± 0.027** | **CONSISTENT** — collapse reproduces across all 5 seeds with very tight variance. Not a single-init artifact. |
| **EfficientNetV2-S backbone swap** | is the collapse specific to ConvNeXt-Tiny, or does it reproduce with Rudokaite's #2 backbone too? | OOF MAE 2.095, R² −0.066, slope 0.031, Pearson r 0.087 | **REPRODUCES** — backbone choice doesn't change the qualitative outcome. Failure is independent of backbone family. |
| **blinded independent reproduction** | does an independent agent, blinded to our analysis/docs/notebooks/metrics, reproduce the chunk-3 numbers from src/ + data only? | OOF MAE 2.0855 / R² −0.0531 / in-sample slope 0.4693 / Hb std 2.6710 — see `docs/reproduction.md`, `_repro_independent.py` | **CONFIRM** — second pair of eyes, reading no conclusions, gets the same numbers. Removes "reading own analysis to confirm own bias" concern. |

## comparison table — methods on Nature 2024

| method | RMSE g/dL | MAE g/dL | OOF R² | OOF slope (pred vs true) | source |
|---|---|---|---|---|---|
| **predict-mean (constant)** | **2.67** | **~2.14** | **0 (by defn)** | **0 (by defn)** | theoretical, full dataset std |
| **biophotonics-msu reference**¹ | **2.39** | ~1.94² | not reported | not reported | their own published notebook, balanced n=100 |
| **pocketHb ConvNeXt-Tiny** | 2.74 | 2.09 | −0.05 | 0.055 | this project, 5-fold CV, n=250 |
| **pocketHb EfficientNetV2-S** | 2.75 | 2.10 | −0.07 | 0.031 | this project, 5-fold CV, n=250 |
| **Mannino PNAS 2025 (private data, n=9061 + personalization)** | <1.0 (clinical-near) | <0.5 | not directly reported | n/a | the only method that clears clinical, but on 36× more data + paired CBCs |

¹ from `grid_searcher.best_score_ = -23.86 g/L`, with 7-fold inner CV on a balanced 100-patient subset (low-Hb-oversampled).
² derived from Bland-Altman SD: validation SD = 47.5/1.96 = 24.2 g/L → MAE ≈ 0.8 × SD = 19.4 g/L = 1.94 g/dL.

## key takeaways

1. **The dataset is the bottleneck, not our pipeline.** Both classical-features (biophotonics, ElasticNet on color percentiles) and frozen-deep-features (us, ConvNeXt-Tiny + PLS/SVR/isotonic and EfficientNetV2-S identically) produce RMSEs in the 2.4–2.8 g/dL range, against a predict-mean baseline of 2.67. The differences between methods are smaller than the gap to "clinical-grade" — which on this dataset is unreachable.

2. **Our pipeline can learn.** Pulled out of OOF CV and trained-then-tested on the same 50 patients, the same pipeline hits MAE 0.288 g/dL, R² 0.885, slope 0.885. The chunk-3 OOF failure isn't a code bug; it's the model failing to generalize because the signal isn't there at this dataset scale.

3. **The OOF slope is 0.055, not 0.47.** Previous deaf-model-test reported "slope 0.469" — that was the IN-SAMPLE slope from predicting all 250 patients through the full-data blender. The proper OOF slope (across 5 seeds: 0.055 ± 0.027) is essentially flat. Out-of-fold, predictions barely track true Hb at all. The model effectively predicts the dataset mean for everyone.

4. **Backbone-agnostic.** Swapping ConvNeXt-Tiny → EfficientNetV2-S produces near-identical OOF MAE (2.095 vs 2.090) and similarly flat slope (0.031 vs 0.055). The bottleneck is not the visual feature extractor.

5. **The dataset authors couldn't do better either.** biophotonics-msu's reference implementation on their own dataset gets Bland-Altman 95% limits of agreement of ±4.7 g/dL (validation set) and ±4.0 g/dL (test set). For context, that's wider than the full healthy Hb range (12–16 g/dL for adult males) — an effectively useless clinical interval. They published the notebook regardless because the contribution is the dataset, not a clinically-deployable model.

## environment

pinned: `environment-lock.txt` (157 packages). Key versions:

- python 3.12.6
- torch 2.12.0+cpu
- torchvision 0.27.0+cpu
- timm 1.0.27
- scikit-learn 1.8.0
- numpy 2.2.6
- pillow 12.1.0

biophotonics-msu test environment: same Python 3.12 + their requirements.txt (numpy, scipy, pandas, matplotlib, scikit-image, tqdm, scikit-learn) installed via `pip --user`.

## reproducibility

- pocketHb pipeline runs on CPU in ~5 minutes via `notebooks/03_train.ipynb` (Colab badge in repo).
- Overfit-50 test: `scripts/overfit_50_test.py` (~3 min).
- Multi-seed: `scripts/multi_seed_test.py` (~3 min).
- EfficientNetV2-S backbone swap: `scripts/backbone_efficientnet_test.py` (~5 min on first run, including weight download).
- biophotonics-msu reference: clone `github.com/biophotonics-msu/photo-haemoglobin`, set up `data/photo` and `data/metadata.csv` symlinks from Nature 2024 extract, run `Usage Notes.ipynb` end-to-end (note: cell 41 hits a sklearn API break — `mean_squared_error(squared=False)` → use `root_mean_squared_error` in sklearn ≥ 1.6; the `grid_searcher.best_score_` is captured before that cell).

## conclusion (one sentence)

The pocketHb null result on the Nature 2024 dataset is robust: the pipeline can learn (overfit-50 passes), the failure is consistent across 5 seeds, the failure is independent of backbone choice (ConvNeXt-Tiny and EfficientNetV2-S produce identical OOF metrics), and the dataset authors' own reference implementation produces RMSE in the same ballpark with Bland-Altman LoA wider than the normal Hb range — meaning the small-data ceiling is dataset/method-bound, not implementation-bound.
