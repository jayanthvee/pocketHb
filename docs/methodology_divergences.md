# methodology divergences vs rudokaite et al. (bnaic 2025)

a precise audit of every methodological choice in our pocketHb chunk-3 pipeline,
cross-checked against rudokaite et al. "comparative study of cnn backbones for
hemoglobin estimation from fingernail images" (bnaic 2025). headline result we
are explaining: oof 5-fold cv on the nature 2024 nail dataset (n=250) gives
**mae 2.09 g/dL, r² −0.05, in-sample linear slope 0.47** (`docs/reproduction.md`
results #1, #3). rudokaite's deep-only convnext-tiny configuration on n=159
reports **mae 0.606 mmol/L, calibration slope 0.76** (paper tables 1, 4).

## 1. exact matches

| choice | what we did (file:line) | what rudokaite did (§) | status |
|---|---|---|---|
| backbone freezing | `embed.py:38-40` — `requires_grad=False` after `timm.create_model(...)`; no fine-tuning anywhere | §2.4: "each network was initialized with imagenet-1k pretrained weights and kept frozen" | ✓ exact |
| backbone family | `embed.py:26` — `convnext_tiny.fb_in22k_ft_in1k` | §2.4: "convnext-tiny ... from the timm library" | ≈ exact (variant ambiguity, see §3) |
| classifier head stripped + gap | `embed.py:36` — `num_classes=0, global_pool="avg"` (768-d) | §2.4: "the final classification head was removed, and global average pooling was applied to the last convolutional feature maps. embedding dimensionality was 768 ... for convnext-tiny" | ✓ exact |
| input size 224×224 | `embed.py:27,48` — `IMAGE_SIZE = 224`; `Image.resize((224,224), BILINEAR)` | §2.3: "resized to 224×224" | ✓ exact |
| imagenet normalisation | `embed.py:28-29,50` — mean (0.485, 0.456, 0.406), std (0.229, 0.224, 0.225) | §2.3: "normalized with imagenet statistics [27]" (ref 27 = imagenet/deng 2009) | ✓ exact |
| shades-of-gray p-norm | `preprocess.py:12,47` — `shades_of_gray(img, p=6)` | §2.3: "computational color constancy using the generalized gray-world model (minkowski p-norm, p=6)" | ✓ exact |
| per-patient mean+std aggregation | `embed.py:91-121`, esp. line 119: `np.concatenate([sub.mean(0), sub.std(0, ddof=0)])` → 1536-d | §2.4: "aggregated by element-wise mean and standard deviation, yielding 2×d–dimensional participant-level vectors" | ✓ exact |
| pls with inner-cv n_components | `regressor.py:46-63` — sweep 1..min(20, d, n−1), inner 5-fold | §2.5(1): "number of latent components was tuned by 5-fold inner cross-validation, searching from 1 up to 20" | ✓ exact |
| svr(rbf) grid | `regressor.py:68-69` — C ∈ {0.5, 1, 2, 4}, gamma ∈ {scale, 1e-3, 3e-3, 1e-2} | §2.5(2): identical grid, "3-fold inner cross-validation over C ∈ {0.5, 1, 2, 4} and γ ∈ {scale, 10⁻³, 3×10⁻³, 10⁻²}" | ✓ exact |
| svr inner cv folds | `regressor.py:70` — `KFold(n_splits=3)` | §2.5(2): "3-fold inner cross-validation" | ✓ exact |
| isotonic calibration per regressor | `regressor.py:112-113` — `IsotonicRegression(out_of_bounds="clip").fit(raw, y)` for both pls and svr, within each training fold (`stratified_kfold_cv` refits per fold) | §2.5(3) + §2.6: "each regressor's predictions were first corrected for bias using isotonic regression ... applied only within each training fold to avoid data leakage" | ✓ exact |
| weighted calibrated blend | `regressor.py:88-97` — grid w ∈ {0.0, 0.05, ..., 1.0}, minimise mae of `w·c_pls + (1−w)·c_svr` on train fold | §2.5(3): "two calibrated outputs were then combined into a weighted average, with the weight chosen to minimize error on the training folds" | ✓ exact |
| stratified 5-fold cv on hb bins | `regressor.py:142-158` — `StratifiedKFold(n_splits=5)` with bins = `np.digitize(y, quantiles(y, 6)[1:-1])` (5 quantile bins) | §2.6: "5-fold stratified cross-validation ... hb values were stratified into 5 quantile bins before splitting" | ✓ exact |
| subject-disjoint splits | `regressor.py:154` — X is at patient level by construction (`aggregate_per_patient` returns one row per patient); folds are over patient rows | §2.6: "ensuring that all images from a given donor stayed in the same fold" | ✓ exact |
| standardisation scope | `regressor.py:102-103` — `StandardScaler().fit_transform` on the outer training split inside `fit_blender`; transform applied to test fold | §2.5: "standardized to zero mean and unit variance based on the training set, and the same transformation was applied to the validation set" | ✓ exact |

## 2. known divergences

1. **deep-only vs color+deep fusion.** rudokaite evaluates color-only, deep-only, and color+deep early/late fusion (§2.4). we ship deep-only (no hand-crafted color statistics are passed into the regressor; `data.mean_rgb_features` exists at `data.py:101-110` but is unused in the chunk-3 pipeline). expected gap: table 1 shows deep-only mae 0.606 vs concat-early 0.610 vs late-fusion 0.614 mmol/L — the deep-only column is in fact the **best** mae overall; fusion only helps auroc. so this divergence cannot explain our collapse.

2. **dataset.** ours is the nature sci data 2024 release (n=250, dutch+ self-captures with manual nail and skin bboxes, hb in g/L → g/dL). theirs is the sanquin amersfoort donor cohort (n=159, all hb measured by donation-site physicians in mmol/L). distinct population, distinct device mix, distinct labelling pipeline. this matters; see §4.

3. **bag size per patient.** rudokaite: "3 to 5 image crops per participant" (§2.1). ours: 3 nail bboxes per patient in the metadata for all 250 patients, but only **719 of 750 nominal crops survive** after in-bounds clipping (`docs/reproduction.md` factual notes), so a handful of patients land at 1 or 2 crops. `embed.aggregate_per_patient:115-117` handles the 1-crop case by zero-filling the std half of the vector. effect on our mae is bounded (<31 of 250 vectors have any zero-fill).

4. **hb unit.** we report g/dL, they report mmol/L. conversion 1 mmol/L ≈ 1.611 g/dL. their mae 0.606 mmol/L ≈ 0.98 g/dL; our oof mae 2.09 g/dL ≈ 1.30 mmol/L — still a >2× absolute gap in the same units.

5. **fold count.** both 5-fold (§2.6 explicit; `regressor.py:146` default `n_splits=5`). ✓ no divergence — listed for completeness.

6. **calibration slope vs in-sample linear fit slope.** rudokaite's table 4 slope 0.76 is the slope of a "linear fit of predicted vs true hb" on the **isotonic-calibrated oof predictions** (§3.3, §2.6). our slope 0.47 in `_repro_independent.py:56-65` is from `np.polyfit(y, preds, 1)` where `preds` are predictions from the **shipped global blender applied in-sample to all 250 patients** — not oof. this is the closest equivalent we computed, but it is not the identical statistic. our oof predictions would give a comparable slope; we have not regenerated that exact number, and any future apples-to-apples claim should compute slope on `cv.oof_pred` vs `cv.oof_true`.

7. **bbox source.** rudokaite manually cropped nails per submission (§2.1). we use the pre-labelled nail bboxes shipped with the nature 2024 release (`data.py:64-98`); the **skin** bboxes overshoot the image bottom edge in many cases and are clipped (`data.py:80-89`), but our chunk-3 pipeline uses only `region="nail"` (`_repro_independent.py:21`), so the skin-bbox bug does not enter this result.

## 3. paper ambiguities + our assumptions

- **exact timm variant.** paper says "convnext-tiny [29] ... from the timm library" (§2.4). it does not specify a pretraining tag. we chose `convnext_tiny.fb_in22k_ft_in1k` (`embed.py:26`) — imagenet-22k pretrain, in1k fine-tune, 768-d. alternative `convnext_tiny.fb_in1k` exists. the paper says weights were "imagenet-1k pretrained"; our variant is the in22k→in1k fine-tune, which is stronger and gives near-identical or better features for downstream tasks. unverified.
- **standardiser ddof.** paper says "zero mean and unit variance" (§2.5). `sklearn.StandardScaler` uses ddof=0 by default; we use the default. paper does not specify.
- **aggregation std ddof.** we use `ddof=0` (`embed.py:119`). paper does not specify; with 3–5 crops, ddof=0 vs 1 changes each std component by factor √(n/(n−1)) ∈ [1.12, 1.22] — a uniform scale on half the features, absorbed by the standardiser. negligible.
- **inner cv for pls.** paper specifies "5-fold inner cross-validation" (§2.5(1)). we match (`regressor.py:50`). when n_train < 5 (never the case here, but the code guards via `min(5, len(y))`), behaviour is undefined in the paper; we degrade gracefully.
- **augmentation.** paper does not mention any augmentation for the frozen-backbone configuration; the description in §2.3–§2.4 is "resize → normalize → frozen backbone". we apply none (`embed.py:43-52`). consistent.
- **isotonic on training-fold raw predictions.** paper §2.6: isotonic "applied only within each training fold". we fit isotonic on `pls.predict(X_train)` and `svr.predict(X_train)` — i.e. in-sample on the training fold (`regressor.py:108-113`). paper does not specify whether their isotonic fit uses in-sample training predictions or held-out inner-cv predictions of the training fold. using in-sample training-fold predictions risks over-flat calibration (isotonic shrinks toward y_train on the same points). this is one place where a defensible alternative exists.
- **blend weight grid resolution.** paper says "weight chosen to minimize error on the training folds". we sweep 21 points on [0,1] (`regressor.py:90`); they don't specify. trivially equivalent.
- **bag size handling for n=1.** paper guarantees 3–5 crops per patient; we have a few n=1 cases (see §2.3). we zero-fill std (`embed.py:117`). their pipeline likely never encountered this case.

## 4. why none of these plausibly explain the collapse

- **deep-only choice (divergence #1):** fusion in rudokaite's table 1 changes mae by 0.004 mmol/L (0.606 → 0.610). adding hand-crafted color features cannot move us from r² −0.05 to anywhere near +0.5; the lever is the wrong size by two orders of magnitude.
- **bag-size jitter (divergence #3):** at most ~12% of patients have a degraded bag (1 or 2 crops). even if those patients contributed zero predictive information, the remaining 88% with 3 crops should still drive r² well above 0. they don't, so bag size isn't the bottleneck.
- **bbox source (divergence #7):** nail bboxes from the dataset are tight on the nail plate by construction; the skin-bbox clipping issue does not touch the nail pipeline that produced our numbers.
- **timm variant (ambiguity):** in22k→in1k vs in1k changes downstream linear-probe performance on standard benchmarks by <1% absolute. cannot account for a slope going from 0.76 to 0.47.
- **isotonic on in-sample training preds (ambiguity):** would bias us toward *better* calibration on training, not worse, and the held-out fold is independently isotonic-fit per outer fold. could not flip the sign of the gap.
- **std ddof, blend grid, augmentation (ambiguities):** all are decimal-place perturbations.

**most likely overall explanation — wider hb distribution.** rudokaite's labels have sd 0.79 mmol/L ≈ 1.27 g/dL (§2.2). our labels have sd 2.67 g/dL ≈ 1.66 mmol/L (`docs/reproduction.md` result #4) — **2.1× wider** in g/dL, **2.1× wider** in mmol/L too. with a frozen-backbone deep-only pipeline whose embeddings carry only weak hb signal (table 1 mae 0.606 mmol/L against an sd of 0.79 mmol/L is already a variance-explained of only ~1 − (0.748/0.79)² ≈ 10%; the deep features are not strong learners), the regressor compensates by shrinking toward the mean. shrinkage magnitude scales with the ratio of label variance to feature signal: the wider the distribution, the more aggressively any weak regressor regresses to the centroid. for rudokaite, residual variance ≈ label variance, so slope ≈ 0.76 is what an honest calibrated regressor produces. for us, with a label distribution stretched to span 2.7–18 g/dL (versus their 6.5–11.2 mmol/L = 10.5–18 g/dL — a roughly 4× wider tail), the same weak signal produces a slope ≈ 0.47 and an oof r² ≈ 0. our numbers are not evidence of a broken implementation; they are evidence that the frozen-backbone deep-only pipeline is at its honest performance ceiling, and that ceiling looks worse on a harder, wider, more anaemia-skewed dataset than on a tight donor cohort. fusing color features, fine-tuning, or personalisation would each be expected to move the needle, in that order.
