# pocketHb

> **status: work in progress.** an external audit on 2026-06-02 caught a one-line bbox bug in the data loader that invalidates earlier numeric claims. the pipeline is now correct and the result on actual nail tissue is a null. an unexpected confound observation came out of fixing it. read [`docs/audit_response.md`](docs/audit_response.md) before assuming anything.

an open-source attempt at replicating the per-user calibration layer from the [mannino et al PNAS 2025 paper](https://www.pnas.org/doi/abs/10.1073/pnas.2424677122) on smartphone hemoglobin estimation from fingernail photos. sanguina licensed and locked the original code (patent US 12268498) — pocketHb tries to open the per-user calibration piece that turns a noisy global model into a useful per-person tracker.

method is structured similarly to [rudokaite et al, BNAIC 2025](https://openreview.net/forum?id=FEg5MG6l54) — frozen ConvNeXt-Tiny + Shades-of-Gray + per-patient mean+std + PLS+SVR+isotonic blender + per-user affine head. applied to the [yakimov et al, Nature Sci Data 2024 fingernail dataset](https://www.nature.com/articles/s41597-024-03895-9) (n=250), which is the largest publicly available smartphone-Hb cohort. NOTE: rudokaite's actual paper used a different private n=159 sanquin cohort, not yakimov — this repo is "BNAIC-style pipeline applied to the public yakimov data."

- [HF Space (demo)](https://huggingface.co/spaces/bubbaonbubba/pockethb-demo) — currently serves pre-fix weights; will be retrained when next we touch it
- [HF Hub (model weights)](https://huggingface.co/bubbaonbubba/pockethb-base) — same caveat
- [capture protocol](docs/capture_protocol.md) — still valid for any future unconfounded dataset

## current state in one paragraph

implementation is faithful and verified five independent ways (overfit positive control, multi-seed, alternate backbone, blinded reproduction, dataset-authors' reference code). on correctly-cropped nail tissue (Yakimov n=250), OOF R² = −0.058 across 3 seeds. no Hb-predictive signal. a coordinate-convention bug in the original loader put the model's input on a non-nail region in the bottom of the frame — and that variant of the pipeline gets OOF R² = +0.288. a clean top-corner background patch gets R² = −0.130. so the apparent signal lives specifically in the bottom-of-frame region, NOT in nails and NOT in arbitrary background. mechanism is unknown — the standardized Yakimov rig argues against camera/date confound; demographic leakage or paper/hand positioning are the most plausible candidates. proper controls (LODO CV, metadata-only regression, Grad-CAM) are not yet run.

## what works

- end-to-end implementation of the methodology
- correct bbox handling (after CRIT #1 fix on 2026-06-02; see [`_audit_check/`](_audit_check/) for visual proof on 20 patients)
- bbox-aware inference (`embed_image(image, bbox=...)`)
- bag-aggregate inference with leave-one-out per-photo diagnostics, no zero-std OOD vectors
- patient-uniqueness assertion in CV to prevent crop-level leakage
- 5-evidence implementation correctness suite (`docs/implementation_correctness_report.md`)
- 4-variant confound diagnostic ([`scripts/confound_test.py`](scripts/confound_test.py))
- per-user affine + MLP personalization layers (code works mechanically; nothing meaningful to personalize against, since the base model has no signal on real nails)
- iPhone capture protocol + interactive bbox annotator (`scripts/annotate_user_bboxes.py`)

## what doesn't work / current limitations

- **does not estimate hemoglobin.** R² on actual nails is −0.06. predicting the dataset mean does better.
- HF Hub weights are from before the bbox fix — they encode the broken-input model and should not be used. retraining is pending.
- HF Space serves those same weights.
- the personalization layer is implemented but has no useful base signal to refine.

## what would be needed to make this a defensible audit (TODO)

per a peer-style fact-check on 2026-06-02:

1. **leave-one-date-out CV** — does the buggy-pipeline R² survive when train and test never share a measurement date? ~30 min experiment, would prove or refute the batch-confound mechanism.
2. **acquisition-metadata-only regression** — fit Hb from PATIENT_ID order, white-reference RGB, file timestamps. if metadata alone predicts Hb, that's clean evidence of leakable signal. ~30 min.
3. **Grad-CAM / occlusion on the buggy pipeline** — localise WHERE the predictive signal lives in the bottom-region patch. concretizes the artifact. ~1 hour.

if these three controls land, the project becomes a publishable methodology audit. without them, it's an interesting unconfirmed observation.

## what would be needed to make this a working product (TODO)

per the same fact-check, the most likely paths:

1. **different dataset.** Yakimov n=250 doesn't have the signal at this scale; Mannino used n=9,061 (private). a less confounded public dataset doesn't currently exist.
2. **different signal/region.** conjunctival pallor, palm creases, sublingual mucosa — established noninvasive Hb landmarks that fingernail pallor is meant to substitute for.
3. **multispectral or near-IR imaging.** iPhone LiDAR or external NIR sensors have channels RGB doesn't.
4. **multi-CBC personalisation.** even two paired Hb anchors at meaningfully different values would let the affine layer fit slope (not just bias), giving it something real to learn.

infrastructure (HF Space, capture protocol, calibration code, inference path) is built and waiting for a dataset that meaningfully encodes Hb in nail tissue.

## NOT a medical device

not FDA cleared. do not use to estimate anyone's actual hemoglobin in any clinical, diagnostic, or treatment context. all claims are limited to the Yakimov n=250 dataset and the specific pipeline implemented here. observations do not generalize to mannino or rudokaite results on their respective private cohorts. anyone who is anemic, suspects anemia, or wants to track Hb should get a real blood test.

## reproducing the confound diagnostic

```
# clone repo, set up env (see docs/setup.md)
python scripts/download_data.py   # ~50 MB Yakimov dataset from figshare
python scripts/confound_test.py   # runs the 4-variant CV test, writes _confound_results.json
```

the output is the table referenced above. the canonical result is at [`_confound_results.json`](_confound_results.json).

## what's in this repo

- `src/pockethb/` — library (data, embed, preprocess, regressor, calibration, inference)
- `scripts/` — runnable experiments (`confound_test.py`, `multi_seed_test.py`, `backbone_efficientnet_test.py`, `overfit_50_test.py`, `annotate_user_bboxes.py`)
- `notebooks/` — exploratory Jupyter notebooks (chunks 1–6); numbers from these are PRE-BBOX-FIX and should not be cited
- `space/` — Gradio HF Space app
- `docs/` — methodology, capture protocol, implementation correctness, audit response
- `weights/` — local copy of the trained bundle (pre-fix)
- `_audit_check/` — visual proof of the bbox bug (`p1_asstored.jpg` is the bug, `p1_swapxy.jpg` is the fix)
- `_archive_pre_bbox_fix/` — snapshot of pre-fix outputs for comparison

## citation / credit

if anyone references the bbox bug or the confound finding, the external audit that caught it deserves credit — i didn't catch it in 5 internal correctness checks.

## license

MIT. see [LICENSE](LICENSE).
