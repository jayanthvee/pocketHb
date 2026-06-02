# Audit response

## What happened on 2026-06-02

An external independent code audit caught a one-line bug in `src/pockethb/data.py:load_metadata`. The CSV bounding-box convention for the Yakimov dataset is `(y1, x1, y2, x2)`. The original loader read them as `(x1, y1, x2, y2)`. The result: every nail crop the project had embedded since chunk 1 was actually a patch of empty paper background in the bottom of the frame, not the fingernail.

Visual proof in [`_audit_check/`](../_audit_check/):
- `p1_asstored.jpg` — boxes drawn from raw CSV coords land in the bottom-left corner on white paper. The hand is in the upper-right.
- `p1_swapxy.jpg` — same coords with x and y swapped. All three boxes land exactly on three fingernails.
- `p100_asstored.jpg` / `p100_swapxy.jpg` — same pattern, second patient.
- `post_fix/*.jpg` — 20 random patients after the load-time swap. Every box on its nail.

Numeric corroboration: after the fix, 750/750 nail bboxes are fully in-frame (was clipping ~20%); 750/750 skin bboxes fully in-frame (was clipping ~83%); 100% of nail crops show R > B (pink tissue signature) — pre-fix the same crops were neutral or blue-shifted (paper-in-shadow).

## What changed in the code

Three commits:

- `4408adb` — `fix(data): swap (y,x,y,x) -> (x,y,x,y) on bbox parse`
- `0c45632` — `fix(inference): require bbox or pre-cropped input; warn on full-frame`
- `863d0fc` — `fix(inference): kill the zero-std OOD aggregation path`

The first is the audit fix proper. The second and third address related issues the audit also flagged: the inference path was embedding full frames instead of nail crops, and the per-image aggregation produced `[mean, zeros]` feature vectors that were off-distribution for the bag-of-3-trained blender.

## What changed in the result

Re-running the pipeline on correctly-cropped nail tissue gives OOF R² = −0.058 across 3 seeds (`_confound_results.json`, key `nails_fixed`). That's a null result. The pre-fix headline numbers (OOF MAE 2.06, R² ≈ 0, slope ≈ 0.055 on a 5-fold stratified CV across 5 seeds) characterized the wrong thing — they were the deployed-blender output when fed broken-bbox features. Underneath that, the same broken-bbox features run through the raw patient-vector regressor gave OOF R² = +0.288 (key `nails_buggy`) — that's where the apparent signal lived.

In plain terms: the bug wasn't masking the model's ability to predict Hb. It was *creating* the apparent ability. The model was learning from a fixed paper/table region in the bottom of every photo, not from nails.

## What the confound diagnostic shows

`scripts/confound_test.py` runs the same pipeline on four different feature sources:

| variant | feature source | mean OOF R² (3 seeds) | mean Pearson r |
|---|---|---|---|
| nails_fixed | bbox-corrected real fingernails | −0.058 | 0.106 |
| nails_buggy | original (broken) bbox = bottom-of-frame paper/table | +0.288 | 0.566 |
| full_frame | whole 800×600 photo | +0.189 | 0.464 |
| bg_corner | fixed top-left 160×160 background patch | −0.130 | 0.003 |

The signal is specifically in the bottom-of-frame region the buggy bbox happened to point at. It's NOT in the actual nails and it's NOT in arbitrary background.

## What the mechanism is — honest scope

A peer-style fact-check on 2026-06-02 specifically pushed back on early framings of this as "camera/date/lighting batch confound." The Yakimov rig is deliberately standardized:

- Single Logitech C615 USB camera
- ~40×40×20 cm aluminium light box
- Single fixed-position ~7300K LED, color temperature stable to ~2.5% across the dataset
- White-reference patch at fixed coords, RGB intensity varies <3%
- Single clinical site (Moscow Hospital #67)

This rig argues AGAINST gross camera/lighting/date confound. More plausible candidates per the fact-check:

- demographic leakage (partial skin/hand presence in the bottom region carrying age/sex signal correlated with Hb)
- patient-specific paper or hand positioning artifacts
- some residual content of that region correlated with anemia status
- regression-to-the-mean on a not-particularly-narrow Hb distribution

The mechanism is genuinely unknown right now. The right framing is "shortcut learning of an unidentified type, in the genre of Zech 2018, DeGrave 2021, Winkler 2019, Geirhos 2020." None of those papers ran a background-patch baseline on this dataset; per the fact-check's literature search, no preprint / paper / GitHub issue does — the observation is novel as far as can be determined.

## What this DOES and DOES NOT mean

**DOES mean:** on this specific public dataset (Yakimov n=250), with this BNAIC-style pipeline, the predictive signal extracted by the broken pipeline is NOT in nail tissue. Something specific in the bottom-frame region encodes Hb-correlated information that isn't pallor.

**DOES NOT mean:**
- That the BNAIC 2025 paper (Rudokaite et al.) has this problem — they used a different cohort (n=159 Sanquin, not public)
- That Mannino PNAS 2025 / Sanguina has this problem — they used n=9,061 multi-device selfies
- That smartphone-Hb is "fake" — this is one dataset, one pipeline lineage
- That the underlying personalisation layer concept is wrong

## What's needed to take this further

Three controls would make the confound finding publishable as an audit:

1. **Leave-one-date-out CV** — train and test never share a measurement date. If the buggy-pipeline R² collapses, batch-mediated confound is real. ~30 min.
2. **Acquisition-metadata-only regression** — fit Hb from PATIENT_ID order, white-reference RGB, file timestamps. If metadata alone predicts Hb, that's clean evidence of leakable acquisition signal. ~30 min.
3. **Grad-CAM / occlusion maps** on the buggy pipeline — localise where the predictive signal lives in the bottom-region patch. ~1 hour.

Not yet run. Tracked as `TODO`.

## Pre-fix work that is still valid

- The faithfully implemented BNAIC-style pipeline itself
- The 5-evidence implementation correctness suite (overfit-50, multi-seed, alternate backbone, biophotonics-msu reference, blinded reproduction) — reframed as "the implementation is correct" support, not "the result is meaningful"
- Methodology divergences doc (`docs/methodology_divergences.md`)
- Capture protocol
- Per-user affine + MLP code

## Pre-fix work that is invalid

- All numeric claims in `notebooks/` outputs
- The model weights in `weights/pockethb_base.pkl` (trained on broken crops)
- The HF Hub model card claims (still based on pre-fix numbers)
- The HF Space currently serves the broken model

## Acknowledgement

I did not catch this bug. An external auditor did, in about 10 minutes, by literally drawing the bbox on a sample image. My 5 "independent" correctness checks all shared the same broken input loader and were structurally incapable of catching a data-geometry bug. The lesson is that data-geometry assertions belong at the top of any pipeline — `assert R > B on mean RGB of every crop` would have flagged this on day 1.
