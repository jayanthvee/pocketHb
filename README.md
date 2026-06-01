# pocketHb

an open-source replication of the personalization layer from the [mannino et al PNAS 2025 paper](https://www.pnas.org/doi/abs/10.1073/pnas.2424677122) on smartphone hemoglobin estimation from fingernail photos. sanguina licensed and locked the original code (patent US 12268498) — pocketHb opens the per-user calibration piece that turns a noisy global model into a useful per-person tracker.

- [live demo](https://huggingface.co/spaces/bubbaonbubba/pockethb-demo) on huggingface spaces
- [model weights](https://huggingface.co/bubbaonbubba/pockethb-base) on huggingface hub
- [capture protocol](docs/capture_protocol.md) for taking your own photos

## what's actually in here

the paper showed you can estimate someone's hemoglobin from a phone photo of their fingernails. the **novel part** wasn't the photo-to-Hb mapping (plenty of people have done that). it was the **personalization**: give the model one real bloodwork value as a baseline and it gets meaningfully more accurate for that specific person over time. mannino et al. needed 9,061 paired CBC subjects plus the calibration layer to reach clinically-near accuracy. every existing OSS anemia repo i could find does single-image binary "anemic vs non-anemic" classification. nobody's published the calibration layer. that's the gap this fills.

pocketHb is:

1. a **global Hb regressor** trained on the [Nature Sci Data 2024 fingernail+Hb dataset](https://www.nature.com/articles/s41597-024-03895-9) — frozen ConvNeXt-Tiny embeddings + classical regression. this part is intentionally lightweight and admits its limits.
2. a **per-user calibration layer** (affine v1 + optional MLP v2) that fits per-person against your real bloodwork reading. this is the contribution.
3. a **live HF Space demo** where you can upload your own photos and watch the calibrator fit in the browser.

## status

| chunk | what | result |
|---|---|---|
| 1 | bootstrap | dataset downloaded (md5-verified), 250 subjects, Hb range 4.4–16.9 g/dL |
| 2 | linear baseline | mean-RGB ridge → test patient-MAE 1.79 g/dL, R² ≈ 0. proves spatial features are needed. |
| 3 (redo) | global regressor (BNAIC pipeline) | frozen ConvNeXt-Tiny + Shades-of-Gray + PLS/SVR/isotonic, **5-fold CV OOF MAE 2.09 g/dL, R² −0.05**. honest ceiling on this dataset. |
| 4 | personalization v1 | affine calibration. single-anchor → bias correction. multi-anchor → full LS. |
| 5 | personalization v2 | per-user MLP head. honest finding: at typical anchor scale, v1 affine wins both regimes. v2 reserved for users with many distinct CBCs. |
| 6 | iPhone inference pipeline | `InferenceSession` class + capture protocol. drop photos into `user_data/`, run notebook. |
| 7 | personalize to user | in progress (waiting on the user's own iPhone captures + their real Hb anchor) |
| 8 | live HF Space demo | deployed at [bubbaonbubba/pockethb-demo](https://huggingface.co/spaces/bubbaonbubba/pockethb-demo) |
| 9 | repro polish | README, model card, citation block, deleted dead PyTorch code |

## why the global model's MAE looks bad in isolation

OOF MAE 2.09 g/dL on n=250 sounds discouraging on its own. context:

- the dataset has Hb std 2.67 g/dL — a wide distribution from severe anemia (4.4) to normal-high (16.9). the predict-mean baseline already gives MAE ~2.14. R² ≈ 0 in this regime is not "model broken" — it's "global model isn't extracting more signal than the dataset mean, which is the literature's well-known ceiling at sub-1000-subject scale on regression."
- the BNAIC 2025 paper (closest published peer, n=159 Dutch donors) got MAE 0.6 mmol/L (≈0.97 g/dL) using the *same pipeline* — but on a narrow donor population with Hb std 0.79 mmol/L. their R² was also near zero; the absolute MAE was small because their distribution was tight.
- the entire field acknowledges this. mannino built personalization on top of a global model that's only modestly better than chance, *and that was the point*: the per-user calibration is what makes it useful.

**so the headline number for this project is NOT the global MAE. it's the per-user calibrated MAE on a real subject. that's chunk 7 — waiting on iPhone captures.**

## quickstart

### option a — try the live demo

drop 3+ nail photos at [the HF Space](https://huggingface.co/spaces/bubbaonbubba/pockethb-demo), optionally enter your real Hb in g/dL, hit "run". the calibrator fits on the spot.

### option b — run the notebooks locally

```bash
git clone https://github.com/jayanthvee/pocketHb.git
cd pocketHb
pip install -e .
python scripts/download_data.py
jupyter notebook
```

each notebook in `notebooks/` is a self-contained chapter (01 bootstrap → 02 baseline → 03 train → 04 affine → 05 MLP → 06 iPhone). each one has an "open in Colab" badge — anyone can rerun the full pipeline in ~5 minutes on a T4.

### option c — programmatic inference

```python
from pockethb.inference import InferenceSession

sess = InferenceSession.from_hub()             # loads bubbaonbubba/pockethb-base
result = sess.run(photo_paths, true_hb_g_per_dL=15.3)
print(result.personal_aggregate)               # personalised Hb estimate
print(result.notes)                            # calibrator state
```

## methodology, in one paragraph

`pockethb.embed.load_backbone()` loads ConvNeXt-Tiny via `timm` (ImageNet-22k pretrained, classifier stripped) and freezes it. each crop goes through `pockethb.preprocess.shades_of_gray(p=6)` for illumination correction, resize to 224×224, ImageNet normalize, then the frozen backbone → 768-d embedding. crops are aggregated per patient as `[mean(embedding), std(embedding)]` → 1536-d patient vector. that vector goes through standardize → PLS (n_components inner-CV tuned) + SVR(RBF) (C/γ inner-CV tuned) → isotonic-calibrated → weighted blend. 5-fold stratified-by-Hb CV for honest generalization. personalization v1 is two scalars fit per user against their CBC anchors; v2 is a 2-layer MLP on the frozen embeddings, with leave-one-out early stopping. methodology entirely traceable to Rudokaite et al., BNAIC 2025.

## repo structure

```
pocketHb/
├── src/pockethb/
│   ├── preprocess.py         # Shades-of-Gray illumination correction
│   ├── embed.py              # frozen ConvNeXt-Tiny via timm
│   ├── regressor.py          # PLS + SVR + isotonic blender, 5-fold CV
│   ├── calibration.py        # AffineCalibrator (v1) + PersonalHead (v2)
│   ├── inference.py          # InferenceSession — bundle + photo → Hb
│   └── data.py               # metadata + nail-bbox crop iteration
├── notebooks/
│   ├── 01_bootstrap.ipynb    # EDA + sanity
│   ├── 02_baseline.ipynb     # linear floor
│   ├── 03_train.ipynb        # global model train + 5-fold CV
│   ├── 04_personalize_v1.ipynb
│   ├── 05_personalize_v2.ipynb
│   └── 06_iphone_inference.ipynb
├── scripts/
│   ├── download_data.py      # Figshare with md5 verify
│   ├── eda_quick.py
│   ├── diagnose_bboxes.py
│   ├── diagnose_skin_loss.py # documents the 600x800 vs labelled-frame issue
│   └── dump_nb_outputs.py
├── space/                    # gradio source mirrored to bubbaonbubba/pockethb-demo
└── docs/
    └── capture_protocol.md
```

## limitations

- **dataset scale.** n=250 with one CBC per subject. the field has nothing larger in public Hb-regression with paired bloodwork — adjacent datasets (Mendeley `2xx4j3kjg2`, Asare 2023, Appiahene 2023) are either pediatric, binary-labelled, or non-fingernail. Tilburg/Sanquin n=159 isn't public.
- **dataset skin bboxes are partially broken.** 606 of 750 skin bboxes were labelled in a taller source frame and now sit below the bottom edge of the released 600×800 images. nail bboxes are fine. we run nail-only. `scripts/diagnose_skin_loss.py` documents this.
- **single test subject for personalization.** the personalization story is validated on one person (Hb=15.3 g/dL) so far. multi-subject personalization validation is open future work.
- **not a medical device.** this should be obvious from the R² and the dataset size. it is reinforced here, on the model card, on the demo, and in the docs.

## citations / acknowledgments

- mannino, r. g., et al. *real-world implementation of a noninvasive, ai-augmented, anemia-screening smartphone app and personalization for hemoglobin level self-monitoring.* PNAS 122(20), e2424677122 (2025). [doi](https://doi.org/10.1073/pnas.2424677122)
- rudokaite, j., et al. *comparative study of cnn backbones for hemoglobin estimation from fingernail images.* BNAIC 2025 (Tilburg / Sanquin).
- nature sci data 2024 — fingernail+Hb dataset. [doi](https://doi.org/10.1038/s41597-024-03895-9)

## not a medical device

research replication only. do not use this in place of an actual blood test. not FDA cleared. not validated clinically. not a doctor. if you need a hemoglobin reading, go get one done properly.

## license

MIT.
