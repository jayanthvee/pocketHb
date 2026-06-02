"""HuggingFace Space — pocketHb interactive demo.

Lets a visitor upload a few iPhone-style fingernail photos and (optionally) provide
their real bloodwork Hb. The Space loads the pocketHb global model + fits an
affine per-user calibrator on the spot, returning both raw and personalised
estimates with a small chart.

NOT a medical device. Disclaimers travel with the weights.
"""
from __future__ import annotations

import io
from pathlib import Path

import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from pockethb.inference import InferenceSession

_SESS: InferenceSession | None = None


def get_session() -> InferenceSession:
    global _SESS
    if _SESS is None:
        _SESS = InferenceSession.from_hub(repo_id="bubbaonbubba/pockethb-base")
        # Warm the frozen backbone so the first user-facing request doesn't
        # eat the ~30-60s cold-start cost on cpu-basic.
        _SESS._get_backbone()
    return _SESS


# preload at module-import time so HF Spaces boot absorbs the cold start,
# not the first user request
try:
    get_session()
    print("[startup] InferenceSession warm — backbone loaded.")
except Exception as e:
    print(f"[startup] warm-up failed (will retry on first request): {type(e).__name__}: {e}")


def _make_chart(raw_per: np.ndarray, personal_per: np.ndarray | None, true_hb: float | None) -> Image.Image:
    fig, ax = plt.subplots(figsize=(9, 4.5))
    idx = np.arange(len(raw_per))
    ax.scatter(idx, raw_per, color="red", s=60, alpha=0.7, label=f"global raw  mean={raw_per.mean():.2f}")
    if personal_per is not None:
        ax.scatter(idx, personal_per, color="green", s=60, alpha=0.7,
                   label=f"personalised  mean={personal_per.mean():.2f}")
        for i in idx:
            ax.plot([i, i], [raw_per[i], personal_per[i]], color="grey", lw=0.5, alpha=0.5)
    if true_hb is not None:
        ax.axhline(true_hb, color="black", ls="--", lw=1.2, label=f"your truth = {true_hb}")
    ax.set_xticks(idx)
    ax.set_xticklabels([f"#{i+1}" for i in idx], rotation=0, fontsize=9)
    ax.set_xlabel("photo")
    ax.set_ylabel("Hb estimate (g/dL)")
    ax.set_title("pocketHb — global vs personalised per photo")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def _to_pil(file_obj) -> Image.Image:
    if isinstance(file_obj, str):
        return Image.open(file_obj).convert("RGB")
    if isinstance(file_obj, Image.Image):
        return file_obj.convert("RGB")
    if hasattr(file_obj, "name"):
        return Image.open(file_obj.name).convert("RGB")
    raise ValueError(f"can't convert {type(file_obj)} to PIL.Image")


def predict(files, hb_anchor):
    if not files:
        return "**no photos uploaded.** drag nail photos into the box above.", None, None

    try:
        sess = get_session()
        photos = [_to_pil(f) for f in files]
        raw_per = sess.predict_per_image(photos)

        if hb_anchor is None or hb_anchor == 0 or hb_anchor == "":
            chart = _make_chart(raw_per, None, None)
            md = (
                f"### global model (no personalisation)\n\n"
                f"- mean estimate across {len(photos)} photos: **{raw_per.mean():.2f} g/dL**\n"
                f"- per-photo spread (std): {raw_per.std():.2f} g/dL\n\n"
                f"_to see the personalisation layer in action, enter your real bloodwork Hb on the right and rerun._"
            )
            df = pd.DataFrame({"photo": [f"#{i+1}" for i in range(len(photos))],
                               "global raw (g/dL)": raw_per.round(2)})
            return md, df, chart

        hb = float(hb_anchor)
        if len(photos) < 2:
            return (
                "**need at least 2 photos to fit the personalisation layer.** "
                "single-photo aggregation produces an out-of-distribution feature "
                "vector for the trained blender (the std-half collapses to zeros). "
                "upload 2+ photos of the same finger under varied lighting and rerun.",
                None, None,
            )
        cal = sess.calibrate(photos, hb)
        personal_per = cal.predict(raw_per)
        chart = _make_chart(raw_per, personal_per, hb)

        raw_mae = float(np.mean(np.abs(raw_per - hb)))
        personal_mae = float(np.mean(np.abs(personal_per - hb)))

        md = (
            f"### personalised against your Hb = **{hb:.2f} g/dL**\n\n"
            f"| | mean across photos | per-photo MAE vs your truth |\n"
            f"|---|---|---|\n"
            f"| global raw | **{raw_per.mean():.2f} g/dL** | {raw_mae:.2f} |\n"
            f"| personalised | **{personal_per.mean():.2f} g/dL** | **{personal_mae:.2f}** |\n\n"
            f"calibrator: `mode={cal.mode}, a={cal.a:.3f}, b={cal.b:+.3f}, anchors={cal.n_anchors_used}`\n\n"
            f"the personalisation step removes the global model's systematic bias for you. "
            f"residual per-photo MAE is the irreducible photo-to-photo noise (lighting, focus, "
            f"crop angle) — averaging more photos at inference time reduces it."
        )
        df = pd.DataFrame({
            "photo": [f"#{i+1}" for i in range(len(photos))],
            "global raw (g/dL)": raw_per.round(2),
            "personalised (g/dL)": personal_per.round(2),
            "err vs truth": (personal_per - hb).round(2),
        })
        return md, df, chart

    except Exception as e:
        return f"**error:** `{type(e).__name__}: {e}`", None, None


DESCRIPTION = """
# pocketHb — interactive demo

open-source replication of the **personalisation layer** from the Mannino et al. PNAS 2025
fingernail-Hb paper. drop in a few nail photos (taken with a consistent protocol — same finger,
varied lighting, white paper as reference). optionally provide your real bloodwork Hb value to
see the per-user calibrator fit on the spot.

[github](https://github.com/jayanthvee/pocketHb) ·
[base weights on HF Hub](https://huggingface.co/bubbaonbubba/pockethb-base) ·
[capture protocol](https://github.com/jayanthvee/pocketHb/blob/main/docs/capture_protocol.md)

**not a medical device. research replication only. do not use to estimate anyone's actual hemoglobin in any clinical, diagnostic, or treatment context. not FDA cleared. get a blood test.**

_runs on a free cpu-basic Space; expect ~5 s per photo. uploading 3+ photos at once is recommended — the model was trained on 3-crop bags per patient and single-photo inference is off-distribution._
"""

with gr.Blocks(title="pocketHb demo") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=2):
            files = gr.Files(
                label="upload 3–15 fingernail photos (jpg/png)",
                file_count="multiple",
                file_types=["image"],
            )
        with gr.Column(scale=1):
            hb_anchor = gr.Number(
                label="your real Hb in g/dL (optional, enables personalisation)",
                value=15.3,
                precision=2,
            )
            run_btn = gr.Button("run", variant="primary")

    summary = gr.Markdown()
    table = gr.Dataframe(label="per-photo predictions")
    chart = gr.Image(label="visualisation", type="pil")

    run_btn.click(predict, inputs=[files, hb_anchor], outputs=[summary, table, chart])


if __name__ == "__main__":
    demo.launch()
