"""Interactive bbox annotator for the user's iPhone fingernail photos.

For each *.JPG / *.jpg in user_data/, opens it in a matplotlib window. Drag a
rectangle over the nail. Press ENTER to save and move to the next photo, ESC to
skip it. Bboxes persist to user_data/bboxes.json across sessions — re-running
this script starts from the last unannotated photo.

Output JSON schema:
    {
        "IMG_4435.JPG": [x1, y1, x2, y2],
        ...
    }
in original image pixel coords. Consumers read this and pass `bboxes=[...]` into
InferenceSession.predict_per_image / calibrate / run.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import RectangleSelector
from PIL import Image

USER_DIR = Path("user_data")
BBOX_PATH = USER_DIR / "bboxes.json"


def load_bboxes() -> dict[str, list[int]]:
    if BBOX_PATH.exists():
        return json.loads(BBOX_PATH.read_text())
    return {}


def save_bboxes(bboxes: dict[str, list[int]]) -> None:
    BBOX_PATH.write_text(json.dumps(bboxes, indent=2))


def annotate(image_path: Path, existing: list[int] | None) -> list[int] | None:
    img = Image.open(image_path).convert("RGB")
    W, H = img.size

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img)
    ax.set_title(
        f"{image_path.name} ({W}x{H})  |  drag a tight box around ONE nail  |  "
        "ENTER = save & next  |  ESC = skip"
    )

    state: dict[str, list[int] | None] = {"box": existing}

    if existing is not None:
        x1, y1, x2, y2 = existing
        rect = Rectangle(
            (x1, y1), x2 - x1, y2 - y1, edgecolor="lime", facecolor="none", linewidth=2
        )
        ax.add_patch(rect)

    def onselect(eclick, erelease):
        x1, y1 = int(min(eclick.xdata, erelease.xdata)), int(min(eclick.ydata, erelease.ydata))
        x2, y2 = int(max(eclick.xdata, erelease.xdata)), int(max(eclick.ydata, erelease.ydata))
        state["box"] = [x1, y1, x2, y2]

    selector = RectangleSelector(
        ax,
        onselect,
        useblit=True,
        button=[1],
        minspanx=10,
        minspany=10,
        spancoords="pixels",
        interactive=True,
        props=dict(edgecolor="red", facecolor="none", linewidth=2),
    )

    result: list[list[int] | None] = [None]

    def on_key(event):
        if event.key == "enter":
            result[0] = state["box"]
            plt.close(fig)
        elif event.key == "escape":
            result[0] = None
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    return result[0]


def main():
    if not USER_DIR.exists():
        print(f"missing {USER_DIR}/ — drop your iPhone .JPG files there first", file=sys.stderr)
        sys.exit(1)

    photos = sorted([p for p in USER_DIR.iterdir() if p.suffix.lower() in {".jpg", ".jpeg"}])
    if not photos:
        print(f"no .jpg/.jpeg files in {USER_DIR}/", file=sys.stderr)
        sys.exit(1)

    bboxes = load_bboxes()
    print(f"{len(bboxes)}/{len(photos)} already annotated.")

    for p in photos:
        existing = bboxes.get(p.name)
        if existing is not None:
            print(f"  skip {p.name} (already annotated)")
            continue
        print(f"  annotating {p.name} ...")
        box = annotate(p, existing)
        if box is None:
            print(f"    skipped {p.name}")
            continue
        bboxes[p.name] = box
        save_bboxes(bboxes)
        print(f"    saved {box}")

    print(f"done. {len(bboxes)}/{len(photos)} annotated. bboxes -> {BBOX_PATH}")


if __name__ == "__main__":
    main()
