# pocketHb — iPhone capture protocol

how to capture the ~15 fingernail photos that the personalization layer needs. follow this exactly the first time so the calibration anchors are clean.

## what you need

- iphone (15 pro max for the original demo; any modern iphone should work)
- a sheet of plain white printer paper
- your most recent CBC reading with hemoglobin in g/dL (the anchor value)
- a clean, unpolished nail (index finger recommended — same finger every time)

## physical setup

- wash hand, dry it. no lotion residue.
- sit somewhere you can rotate through 3–4 lighting setups in one session
- hold the printer paper flush next to the nail in EVERY shot — the paper sits in the same frame as the nail. this is your color calibration reference. the model uses Shades-of-Gray illumination correction, but having a known-white surface visible helps you (and any reviewer) see that the lighting was actually controlled.

## the 15 photos

take 15 photos of the same finger:

| photos | lighting condition |
|---|---|
| 4 | bright daylight near a window, indirect (not direct sun) |
| 4 | indoor overhead (kitchen / bathroom ceiling LED) |
| 4 | warm desk lamp (yellow-tinted) close to the hand |
| 3 | dim ambient — evening room light, no flash |

between conditions, give the camera 5–10 seconds to re-meter. don't use flash, ever — it creates specular highlights on the nail plate that wash out the pallor signal.

## per-shot checklist

- distance: ~15 cm (about 6 inches). nail should fill ~30% of the frame width.
- orientation: nail flat-on to the camera, not at an angle. fingerprint facing the lens.
- focus: tap the nail in the iphone camera app to lock focus.
- white paper: visible in the frame, ideally adjacent to the nail (not behind — that adds glare).
- no polish, no lotion residue, no bandaid.
- hand still. no motion blur.

## where to put the photos

drop all 15 jpegs into:

    pocketHb/user_data/

that directory is gitignored — your photos never leave your machine. name them however you want; the inference notebook in `notebooks/06_iphone_inference.ipynb` reads all `.jpg` and `.jpeg` files in that folder.

## telling the model where the nail is

the global model was trained on small (~50px) nail patches cropped out of larger photos. at inference time it expects the same kind of input: a tight crop of one nail, not a full-frame iphone shot. there are two ways to provide this:

1. **crop in the iphone photos app before dropping into `user_data/`.** simplest. zoom in until the nail fills most of the frame, hit crop, save. the model takes whatever you give it and resizes it to 224×224 internally.
2. **keep the full-frame photos and annotate bboxes once.** run `python scripts/annotate_user_bboxes.py` — for each photo, drag a rectangle around the nail, press ENTER. coordinates get persisted to `user_data/bboxes.json` and the inference notebook reads them automatically. takes ~2 minutes for 15 photos.

option 2 keeps the original photos intact (useful if you want to re-annotate later or visualise the crops). option 1 is what the live HuggingFace Space expects.

## your bloodwork anchor

edit the first cell of `notebooks/06_iphone_inference.ipynb` and set:

```python
USER_HB_G_PER_DL = 15.3        # your most recent CBC value in g/dL
USER_PHOTO_DIR = "user_data"   # where you dropped the jpegs
```

15.3 is the demo value. if your reading is in g/L (european convention), divide by 10. if it's in mmol/L, multiply by 1.61 to get g/dL.

## why this matters

the global model averaged across 250 strangers (skin tones, ages, nail thicknesses, cameras). it's biased for any one specific person — including you. by giving it 15 of YOUR nails paired with YOUR real Hb, the calibrator subtracts that bias. after calibration, predictions on future nail photos (taken with the same protocol) should land near your true Hb plus per-photo noise.

**this is still not a medical device.** the personalized number is more accurate than the global one, but neither replaces a blood test. the resume value of this project is showing that the personalization piece works end-to-end on a real user — not that it's clinically valid.
