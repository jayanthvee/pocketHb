"""Local 1-epoch smoke test for the chunk-3 training pipeline.

Trains ResNet18 on a TINY subset (10/5/5 patients) for 1 epoch on CPU
just to verify all the plumbing (dataset → loader → model → train loop → metrics)
works end-to-end before pushing the notebook for a real Colab run.

Not part of the public artifact path. Run from repo root: python scripts/smoke_train.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

torch.manual_seed(42)

from pockethb.data import load_metadata, subject_disjoint_split
from pockethb.dataset import NailCropDataset, crops_for_patients
from pockethb.model import HbRegressor
from pockethb.train import TrainConfig, train_model


def main() -> int:
    df = load_metadata()
    splits = subject_disjoint_split(df, seed=42)

    train_crops = crops_for_patients(df, splits["train"][:10], region="nail")
    val_crops = crops_for_patients(df, splits["val"][:5], region="nail")
    test_crops = crops_for_patients(df, splits["test"][:5], region="nail")
    print(f"smoke subsets: train={len(train_crops)} val={len(val_crops)} test={len(test_crops)}")

    train_ds = NailCropDataset(train_crops, train=True)
    val_ds = NailCropDataset(val_crops, train=False)
    test_ds = NailCropDataset(test_crops, train=False)

    model = HbRegressor(backbone="resnet18", pretrained=True)
    cfg = TrainConfig(epochs=1, batch_size=8, patience=10, num_workers=0, device="cpu")
    result = train_model(model, train_ds, val_ds, test_ds=test_ds, cfg=cfg)

    assert result.best_epoch == 1, "expected best_epoch=1 in single-epoch smoke"
    assert result.test_metrics is not None, "test_metrics missing"
    for level in ("crop", "patient"):
        for k in ("MAE", "RMSE", "R2", "n"):
            assert k in result.test_metrics[level], f"missing {level}.{k}"
    print("\nSMOKE PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())
