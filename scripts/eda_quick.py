"""Quick EDA: inventory, Hb stats, bbox stats. Run from repo root."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
META = ROOT / "data" / "extracted" / "metadata.csv"
PHOTOS = ROOT / "data" / "extracted" / "photo"

df = pd.read_csv(META)
print("SCHEMA:")
print(df.dtypes)
print()
print(f"rows: {len(df)}")
print(f"unique patients: {df['PATIENT_ID'].nunique()}")
print(f"unique dates (hashed): {df['MEASUREMENT_DATE'].nunique()}")
print(f"image files: {len(list(PHOTOS.glob('*.jpg')))}")
print()
print("HB (g/L):")
print(df["HB_LEVEL_GperL"].describe())
print()
print("HB IN g/dL (converted):")
print((df["HB_LEVEL_GperL"] / 10.0).describe())
print()
# anemia thresholds: WHO defines anemia at <130 g/L (men) and <120 g/L (women)
anemic_135 = (df["HB_LEVEL_GperL"] < 135).sum()
print(f"<135 g/L (rough anemic cutoff): {anemic_135}/{len(df)} ({100*anemic_135/len(df):.1f}%)")
print()
# images per patient
imgs_per_pt = df.groupby("PATIENT_ID").size()
print("IMAGES PER PATIENT:")
print(imgs_per_pt.describe())
