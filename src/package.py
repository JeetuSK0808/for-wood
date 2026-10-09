"""Build a FOR-wood submission zip from preds/whorls_<wtag>.csv and preds/stem_<stag>.csv.

usage: python src/package.py <whorl_tag> <stem_tag> <name>   -> submissions/<name>.zip (branch_references.csv + stem_references.csv)
Checks: exact column names, every test tree present, finite values, whorl_number = 1..n in height order."""
import sys, zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TEST = sorted(p.stem for p in (ROOT / "data/ST_WhorlDetection_StemEstimation/test/laz").glob("*.laz"))


def main():
    wtag, stag, name = sys.argv[1:4]
    w = pd.read_csv(ROOT / "preds" / f"whorls_{wtag}.csv"); s = pd.read_csv(ROOT / "preds" / f"stem_{stag}.csv")
    w["TreeID"] = w.TreeID.astype(str); s["TreeID"] = s.TreeID.astype(str)
    w = w.sort_values(["TreeID", "Z"]); w["whorl_number"] = w.groupby("TreeID").cumcount() + 1
    assert list(w.columns) == ["TreeID", "Z", "whorl_number"], w.columns
    assert list(s.columns) == ["TreeID", "Z", "D"], s.columns
    assert np.isfinite(w.Z).all() and np.isfinite(s[["Z", "D"]].values).all()
    miss_w = set(TEST) - set(w.TreeID); miss_s = set(TEST) - set(s.TreeID)
    print(f"whorls: {len(w)} rows, {w.TreeID.nunique()} trees (missing {sorted(miss_w)[:5]}) | stem: {len(s)} rows, {s.TreeID.nunique()} trees (missing {sorted(miss_s)[:5]})")
    out = ROOT / "submissions" / f"{name}.zip"; out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("branch_references.csv", w.to_csv(index=False)); z.writestr("stem_references.csv", s.to_csv(index=False))
    print("wrote", out, out.stat().st_size, "bytes")


if __name__ == "__main__":
    main()
