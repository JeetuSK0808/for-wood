"""Stem profile (diameter vs height) for FOR-wood: geometric fit (stem.py, cached by prep.py) + a learned residual correction.

usage: python src/stem_diam.py cv            grouped 5-fold CV of the correction, RMSE by source vs the raw fit
       python src/stem_diam.py fit [--tag v1] train on all trees and write preds/stem_<tag>.csv for the test trees
The correction is a HistGradientBoostingRegressor on D_ref - D_fit with features from the fitted profile and the tree's source
(the source is read from the TreeID pattern: numeric = FI, plotN_treeN = NO, center/edge/north/road/west_N = SE)."""
import json, re, sys, zlib
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/ST_WhorlDetection_StemEstimation"


def source_of(tid):
    if re.fullmatch(r"\d+", tid): return "FI"
    if re.fullmatch(r"plot\d+_tree\d+", tid): return "NO"
    return "SE"


def load(split):
    out = {}
    for f in sorted((ROOT / "cache" / split).glob("*.npz")):
        d = np.load(f); out[f.stem] = {k: d[k] for k in ("sz", "sr", "srel", "H", "n_points", "z_crown")}
    return out


def feats(t, tid, z):
    z = np.asarray(z, float); H = float(t["H"]); sz, sr = t["sz"], t["sr"]
    dfit = 2 * np.interp(z, sz, sr); dbh = 2 * np.interp(1.3, sz, sr)
    relw = np.array([t["srel"][(sz > zz - 0.5) & (sz < zz + 0.5)].mean() if ((sz > zz - 0.5) & (sz < zz + 0.5)).any() else 0 for zz in z])
    s = source_of(tid); zc = float(t["z_crown"]) if np.isfinite(t["z_crown"]) else H
    return np.column_stack([dfit, z, z / H, np.full(len(z), H), np.full(len(z), dbh), relw, np.full(len(z), zc / H), (z > zc).astype(float),
                            np.full(len(z), np.log10(float(t["n_points"]))), np.full(len(z), s == "FI"), np.full(len(z), s == "NO"),
                            np.full(len(z), s == "SE")]), dfit


def train_rows(trees, refs):
    X, y, d0, g, src = [], [], [], [], []
    for tid, r in refs.groupby("TreeID"):
        if tid not in trees: continue
        F, dfit = feats(trees[tid], tid, r.Z.values); X.append(F); y.append(r.D.values - dfit); d0.append(dfit); g += [tid] * len(r)
        src += [source_of(tid)] * len(r)
    return np.vstack(X), np.concatenate(y), np.concatenate(d0), np.array(g), np.array(src)


def model():
    return HistGradientBoostingRegressor(max_iter=400, learning_rate=0.04, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0, random_state=0)


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "cv"; tag = sys.argv[sys.argv.index("--tag") + 1] if "--tag" in sys.argv else "v1"
    refs = pd.read_csv(DATA / "train/stem_references.csv").dropna(subset=["Z", "D"]); refs["TreeID"] = refs.TreeID.astype(str)
    trees = load("train"); X, y, d0, g, src = train_rows(trees, refs); dref = y + d0
    if mode == "cv":
        fold = np.array([zlib.crc32(t.encode()) % 5 for t in g]); pred = np.zeros(len(y))
        for k in range(5):
            m = model().fit(X[fold != k], y[fold != k]); pred[fold == k] = m.predict(X[fold == k])
        for name, est in (("raw fit", d0), ("fit + GBM", np.maximum(d0 + pred, 0.005))):
            e = est - dref
            per = {s: round(float(np.sqrt(np.mean(e[src == s] ** 2))), 4) for s in ("FI", "SE", "NO")}
            print(f"{name:10s} RMSE {np.sqrt(np.mean(e ** 2)):.4f} MAE {np.mean(np.abs(e)):.4f} bias {e.mean():+.4f} | {per}")
    else:
        m = model().fit(X, y); test = load("test"); rows = []
        for tid, t in test.items():
            H = float(t["H"]); zz = np.unique(np.round(np.concatenate([[0.6, 1.3, 2.0], np.arange(0.3, H, 0.1)]), 3))
            F, dfit = feats(t, tid, zz); D = np.maximum(dfit + m.predict(F), 0.005)
            rows += [(tid, float(a), round(float(b), 5)) for a, b in zip(zz, D)]
        (ROOT / "preds").mkdir(exist_ok=True)
        pd.DataFrame(rows, columns=["TreeID", "Z", "D"]).to_csv(ROOT / "preds" / f"stem_{tag}.csv", index=False)
        print("test trees", len(test), "rows", len(rows))


if __name__ == "__main__":
    main()
