"""Tune per-source (thr, dist) for the pooled (micro) F1 with sources reweighted to the test mix (FI 45/131, SE 12/25, NO 7/33 trees).
usage: python src/tune_mix.py <tag1> <tag2> [w1]  -> preds/enssrc_<t1>_<t2>.json (overwrites) ; prints test-mix F1 vs global setting"""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "src"); import whorl_cnn as W
from ensemble import load_grid, src_of
t1, t2 = sys.argv[1:3]; w1 = float(sys.argv[3]) if len(sys.argv) > 3 else 0.5
b = pd.read_csv(W.DATA / "train/branch_references.csv"); b["TreeID"] = b.TreeID.astype(str)
refs = {k: g.Z.values for k, g in b.groupby("TreeID")}
g2 = load_grid("cache2", "train"); W.targets(g2, refs); g1 = load_grid("cache", "train")
o1 = dict(np.load(f"preds/oof_{t1}.npz")); o2 = dict(np.load(f"preds/oof_{t2}.npz"))
ens = {k: w1 * np.interp(g2[k]["zb"], g1[k]["zb"], o1[k]) + (1 - w1) * o2[k] for k in o2}
SRC = ("FI", "SE", "NO"); ntrain = {s: sum(src_of(k) == s for k in ens) for s in SRC}; ntest = {"FI": 45, "SE": 12, "NO": 7}
wt = {s: ntest[s] / ntrain[s] for s in SRC}
grid = [(round(t, 2), d) for t in np.arange(0.1, 0.61, 0.05) for d in (4, 5, 6, 8, 10, 12)]
cnt = {}
for s in SRC:
    sub = {k: v for k, v in ens.items() if src_of(k) == s}
    for thr, dist in grid:
        tp = fp = fn = 0
        for k, v in sub.items():
            z = W.peaks(v, g2[k]["zb"], thr, dist); a, b_, c, _ = W.match(z, g2[k]["Z"], 0.05); tp += a; fp += b_; fn += c
        cnt[(s, thr, dist)] = (tp, fp, fn)
def mixF1(sel, weights):
    TP = sum(weights[s] * cnt[(s,) + sel[s]][0] for s in SRC); FP = sum(weights[s] * cnt[(s,) + sel[s]][1] for s in SRC); FN = sum(weights[s] * cnt[(s,) + sel[s]][2] for s in SRC)
    P = TP / max(TP + FP, 1e-9); R = TP / max(TP + FN, 1e-9); return 2 * P * R / max(P + R, 1e-9)
glob = max(grid, key=lambda g: mixF1({s: g for s in SRC}, wt)); sel = {s: glob for s in SRC}
for _ in range(4):
    for s in SRC:
        sel[s] = max(grid, key=lambda g: mixF1({**sel, s: g}, wt))
one = {s: 1.0 for s in SRC}
print(f"global setting {glob}: test-mix F1 {mixF1({s: glob for s in SRC}, wt):.4f} | train-pooled F1 {mixF1({s: glob for s in SRC}, one):.4f}")
print(f"per-source {sel}: test-mix F1 {mixF1(sel, wt):.4f} | train-pooled F1 {mixF1(sel, one):.4f}")
cfg = {"w1": w1, "per_source": {s: dict(thr=float(sel[s][0]), dist=int(sel[s][1])) for s in SRC}, "test_mix_F1": mixF1(sel, wt), "pooled_F1": mixF1(sel, one)}
json.dump(cfg, open(f"preds/enssrc_{t1}_{t2}.json", "w"), indent=1)
