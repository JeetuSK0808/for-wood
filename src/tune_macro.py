"""Tune per-source (thr, dist) for the MEAN PER-TREE F1, which is what the official scorer reports (8 Oct: board F1 0.5064 with
P 0.4986 / R 0.5394 is below their harmonic mean 0.518, so F1 is averaged over trees, not pooled). With a per-tree mean the
sources are independent, so each source gets the setting that maximises its own mean tree F1; the test-mix estimate weights
sources by test tree counts (FI 45, SE 12, NO 7).
usage: python src/tune_macro.py <tag1> <tag2> [w1]  -> preds/ensmacro_<t1>_<t2>.json"""
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
SRC = ("FI", "SE", "NO"); NTEST = {"FI": 45, "SE": 12, "NO": 7}


def tree_f1(v, k, thr, dist):
    z = W.peaks(v, g2[k]["zb"], thr, dist)
    if len(z) == 0:                                    # submission rule: every tree gets its most confident whorl
        z = np.array([float(g2[k]["zb"][int(np.argmax(v))])])
    tp, fp, fn, _ = W.match(z, g2[k]["Z"], 0.05)
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    return 2 * P * R / max(P + R, 1e-9), P, R


grid = [(round(t, 2), d) for t in np.arange(0.05, 0.66, 0.05) for d in (4, 5, 6, 8, 10, 12, 16, 20)]
cfg = {"w1": w1, "per_source": {}}
import os  # compare against this pair's own pooled-F1 settings if they exist, else against v3's
_o = f"preds/enssrc_{t1}_{t2}.json" if os.path.exists(f"preds/enssrc_{t1}_{t2}.json") else "preds/enssrc_v1s3m_src_v2s3m_src.json"
old = json.load(open(_o))["per_source"]
est_new = est_old = 0.0
for s in SRC:
    ks = [k for k in ens if src_of(k) == s]
    res = {g: np.mean([tree_f1(ens[k], k, *g)[0] for k in ks]) for g in grid}
    best = max(res, key=res.get)
    o = old.get(s); o_f1 = np.mean([tree_f1(ens[k], k, o["thr"], o["dist"])[0] for k in ks]) if o else np.nan
    PR = np.mean([tree_f1(ens[k], k, *best)[1:] for k in ks], axis=0)
    cfg["per_source"][s] = dict(thr=float(best[0]), dist=int(best[1]), tree_F1=float(res[best]))
    est_new += NTEST[s] * res[best] / 64; est_old += NTEST[s] * o_f1 / 64
    print(f"{s}: best mean tree F1 {res[best]:.4f} at thr {best[0]} dist {best[1]} (P {PR[0]:.3f} R {PR[1]:.3f}) | old setting {o['thr'], o['dist']} -> {o_f1:.4f}")
print(f"test-mix mean tree F1 estimate: new {est_new:.4f} vs old {est_old:.4f}")
cfg["test_mix_tree_F1"] = est_new
json.dump(cfg, open(f"preds/ensmacro_{t1}_{t2}.json", "w"), indent=1)
