"""Honest out-of-fold evaluation of whorl heatmaps for the official metric (mean per-tree F1 at +-5 cm).

Per-source (thr, dist) are tuned with NESTED CV: for each outer fold the setting is chosen on the other 4 folds' OOF trees
and applied to the held-out fold, so the reported number carries no threshold-tuning optimism. Also reports the score at
v3's fixed settings (FI 0.2/8, SE 0.3/5, NO 0.6/4 on the 1-cm grid). Test-mix estimate weights sources by test tree counts.

usage: python src/evalcv.py <tag_2cm or -> <tag_1cm or -> [w1=0.5]
       a tag may be a comma list (averaged), e.g. v1s3m_src,v1x  ;  '-' skips that grid
"""
import sys
import numpy as np, pandas as pd
sys.path.insert(0, "src"); import whorl_cnn as W
from ensemble import load_grid, src_of

NTEST = {"FI": 45, "SE": 12, "NO": 7}
V3 = {"FI": (0.2, 8), "SE": (0.3, 5), "NO": (0.6, 4)}
GRID = [(round(t, 2), d) for t in np.arange(0.10, 0.61, 0.05) for d in (4, 5, 6, 8, 10, 12)]
_G = {}


def grids():
    if not _G:
        b = pd.read_csv(W.DATA / "train/branch_references.csv"); b["TreeID"] = b.TreeID.astype(str)
        refs = {k: g.Z.values for k, g in b.groupby("TreeID")}
        g2 = load_grid("cache2", "train"); W.targets(g2, refs); g1 = load_grid("cache", "train")
        src = dict(zip(b.TreeID, b.source)); fold = W.folds_for(sorted(g2), src)
        _G.update(g1=g1, g2=g2, fold=fold)
    return _G


def to1cm(h):
    """map a heatmap dict onto the 1-cm (cache2) grid; 2-cm (cache) heatmaps are linearly interpolated"""
    G = grids(); g1, g2 = G["g1"], G["g2"]
    return {k: (v if len(v) == len(g2[k]["zb"]) else np.interp(g2[k]["zb"], g1[k]["zb"], v)) for k, v in h.items()}


def load_tags(spec):
    if spec == "-":
        return None
    hs = [to1cm(dict(np.load(f"preds/oof_{t}.npz"))) for t in spec.split(",")]
    return {k: np.mean([h[k] for h in hs], 0) for k in hs[0]}


def combine(o1, o2, w1=0.5):
    if o1 is None:
        return o2
    if o2 is None:
        return o1
    return {k: w1 * o1[k] + (1 - w1) * o2[k] for k in o2}


def combine_many(hs, ws=None):
    hs = [h for h in hs if h is not None]; ws = ws or [1.0 / len(hs)] * len(hs)
    return {k: sum(w * h[k] for w, h in zip(ws, hs)) for k in hs[0]}


def tree_f1(v, k, thr, dist):
    g2 = grids()["g2"]
    z = W.peaks(v, g2[k]["zb"], thr, dist)
    if len(z) == 0:
        z = np.array([float(g2[k]["zb"][int(np.argmax(v))])])
    tp, fp, fn, _ = W.match(z, g2[k]["Z"], 0.05)
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    return 2 * P * R / max(P + R, 1e-9), P, R


def score(ens, verbose=True, label=""):
    """returns dict with nested per-source mean tree F1, fixed-v3 F1 and the test-mix estimates"""
    G = grids(); fold = G["fold"]; out = {}
    for s in ("FI", "SE", "NO"):
        ks = sorted(k for k in ens if src_of(k) == s)
        tab = {g: {k: tree_f1(ens[k], k, *g) for k in ks} for g in GRID}
        nested = []
        for f in range(5):
            tr = [k for k in ks if fold[k] != f]; te = [k for k in ks if fold[k] == f]
            best = max(GRID, key=lambda g: np.mean([tab[g][k][0] for k in tr]))
            nested += [tab[best][k] for k in te]
        fixed = [tree_f1(ens[k], k, *V3[s]) for k in ks]
        best_all = max(GRID, key=lambda g: np.mean([tab[g][k][0] for k in ks]))
        out[s] = dict(nested=np.mean([x[0] for x in nested]), nP=np.mean([x[1] for x in nested]), nR=np.mean([x[2] for x in nested]),
                      fixed=np.mean([x[0] for x in fixed]), best=np.mean([tab[best_all][k][0] for k in ks]), best_g=best_all,
                      per_tree=np.array([x[0] for x in nested]))
    # NO is handled as in v3 (abstain, fixed 0.6/4) for the test-mix estimate: its test labels did not follow train
    mix_nested = sum(NTEST[s] * (out[s]["nested"] if s != "NO" else out[s]["fixed"]) for s in NTEST) / 64
    mix_fixed = sum(NTEST[s] * out[s]["fixed"] for s in NTEST) / 64
    out["mix_nested"] = mix_nested; out["mix_fixed"] = mix_fixed
    out["fise_nested"] = (45 * out["FI"]["nested"] + 12 * out["SE"]["nested"]) / 57
    if verbose:
        print(f"{label:28s} FI nested {out['FI']['nested']:.4f} (P {out['FI']['nP']:.3f} R {out['FI']['nR']:.3f}) fixed {out['FI']['fixed']:.4f} | "
              f"SE nested {out['SE']['nested']:.4f} fixed {out['SE']['fixed']:.4f} | NO fixed {out['NO']['fixed']:.4f} nested {out['NO']['nested']:.4f} | "
              f"FI+SE nested {out['fise_nested']:.4f} | mix nested {mix_nested:.4f} fixed {mix_fixed:.4f}", flush=True)
    return out


if __name__ == "__main__":
    a = sys.argv[1:]
    score(combine_many([load_tags(x) for x in a]), label=" + ".join(a))
