"""Final whorl predictions for the 9 Nov 2026 rules (Skill per source, predictions above a tree's top label ignored).

FI and SE: average the CNN heatmaps, pick peaks with (thr, dist) and optionally extend above the highest peak with teeth every
`ext` m up to the tree top (`ext` 0 = off); the setting is chosen on the out-of-fold heatmaps of the same ensemble against the
given labels (mean tree F1, score_v2). NO: the model is no better than the blind comb there (nested CV, 9 Nov), so predict the
comb (start, spacing) tuned on the training trees, from the start height to the tree top. Writes preds/whorls_<out>.csv.

usage: python src/final_v2.py <out_tag> <labels.csv> <tag1> <tag2> ...
"""
import itertools, json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "src"); import whorl_cnn as W, evalcv as E, score_v2 as S, eval_v2 as V
from ensemble import load_grid, src_of

EXT = (0, 0.10, 0.15, 0.20, 0.30)


def predict(h, zb, top, thr, dist, ext):
    z = np.sort(W.peaks(h, zb, thr, dist))
    if len(z) == 0:
        z = np.array([float(zb[int(np.argmax(h))])])
    if ext > 0:
        z = np.concatenate([z, S.comb(z.max() + ext, ext, top)])
    return z


def main():
    out, labels, tags = sys.argv[1], sys.argv[2], sys.argv[3:]
    refs, g2, fold, oof, top = V.setup(labels, ",".join(tags)); cfg = {"tags": tags, "labels": labels, "per_source": {}}
    for s in ("FI", "SE"):
        ks = sorted(k for k in oof if src_of(k) == s and k in refs)
        res = {g: np.mean([S.tree_prf(predict(oof[k], g2[k]["zb"], top[k], *g), refs[k])[2] for k in ks])
               for g in itertools.product(V.THR, V.DIST, EXT)}
        g = max(res, key=res.get); cfg["per_source"][s] = dict(rule="peaks", thr=float(g[0]), dist=int(g[1]), ext=float(g[2]), oof_tree_F1=float(res[g]))
    ks = sorted(k for k in oof if src_of(k) == "NO" and k in refs)
    z0, sp, f = S.fit_comb([(refs[k], top[k]) for k in ks], z0s=np.arange(0.0, 8.01, 0.1), ss=np.arange(0.10, 0.601, 0.01))
    cfg["per_source"]["NO"] = dict(rule="comb", start=round(z0, 2), spacing=round(sp, 2), train_tree_F1=f)
    t2 = load_grid("cache2", "test"); t1 = load_grid("cache", "test")
    heats = [dict(np.load(f"preds/test_heat_{t}.npz")) for t in tags]; rows = []
    for k in sorted(t2):
        zb = t2[k]["zb"]; tp = float(zb[-1]); c = cfg["per_source"][src_of(k)]
        if c["rule"] == "comb":
            z = S.comb(c["start"], c["spacing"], tp)
        else:
            h = np.mean([x[k] if len(x[k]) == len(zb) else np.interp(zb, t1[k]["zb"], x[k]) for x in heats], 0)
            z = predict(h, zb, tp, c["thr"], c["dist"], c["ext"])
        rows += [(k, round(float(v), 4), j + 1) for j, v in enumerate(np.sort(z))]
    pd.DataFrame(rows, columns=["TreeID", "Z", "whorl_number"]).to_csv(f"preds/whorls_{out}.csv", index=False)
    json.dump(cfg, open(f"preds/final_{out}.json", "w"), indent=1)
    print(json.dumps(cfg["per_source"]), "| test whorls", len(rows), {s: sum(1 for r in rows if src_of(r[0]) == s) for s in ("FI", "SE", "NO")})


if __name__ == "__main__":
    main()
