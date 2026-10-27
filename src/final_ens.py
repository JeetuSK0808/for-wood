"""Final whorl ensemble for a submission: average any set of CNN heatmaps (2-cm ones interpolated to the 1-cm grid), pick
per-source (thr, dist) on the out-of-fold heatmaps of the SAME ensemble (mean per-tree F1, FI and SE), keep v3's Norway
handling (only confident peaks, thr 0.6 / dist 4), and write preds/whorls_<out>.csv.

usage: python src/final_ens.py <out_tag> <tag1> <tag2> ... [--fixed-v3]
  each tag needs preds/oof_<tag>.npz (CV) and preds/test_heat_<tag>.npz (fit on all trees)
  --fixed-v3: use v3's settings (FI 0.2/8, SE 0.3/5) instead of tuning (reproduction check)
"""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "src"); import whorl_cnn as W, evalcv as E
from ensemble import load_grid, src_of


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]; fixed = "--fixed-v3" in sys.argv
    out, tags = args[0], args[1:]
    oof = E.combine_many([E.load_tags(t) for t in tags])
    cfg = {"tags": tags, "per_source": {}}
    for s in ("FI", "SE"):
        ks = sorted(k for k in oof if src_of(k) == s)
        if fixed:
            g = E.V3[s]
        else:
            res = {g: np.mean([E.tree_f1(oof[k], k, *g)[0] for k in ks]) for g in E.GRID}; g = max(res, key=res.get)
        cfg["per_source"][s] = dict(thr=float(g[0]), dist=int(g[1]), oof_tree_F1=float(np.mean([E.tree_f1(oof[k], k, *g)[0] for k in ks])))
    cfg["per_source"]["NO"] = dict(thr=0.6, dist=4)
    sc = E.score(oof, verbose=False); cfg["nested"] = {s: sc[s]["nested"] for s in ("FI", "SE")}; cfg["fise_nested"] = sc["fise_nested"]
    t2 = load_grid("cache2", "test"); t1 = load_grid("cache", "test")
    heats = [dict(np.load(f"preds/test_heat_{t}.npz")) for t in tags]; rows = []
    for k in sorted(t2):
        zb = t2[k]["zb"]
        p = np.mean([h[k] if len(h[k]) == len(zb) else np.interp(zb, t1[k]["zb"], h[k]) for h in heats], 0)
        c = cfg["per_source"][src_of(k)]; z = np.sort(W.peaks(p, zb, c["thr"], c["dist"]))
        if len(z) == 0:
            z = np.array([float(zb[int(np.argmax(p))])])
        rows += [(k, round(float(v), 4), j + 1) for j, v in enumerate(z)]
    pd.DataFrame(rows, columns=["TreeID", "Z", "whorl_number"]).to_csv(f"preds/whorls_{out}.csv", index=False)
    json.dump(cfg, open(f"preds/final_{out}.json", "w"), indent=1)
    print(json.dumps(cfg["per_source"]), "| nested FI+SE", round(cfg["fise_nested"], 4), "| test whorls", len(rows),
          {s: sum(1 for r in rows if src_of(r[0]) == s) for s in ("FI", "SE", "NO")})


if __name__ == "__main__":
    main()
