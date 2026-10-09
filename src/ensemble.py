"""Average two whorl heatmap models on the 1-cm grid and pick peaks.
usage: python src/ensemble.py oof <tag1(2cm)> <tag2(1cm)> [w1]          -> tune thr/dist on out-of-fold heatmaps, write preds/ens_<t1>_<t2>.json
       python src/ensemble.py test <tag1> <tag2> <out_tag>              -> preds/whorls_<out_tag>.csv from preds/test_heat_<tag>.npz"""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, "src"); import whorl_cnn as W


def load_grid(cache, split):
    W.CACHE = cache; return W.load(split)


def src_of(t):
    import re
    if re.fullmatch(r"\d+", t): return "FI"
    if re.fullmatch(r"plot\d+_tree\d+", t): return "NO"
    return "SE"


def main():
    mode, t1, t2 = sys.argv[1:4]
    if mode == "oofsrc":
        w1 = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
        b = pd.read_csv(W.DATA / "train/branch_references.csv"); b["TreeID"] = b.TreeID.astype(str)
        refs = {k: g.Z.values for k, g in b.groupby("TreeID")}
        g2 = load_grid("cache2", "train"); W.targets(g2, refs); g1 = load_grid("cache", "train")
        o1 = dict(np.load(f"preds/oof_{t1}.npz")); o2 = dict(np.load(f"preds/oof_{t2}.npz"))
        ens = {k: w1 * np.interp(g2[k]["zb"], g1[k]["zb"], o1[k]) + (1 - w1) * o2[k] for k in o2}
        cfg = {"w1": w1, "per_source": {}}; tp = fp = fn = 0
        for s_ in ("FI", "SE", "NO"):
            sub = {k: v for k, v in ens.items() if src_of(k) == s_}; best = None
            for thr in np.arange(0.1, 0.61, 0.05):
                for dist in (4, 5, 6, 7, 8, 10, 12):
                    r = W.evaluate(sub, g2, thr, dist)["all"]
                    if best is None or r["F1"] > best[0]["F1"]: best = (r, float(round(thr, 2)), dist)
            r, thr, dist = best; cfg["per_source"][s_] = dict(thr=thr, dist=dist, F1=r["F1"])
            for k, v in sub.items():
                z = W.peaks(v, g2[k]["zb"], thr, dist); a, b_, c, _ = W.match(z, g2[k]["Z"], 0.05); tp += a; fp += b_; fn += c
            print(f"{s_}: F1 {r['F1']:.4f} P {r['P']:.3f} R {r['R']:.3f} thr {thr} dist {dist}")
        P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1); cfg["F1"] = 2 * P * R / (P + R)
        print(f"per-source ensemble overall F1 {cfg['F1']:.4f}")
        json.dump(cfg, open(f"preds/enssrc_{t1}_{t2}.json", "w"), indent=1); return
    if mode == "testsrc":
        out = sys.argv[4]; cfg = json.load(open(sys.argv[5] if len(sys.argv) > 5 else f"preds/enssrc_{t1}_{t2}.json"))
        g2 = load_grid("cache2", "test"); g1 = load_grid("cache", "test")
        h1 = dict(np.load(f"preds/test_heat_{t1}.npz")); h2 = dict(np.load(f"preds/test_heat_{t2}.npz")); rows = []
        for k in h2:
            c = cfg["per_source"][src_of(k)]
            p = cfg["w1"] * np.interp(g2[k]["zb"], g1[k]["zb"], h1[k]) + (1 - cfg["w1"]) * h2[k]
            z = np.sort(W.peaks(p, g2[k]["zb"], c["thr"], c["dist"]))
            if len(z) == 0:                                   # every tree gets at least its most confident whorl
                z = np.array([float(g2[k]["zb"][int(np.argmax(p))])])
            rows += [(k, round(float(v), 4), j + 1) for j, v in enumerate(z)]
        pd.DataFrame(rows, columns=["TreeID", "Z", "whorl_number"]).to_csv(f"preds/whorls_{out}.csv", index=False)
        print("test trees", len(h2), "whorls", len(rows)); return
    if mode == "oof":
        w1 = float(sys.argv[4]) if len(sys.argv) > 4 else 0.5
        b = pd.read_csv(W.DATA / "train/branch_references.csv"); b["TreeID"] = b.TreeID.astype(str)
        refs = {k: g.Z.values for k, g in b.groupby("TreeID")}; src = dict(zip(b.TreeID, b.source))
        g2 = load_grid("cache2", "train"); W.targets(g2, refs); g1 = load_grid("cache", "train")
        o1 = dict(np.load(f"preds/oof_{t1}.npz")); o2 = dict(np.load(f"preds/oof_{t2}.npz"))
        ens = {k: w1 * np.interp(g2[k]["zb"], g1[k]["zb"], o1[k]) + (1 - w1) * o2[k] for k in o2}
        best = None
        for thr in np.arange(0.15, 0.56, 0.05):
            for dist in (5, 6, 7, 8, 10, 12):
                r = W.evaluate(ens, g2, thr, dist)["all"]
                if best is None or r["F1"] > best[0]["F1"]: best = (r, float(round(thr, 2)), dist)
        r, thr, dist = best; bs = W.evaluate(ens, g2, thr, dist, by=lambda t: src[t])
        print(f"ensemble w1 {w1}: F1 {r['F1']:.4f} P {r['P']:.3f} R {r['R']:.3f} thr {thr} dist {dist} | " + " ".join(f"{k} {v['F1']:.3f}" for k, v in bs.items()))
        json.dump(dict(w1=w1, thr=thr, dist=dist, F1=r["F1"], by_source={k: v["F1"] for k, v in bs.items()}), open(f"preds/ens_{t1}_{t2}.json", "w"), indent=1)
    else:
        out = sys.argv[4]; cfg = json.load(open(f"preds/ens_{t1}_{t2}.json"))
        g2 = load_grid("cache2", "test"); g1 = load_grid("cache", "test")
        h1 = dict(np.load(f"preds/test_heat_{t1}.npz")); h2 = dict(np.load(f"preds/test_heat_{t2}.npz")); rows = []
        for k in h2:
            p = cfg["w1"] * np.interp(g2[k]["zb"], g1[k]["zb"], h1[k]) + (1 - cfg["w1"]) * h2[k]
            z = np.sort(W.peaks(p, g2[k]["zb"], cfg["thr"], cfg["dist"]))
            rows += [(k, round(float(v), 4), j + 1) for j, v in enumerate(z)]
        pd.DataFrame(rows, columns=["TreeID", "Z", "whorl_number"]).to_csv(f"preds/whorls_{out}.csv", index=False)
        print("test trees", len(h2), "whorls", len(rows))


if __name__ == "__main__":
    main()
