"""Final stem profiles for the test trees (v8c): the v8b GBM correction (stem_diam) blended per source with a height-only taper GBM
(eval_stem_var.honly_feats: z, z/H, H, z_crown/H, source; no point-cloud diameter), then optionally a non-increasing isotonic fit.
The blend weight lam_s and the isotonic switch are picked per source on all training trees' out-of-fold profiles (lowest MSE at
the scorer's evaluation heights), like the whorl settings. Both GBMs are then refit on all v2-label rows.
lam = 0 and iso off for every source reproduces preds/stem_v2l.csv (the v8b stems).
usage: python src/stem_final.py <tag> [--force-off]   -> preds/stem_<tag>.csv + preds/stem_<tag>.json
"""
import json, sys
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.insert(0, "src"); import eval_stem as ES, eval_stem_var as EV, stem_diam as SD


def main():
    tag = sys.argv[1]; trees = SD.load("train"); test = SD.load("test"); refs2, r = ES.read_refs(ES.V2)
    post = lambda on, D: ES.iso_dec(D, np.ones(len(D))) if on else D
    cfg = {}
    if "--force-off" in sys.argv:
        cfg = {s: dict(lam=0.0, iso=0) for s in ES.SRC}
    else:
        p8 = ES.oof_profiles(trees, ES.V2, 0.5); ph = EV.oof_honly(trees, ES.V2)
        for s in ES.SRC:
            ks = [t for t in p8 if t in refs2 and SD.source_of(t) == s]; mse = {}
            for lam in EV.LAMS:
                for on in (0, 1):
                    e = [ES.eval_points(refs2[t], (p8[t][0], post(on, (1 - lam) * p8[t][1] + lam * np.interp(p8[t][0], *ph[t]))))[0] for t in ks]
                    mse[(float(lam), on)] = float(np.mean(np.concatenate(e) ** 2))
            b = min(mse, key=mse.get); cfg[s] = dict(lam=b[0], iso=b[1], oof_rmse=round(np.sqrt(mse[b]), 5), oof_rmse_v8b=round(np.sqrt(mse[(0.0, 0)]), 5))
    X, y, d0, g, src = SD.train_rows(trees, r); m8 = SD.model().fit(X, y)
    Xh, yh = [], []
    for tid, gr in r.groupby("TreeID"):
        if tid in trees: Xh.append(EV.honly_feats(trees[tid], tid, gr.Z.values)); yh.append(gr.D.values)
    mh = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0,
                                       random_state=0).fit(np.vstack(Xh), np.concatenate(yh))
    rows = []
    for tid, t in test.items():
        z = ES.grid(t, 0.5); F, dfit = SD.feats(t, tid, z); D8 = np.maximum(dfit + m8.predict(F), 0.005)
        c = cfg[SD.source_of(tid)]; D = D8
        if c["lam"] > 0:
            D = (1 - c["lam"]) * D8 + c["lam"] * np.maximum(mh.predict(EV.honly_feats(t, tid, z)), 0.005)
        D = post(c["iso"], D)
        rows += [(tid, float(a), round(float(b), 5)) for a, b in zip(z, D)]
    pd.DataFrame(rows, columns=["TreeID", "Z", "D"]).to_csv(SD.ROOT / "preds" / f"stem_{tag}.csv", index=False)
    json.dump(cfg, open(SD.ROOT / "preds" / f"stem_{tag}.json", "w"), indent=1); print(json.dumps(cfg), "| test rows", len(rows))


if __name__ == "__main__":
    main()
