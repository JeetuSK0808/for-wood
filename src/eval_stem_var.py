"""Nested CV of stem-profile variants vs v8b (scored like the 9 Nov scorer, see eval_stem.py).

Variants (all out-of-fold, folds = crc32 % 5 by tree):
  v8b       GBM correction of the fitted profile (stem_diam.model / feats), v2 labels, grid to H + 0.5 m
  honly     GBM on D directly from height-only features (z, z/H, H, z_crown/H, source): an allometric taper
  blend     D = (1 - lam_s) v8b + lam_s honly, lam_s per source chosen nested (on the other 4 folds)
  iso       non-increasing isotonic fit of the v8b profile (point-count free, unweighted), per source on/off chosen nested
  blend+iso both chosen jointly per source, nested (= v8c, src/stem_final.py)
Nested pick: for each outer fold the setting with the lowest MSE on the other folds (per source); held-out errors are kept.
CI: paired, per-source stratified tree bootstrap of (variant - v8b) in pooled RMSE and in the test-mix RMSE.
usage: python src/eval_stem_var.py
"""
import sys, zlib
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
sys.path.insert(0, "src"); import eval_stem as ES, stem_diam as SD

LAMS = np.round(np.arange(0, 1.01, 0.1), 2)


def honly_feats(t, tid, z):
    z = np.asarray(z, float); H = float(t["H"]); s = SD.source_of(tid); zc = float(t["z_crown"]) if np.isfinite(t["z_crown"]) else H
    return np.column_stack([z, z / H, np.full(len(z), H), np.full(len(z), zc / H), np.full(len(z), s == "FI"), np.full(len(z), s == "NO"),
                            np.full(len(z), s == "SE")])


def oof_honly(trees, label_path, top_pad=0.5):
    _, r = ES.read_refs(label_path); X, y, g = [], [], []
    for tid, gr in r.groupby("TreeID"):
        if tid not in trees: continue
        X.append(honly_feats(trees[tid], tid, gr.Z.values)); y.append(gr.D.values); g += [tid] * len(gr)
    X, y, g = np.vstack(X), np.concatenate(y), np.array(g); tids = sorted(set(g)); ft = {t: zlib.crc32(t.encode()) % 5 for t in tids}
    fold = np.array([ft[t] for t in g]); out = {}
    for k in range(5):
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=20, l2_regularization=1.0,
                                          random_state=0).fit(X[fold != k], y[fold != k])
        for t in (t for t in tids if ft[t] == k):
            z = ES.grid(trees[t], top_pad); out[t] = (z, np.maximum(m.predict(honly_feats(trees[t], t, z)), 0.005))
    return out


def nested(err_by_setting, per_source=True):
    """err_by_setting: {setting: {tid: err array}} -> held-out {tid: err array} and the picks per fold/source"""
    sets = list(err_by_setting); tids = sorted(err_by_setting[sets[0]]); ft = {t: zlib.crc32(t.encode()) % 5 for t in tids}
    out, picks = {}, []
    for k in range(5):
        for s in (ES.SRC if per_source else (None,)):
            tr = [t for t in tids if ft[t] != k and (s is None or SD.source_of(t) == s)]
            te = [t for t in tids if ft[t] == k and (s is None or SD.source_of(t) == s)]
            mse = {v: np.mean(np.concatenate([err_by_setting[v][t] for t in tr]) ** 2) for v in sets}
            b = min(mse, key=mse.get); picks.append((k, s, b))
            for t in te: out[t] = err_by_setting[b][t]
    return out, picks


def compare(name, err, base, w, bi):
    s = ES.summary(err, w); line = f"{name:34s} {ES.fmt(s)}"
    if err is not base:
        for key in ("rmse", "rmse_testmix"):
            d = np.array([ES.summary(err, w, i)[key] - ES.summary(base, w, i)[key] for i in bi])
            line += f" | d {key} {s[key] - ES.summary(base, w)[key]:+.4f} [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]"
    print(line, flush=True)


def main():
    trees = SD.load("train"); test = SD.load("test"); refs2, _ = ES.read_refs(ES.V2); w = ES.test_weights(trees, refs2, test)
    p8 = ES.oof_profiles(trees, ES.V2, 0.5); ph = oof_honly(trees, ES.V2)
    e8 = ES.errors(refs2, p8); bi = ES.boot_idx(e8)
    compare("v8b", e8, e8, w, bi)
    compare("honly (no point-cloud diameter)", ES.errors(refs2, ph), e8, w, bi)
    bl = {lam: ES.errors(refs2, {t: (p8[t][0], (1 - lam) * p8[t][1] + lam * np.interp(p8[t][0], *ph[t])) for t in p8}) for lam in LAMS}
    eb, pk = nested(bl); compare("blend, lam_s nested", eb, e8, w, bi)
    print("   picks", sorted({(s, b) for _, s, b in pk}))
    iso = ES.errors(refs2, {t: (p8[t][0], ES.iso_dec(p8[t][1], np.ones(len(p8[t][1])))) for t in p8})
    compare("iso (all trees)", iso, e8, w, bi)
    ei, pk = nested({"off": e8, "iso": iso}); compare("iso on/off per source, nested", ei, e8, w, bi)
    print("   picks", sorted({(s, b) for _, s, b in pk}))
    post = lambda on, D: ES.iso_dec(D, np.ones(len(D))) if on else D
    combo = {(float(lam), on): ES.errors(refs2, {t: (p8[t][0], post(on, (1 - lam) * p8[t][1] + lam * np.interp(p8[t][0], *ph[t]))) for t in p8})
             for lam in LAMS for on in (0, 1)}
    ec, pk = nested(combo); compare("blend + iso, nested jointly", ec, e8, w, bi)
    print("   picks", sorted({(s, b) for _, s, b in pk}))
    for s in ES.SRC:
        ks = [t for t in e8 if SD.source_of(t) == s]
        mse = {c: np.mean(np.concatenate([combo[c][t] for t in ks]) ** 2) for c in combo}; b = min(mse, key=mse.get)
        print(f"   all-train pick {s}: lam {b[0]} iso {b[1]} (rmse {np.sqrt(mse[b]):.4f} vs v8b {np.sqrt(mse[(0.0, 0)]):.4f})")


if __name__ == "__main__":
    main()
