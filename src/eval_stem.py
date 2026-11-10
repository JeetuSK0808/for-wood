"""Stem profile CV scored the way the 9 Nov scorer does it (Participation page, 'Stem profile'):
  evaluation heights 0.6, 1.3, 2, 3, ... m (every metre) within the height range of each tree's reference profile; the submitted
  profile is linearly interpolated there and NOT extended: an evaluation height outside [lowest, highest submitted Z] scores D = 0.
  RMSE (Stem) = root mean square diameter error over all evaluation heights of all trees (pooled); Bias = mean(pred - ref).
The reference diameter at an evaluation height is linearly interpolated from the tree's reference rows (mode 'interp'); mode
'rows' scores at the reference rows' own Z instead (robustness check).

Out-of-fold profiles: the GBM correction (stem_diam.model) is trained on the reference rows of the other 4 folds (crc32 % 5, as in
stem_diam.py cv) and predicts each held-out tree's profile on the same height grid the test submission uses.
Test-mix estimate: per-source MSE weighted by the expected number of evaluation heights per source on the 64 test trees
(reference top estimated as the training median ratio top/H per source times the test tree's H).
CI: per-source (stratified) tree bootstrap, paired between methods.
usage: python src/eval_stem.py
"""
import sys, zlib
import numpy as np, pandas as pd
sys.path.insert(0, "src"); import stem_diam as SD

V1 = SD.DATA / "train/stem_references.csv"
V2 = SD.ROOT / "data_v2/ST_WhorlDetection_StemEstimation/train/stem_references.csv"
SRC = ("FI", "SE", "NO")


def iso_dec(y, w):
    """weighted non-increasing isotonic regression (pool adjacent violators)"""
    v, ww, n = [], [], []
    for yi, wi in zip(y, w):
        v.append(yi); ww.append(wi); n.append(1)
        while len(v) > 1 and v[-2] < v[-1]:
            t = ww[-2] + ww[-1]; v[-2] = (v[-2] * ww[-2] + v[-1] * ww[-1]) / t; ww[-2] = t; n[-2] += n[-1]; v.pop(); ww.pop(); n.pop()
    return np.repeat(v, n)


def read_refs(path):
    r = pd.read_csv(path).dropna(subset=["Z", "D"]); r["TreeID"] = r.TreeID.astype(str)
    return {k: (g.Z.values[np.argsort(g.Z.values)], g.D.values[np.argsort(g.Z.values)]) for k, g in r.groupby("TreeID")}, r


def grid(t, top_pad):
    H = float(t["H"])
    return np.unique(np.round(np.concatenate([[0.6, 1.3, 2.0], np.arange(0.3, H + top_pad, 0.1)]), 3))


def oof_profiles(trees, label_path, top_pad, model=SD.model, post=None):
    """{tid: (z, D)} out-of-fold over the trees that have labels in label_path"""
    _, r = read_refs(label_path); X, y, d0, g, src = SD.train_rows(trees, r)
    tids = sorted(set(g)); fold_t = {t: zlib.crc32(t.encode()) % 5 for t in tids}; fold = np.array([fold_t[t] for t in g]); out = {}
    for k in range(5):
        m = model().fit(X[fold != k], y[fold != k])
        for t in (t for t in tids if fold_t[t] == k):
            z = grid(trees[t], top_pad); F, dfit = SD.feats(trees[t], t, z); D = np.maximum(dfit + m.predict(F), 0.005)
            out[t] = (z, post(z, D) if post else D)
    return out


def eval_points(ref, prof, mode="interp"):
    """errors (pred - ref) at the scorer's evaluation heights for one tree, and the count of uncovered heights"""
    zr, dr = ref; zp, dp = prof
    if mode == "interp":
        h = np.concatenate([[0.6, 1.3], np.arange(2.0, np.floor(zr.max()) + 1e-9, 1.0)]); h = h[(h >= zr.min() - 1e-9) & (h <= zr.max() + 1e-9)]
        d_ref = np.interp(h, zr, dr)
    else:
        h, d_ref = zr, dr
    cov = (h >= zp.min() - 1e-9) & (h <= zp.max() + 1e-9)
    d_pred = np.where(cov, np.interp(h, zp, dp), 0.0)
    return d_pred - d_ref, int((~cov).sum())


def errors(refs, profs, mode="interp"):
    """{tid: error array}"""
    return {t: eval_points(refs[t], profs[t], mode)[0] for t in profs if t in refs}


def test_weights(trees_train, refs, trees_test):
    """expected number of evaluation heights per source on the test trees"""
    w = {}
    for s in SRC:
        ratio = np.median([refs[t][0].max() / float(trees_train[t]["H"]) for t in refs if t in trees_train and SD.source_of(t) == s])
        n = 0
        for t, tt in trees_test.items():
            if SD.source_of(t) != s: continue
            top = ratio * float(tt["H"]); n += 2 + max(0, int(np.floor(top)) - 1)
        w[s] = n
    return w


def summary(err, w_test=None, idx=None):
    """pooled RMSE / MAE / bias over the trees (optionally a bootstrap index per source), plus per source and test-mix RMSE"""
    per = {}
    for s in SRC:
        ks = sorted(t for t in err if SD.source_of(t) == s)
        sel = [ks[i] for i in idx[s]] if idx is not None else ks
        per[s] = np.concatenate([err[t] for t in sel]) if sel else np.zeros(0)
    e = np.concatenate([per[s] for s in SRC])
    out = dict(rmse=float(np.sqrt(np.mean(e ** 2))), mae=float(np.mean(np.abs(e))), bias=float(np.mean(e)),
               **{f"rmse_{s}": float(np.sqrt(np.mean(per[s] ** 2))) for s in SRC})
    if w_test:
        mse = {s: np.mean(per[s] ** 2) for s in SRC}; out["rmse_testmix"] = float(np.sqrt(sum(w_test[s] * mse[s] for s in SRC) / sum(w_test.values())))
    return out


def boot_idx(err, n=2000, seed=0):
    rng = np.random.default_rng(seed); cnt = {s: sum(1 for t in err if SD.source_of(t) == s) for s in SRC}
    return [{s: rng.integers(0, cnt[s], cnt[s]) for s in SRC} for _ in range(n)]


def fmt(d):
    return " ".join(f"{k} {v:.4f}" if "bias" not in k else f"{k} {v:+.4f}" for k, v in d.items())


def main():
    trees = SD.load("train"); test = SD.load("test"); refs2, _ = read_refs(V2)
    w = test_weights(trees, refs2, test); print("test eval heights per source (est.)", w)
    runs = {"v7 (1014269): v1-label GBM, grid to H": oof_profiles(trees, V1, 0.0),
            "v8a = v8b: v2-label GBM, grid to H+0.5": oof_profiles(trees, V2, 0.5)}
    errs = {k: errors(refs2, p) for k, p in runs.items()}
    for k, p in runs.items():
        unc = sum(eval_points(refs2[t], p[t])[1] for t in p if t in refs2); n = sum(len(e) for e in errs[k].values())
        print(f"{k}: {fmt(summary(errs[k], w))} | uncovered {unc}/{n}")
        print(f"   rows mode: {fmt(summary(errors(refs2, p, 'rows'), w))}")
    base = list(errs)[0]; cand = list(errs)[1]; bi = boot_idx(errs[base])
    for key in ("rmse", "rmse_testmix"):
        d = [summary(errs[cand], w, i)[key] - summary(errs[base], w, i)[key] for i in bi]
        print(f"d {key} (v8b - v7): {summary(errs[cand], w)[key] - summary(errs[base], w)[key]:+.4f} [95% boot {np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]")
    e = errs[cand]; sse = sorted(((float(np.sum(v ** 2)), t, len(v), float(np.sqrt(np.mean(v ** 2)))) for t, v in e.items()), reverse=True)
    tot = sum(x[0] for x in sse); print(f"top trees by squared error (share of total): " + ", ".join(f"{t} {s / tot:.0%} (n {n}, rmse {r:.3f})" for s, t, n, r in sse[:10]))


if __name__ == "__main__":
    main()
