"""Feature prep v6: v2 (1-cm) features + azimuth-resolved stem-surface relief channels -> cache6/.

Below the live crown a whorl is often only a few dead stubs: 1-3 cm bumps on the bark at one or two azimuths. Histograms that
pool all azimuths per height bin (v1/v2) bury such a bump under the stem's own out-of-roundness (+-1-2 cm around a circle fit).
Here the stem surface is modelled per azimuth sector: a running median of the surface radius along z (+-20 cm) in 36 sectors,
so the residual e = rho - surface(z, sector) is ~0 on clean bark and > 1 cm on stubs and branch bases. Counts of points in
residual bands, divided by the local point density, give a relief map f(z, sector); it is smoothed 3x3 (circular in azimuth)
and pooled over azimuth into per-height channels (max, top-3 mean, mean, number of distinct bump sectors) plus coverage.
The pooled channels are appended to cache2's X (same 1-cm grid), and the full relief map is kept as R (float16) for 2-D models.
Uses the stem axis/radius already fitted by prep2 (cache2), so no stem refit.

usage: python src/prep6.py [train|test|all] [--workers 3] [--base cache --out cache7]   (2-cm grid version)
"""
import os, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/ST_WhorlDetection_StemEstimation"
BIN = 0.01
BASE, OUT = "cache2", "cache6"
NS = 36                     # azimuth sectors (10 deg)
SZ = 0.05                   # surface-model z cell
EBANDS = [(0.01, 0.03), (0.03, 0.08), (0.08, 0.30)]   # residual bands: stub, branch base, branch


def surface(zc_idx, sec, rho, nzc):
    """running-median stem-surface offset per (5-cm cell, sector) from near-surface points; NaN-safe, filled across sectors"""
    import pandas as pd
    m = np.abs(rho) < 0.05
    g = pd.DataFrame({"c": zc_idx[m], "s": sec[m], "r": rho[m]}).groupby(["c", "s"]).r.median()
    M = np.full((nzc, NS), np.nan); M[g.index.get_level_values(0), g.index.get_level_values(1)] = g.values
    S = pd.DataFrame(M).rolling(9, center=True, min_periods=2).median().values
    row = np.nanmedian(np.where(np.isfinite(S), S, np.nan), axis=1) if np.isfinite(S).any() else np.zeros(nzc)
    row = np.where(np.isfinite(row), row, 0.0)
    return np.where(np.isfinite(S), S, row[:, None])


def relief(P, d):
    from scipy.ndimage import uniform_filter, uniform_filter1d
    zb = d["zb"]; nb = len(zb); z = P[:, 2]; BIN = round(float(zb[1] - zb[0]), 3)
    ax = np.interp(z, d["sz"], d["sx"]); ay = np.interp(z, d["sz"], d["sy"]); rr = np.interp(z, d["sz"], d["sr"])
    dx, dy = P[:, 0] - ax, P[:, 1] - ay; rho = np.hypot(dx, dy) - rr
    keep = (rho > -0.08) & (rho < 0.35); z, dx, dy, rho = z[keep], dx[keep], dy[keep], rho[keep]
    sec = ((np.arctan2(dy, dx) + np.pi) / (2 * np.pi) * NS).astype(int) % NS
    nzc = int(np.ceil((zb[-1] + BIN) / SZ)) + 1
    ci = np.clip((z / SZ).astype(int), 0, nzc - 1)
    S = surface(ci, sec, rho, nzc)
    u = np.clip(z / SZ - 0.5, 0, nzc - 1); i0 = np.minimum(u.astype(int), nzc - 2); w = u - i0
    e = rho - ((1 - w) * S[i0, sec] + w * S[i0 + 1, sec])
    hb = np.clip((z / BIN).astype(int), 0, nb - 1)
    allc = np.zeros((nb, NS)); m = (e > -0.03) & (e < 0.30); np.add.at(allc, (hb[m], sec[m]), 1.0)
    dens = uniform_filter1d(allc, 2 * int(round(0.10 / BIN)) + 1, axis=0, mode="nearest")   # local expected points per cell (+-10 cm)
    R = []
    for lo, hi in EBANDS:
        c = np.zeros((nb, NS)); m = (e >= lo) & (e < hi); np.add.at(c, (hb[m], sec[m]), 1.0)
        R.append(uniform_filter(c / (dens + 1.0), size=(3, 3), mode=("nearest", "wrap")))
    R = np.stack(R, -1)                                                  # nb x NS x 3
    feats = []
    for k in range(R.shape[2]):
        f = R[:, :, k]; fs = -np.sort(-f, axis=1)
        thr = np.quantile(f[dens > 0.5], 0.95) if (dens > 0.5).any() else np.inf
        lm = (f >= np.roll(f, 1, 1)) & (f > np.roll(f, -1, 1)) & (f > thr)
        feats += [fs[:, 0], fs[:, :3].mean(1), f.mean(1), lm.sum(1).astype(float)]
    feats += [np.log1p(dens.mean(1)), (dens < 0.5).mean(1)]
    return np.column_stack(feats).astype(np.float32), R.astype(np.float16)


def one(args):
    split, tid, base, outd = args
    out = ROOT / outd / split / f"{tid}.npz"
    if out.exists():
        return tid, "cached"
    import laspy
    t0 = time.time()
    d = dict(np.load(ROOT / base / split / f"{tid}.npz"))
    f = laspy.read(str(DATA / split / "laz" / f"{tid}.laz")); P = np.vstack([f.x, f.y, f.z]).T
    F, R = relief(P, d)
    d["X"] = np.hstack([d["X"], F]).astype(np.float32); d["R"] = R
    np.savez_compressed(out, **d)
    return tid, f"{time.time() - t0:.1f}s ch {d['X'].shape[1]}"


def main():
    base, outd = (sys.argv[sys.argv.index("--base") + 1], sys.argv[sys.argv.index("--out") + 1]) if "--base" in sys.argv else (BASE, OUT)
    split = sys.argv[1] if len(sys.argv) > 1 else "all"; workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 3
    jobs = []
    for sp in (["train", "test"] if split == "all" else [split]):
        (ROOT / outd / sp).mkdir(parents=True, exist_ok=True)
        jobs += [(sp, f[:-4], base, outd) for f in sorted(os.listdir(DATA / sp / "laz")) if f.endswith(".laz")]
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    with Pool(workers) as pool:
        for i, (tid, msg) in enumerate(pool.imap_unordered(one, jobs), 1):
            print(f"[{i}/{len(jobs)}] {tid}: {msg}", flush=True)


if __name__ == "__main__":
    main()
