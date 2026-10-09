"""Feature prep v2 for FOR-wood whorls: crown-axis refinement + 1-cm bins + rotation-invariant angular histograms + intensity.

usage: python src/prep2.py [train|test|all] [--workers 4]   -> cache2/<split>/<TreeID>.npz (same keys as prep.py)"""
import os, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/ST_WhorlDetection_StemEstimation"
sys.path.insert(0, str(ROOT / "src"))
BIN = 0.01
RHO_EDGES = np.array([-0.04, -0.015, 0.0, 0.015, 0.03, 0.05, 0.08, 0.12, 0.17, 0.24, 0.33, 0.45, 0.62, 0.85, 1.2, 1.8, 3.0])
BANDS = [(0.0, 0.05), (0.05, 0.15), (0.15, 0.45)]
NSECT = 12


def crown_axis(P, st, step=0.05):
    """Above the last circle fit, follow the stem by iterated median re-centring of points within 12 cm (30-cm slabs);
    below it keep the circle-fit axis. Returns smoothed x(z), y(z) on st['z']."""
    z = st["z"]; x = st["x"].copy(); y = st["y"].copy()
    ztop = st.get("z_top_fit", None)
    if ztop is None or not np.isfinite(ztop):
        return x, y
    order = np.argsort(P[:, 2]); Pz = P[order]; zs = Pz[:, 2]
    k0 = int(np.searchsorted(z, ztop)); k0 = min(max(k0, 1), len(z) - 1)
    cx, cy = x[k0 - 1], y[k0 - 1]; mx, my = [], []
    for k in range(k0, len(z)):
        a, b = np.searchsorted(zs, [z[k] - 0.15, z[k] + 0.15]); Q = Pz[a:b, :2]
        for _ in range(3):
            if len(Q) == 0:
                break
            d = np.hypot(Q[:, 0] - cx, Q[:, 1] - cy); near = Q[d < 0.12]
            if len(near) < 5:
                break
            nx, ny = np.median(near, 0)
            sh = np.hypot(nx - cx, ny - cy)
            if sh > 0.03:                               # limit drift per step
                nx, ny = cx + (nx - cx) * 0.03 / sh, cy + (ny - cy) * 0.03 / sh
            cx, cy = nx, ny
        mx.append(cx); my.append(cy)
    if mx:
        mx, my = np.array(mx), np.array(my)
        ker = np.ones(9) / 9
        if len(mx) > 9:
            mx = np.convolve(np.pad(mx, 4, mode="edge"), ker, mode="valid"); my = np.convolve(np.pad(my, 4, mode="edge"), ker, mode="valid")
        x[k0:], y[k0:] = mx, my
    return x, y


def features(P, st, intensity=None):
    z = P[:, 2]; H = st["H"]
    top = max(H, z.max()) + BIN
    nb = int(np.ceil(top / BIN)); zb = (np.arange(nb) + 0.5) * BIN
    ax_x = np.interp(z, st["z"], st["ax"]); ax_y = np.interp(z, st["z"], st["ay"]); rr = np.interp(z, st["z"], st["r"])
    dx, dy = P[:, 0] - ax_x, P[:, 1] - ax_y; d = np.hypot(dx, dy); rho = d - rr
    hb = np.clip((z / BIN).astype(int), 0, nb - 1)
    rb = np.digitize(rho, RHO_EDGES) - 1; okr = (rb >= 0) & (rb < len(RHO_EDGES) - 1)
    C = np.zeros((nb, len(RHO_EDGES) - 1)); np.add.at(C, (hb[okr], rb[okr]), 1.0)
    pop = C.sum(1) > 0
    scale = C[pop].mean(0) + 1e-3 if pop.any() else np.ones(C.shape[1])
    feats = [np.log1p(C / scale), np.log1p(C.sum(1) / (C.sum(1)[pop].mean() + 1e-3))[:, None]]
    ang = ((np.arctan2(dy, dx) + np.pi) / (2 * np.pi) * NSECT).astype(int) % NSECT
    for lo, hi in BANDS:
        m = (rho > lo) & (rho <= hi)
        S = np.zeros((nb, NSECT)); np.add.at(S, (hb[m], ang[m]), 1.0)
        tot = S.sum(1, keepdims=True)
        Ss = -np.sort(-S, axis=1) / np.maximum(tot, 1)            # rotation-invariant: sorted sector shares
        feats += [Ss, ((S > 0).sum(1) / NSECT)[:, None], np.log1p(tot / (tot[pop].mean() + 1e-3))]
    if intensity is not None:
        m = (rho > -0.02) & (rho < 0.08)
        si = np.zeros(nb); ni = np.zeros(nb); np.add.at(si, hb[m], intensity[m]); np.add.at(ni, hb[m], 1)
        mi = si / np.maximum(ni, 1); ref = np.median(mi[ni > 0]) if (ni > 0).any() else 1.0
        feats.append(np.where(ni > 0, mi / (ref + 1e-6) - 1, 0)[:, None])
    else:
        feats.append(np.zeros((nb, 1)))
    stem_r = np.interp(zb, st["z"], st["r"]); rel = np.interp(zb, st["z"], st["reliable"].astype(float))
    feats.append(np.column_stack([zb / max(H, 1.0), np.clip(H - zb, 0, 40) / 10.0, stem_r * 10, rel, np.full(nb, np.log10(max(len(P), 1)) / 7.0)]))
    return zb.astype(np.float32), np.hstack(feats).astype(np.float32)


def one(args):
    split, tid = args
    out = ROOT / "cache2" / split / f"{tid}.npz"
    if out.exists():
        return tid, "cached"
    import laspy
    from stem import fit_stem_multi as fit_stem
    t0 = time.time()
    f = laspy.read(str(DATA / split / "laz" / f"{tid}.laz")); P = np.vstack([f.x, f.y, f.z]).T
    inten = np.asarray(f.intensity, dtype=np.float64) if "intensity" in f.point_format.dimension_names else None
    if inten is not None and not np.any(inten):
        inten = None
    st = fit_stem(P)
    if not np.isfinite(st["r"]).any():
        return tid, "NO STEM"
    st["ax"], st["ay"] = crown_axis(P, st)
    zb, X = features(P, st, inten)
    np.savez_compressed(out, zb=zb, X=X, sz=st["z"], sx=st["ax"], sy=st["ay"], sr=st["r"], srel=st["reliable"], H=st["H"],
                        n_points=len(P), z_crown=st.get("z_crown", np.nan))
    return tid, f"{time.time() - t0:.1f}s bins {len(zb)} ch {X.shape[1]}"


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "all"; workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
    jobs = []
    for sp in (["train", "test"] if split == "all" else [split]):
        (ROOT / "cache2" / sp).mkdir(parents=True, exist_ok=True)
        jobs += [(sp, f[:-4]) for f in sorted(os.listdir(DATA / sp / "laz")) if f.endswith(".laz")]
    with Pool(workers) as pool:
        for i, (tid, msg) in enumerate(pool.imap_unordered(one, jobs), 1):
            print(f"[{i}/{len(jobs)}] {tid}: {msg}", flush=True)


if __name__ == "__main__":
    main()
