"""Per-tree preprocessing for FOR-wood: stem axis/radius (stem.py) + height-resolved whorl features on a 2-cm grid.

usage: python src/prep.py [train|test|all] [--workers 4]
Writes cache/<split>/<TreeID>.npz with: zb (bin centres), X (bins x channels, float32), stem arrays (z, x, y, r, reliable), H, n_points."""
import os, sys, time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/ST_WhorlDetection_StemEstimation"
sys.path.insert(0, str(ROOT / "src"))
BIN = 0.02
RHO_EDGES = np.array([-0.04, -0.015, 0.0, 0.015, 0.03, 0.05, 0.08, 0.12, 0.17, 0.24, 0.33, 0.45, 0.62, 0.85, 1.2, 1.8, 3.0])
NSECT = 12


def features(P, st):
    z = P[:, 2]; H = st["H"]
    top = max(H, z.max()) + 0.02
    zb_edges = np.arange(0.0, top + BIN, BIN); nb = len(zb_edges) - 1; zb = zb_edges[:-1] + BIN / 2
    ax_x = np.interp(z, st["z"], st["x"]); ax_y = np.interp(z, st["z"], st["y"]); rr = np.interp(z, st["z"], st["r"])
    dx, dy = P[:, 0] - ax_x, P[:, 1] - ax_y; d = np.hypot(dx, dy); rho = d - rr
    hb = np.clip(((z - 0.0) / BIN).astype(int), 0, nb - 1)
    rb = np.digitize(rho, RHO_EDGES) - 1; okr = (rb >= 0) & (rb < len(RHO_EDGES) - 1)
    C = np.zeros((nb, len(RHO_EDGES) - 1), np.float64)
    np.add.at(C, (hb[okr], rb[okr]), 1.0)
    # angular occupancy of near-stem branch zone (rho 3-45 cm) and its concentration
    zone = (rho > 0.03) & (rho < 0.45)
    sect = ((np.arctan2(dy[zone], dx[zone]) + np.pi) / (2 * np.pi) * NSECT).astype(int) % NSECT
    S = np.zeros((nb, NSECT)); np.add.at(S, (hb[zone], sect), 1.0)
    occ = (S > 0).sum(1) / NSECT
    conc = S.max(1) / np.maximum(S.sum(1), 1)
    # mean height offset of branch-zone points inside each bin's 10-cm window is implicit in the CNN; add mean rho of zone points
    msum = np.zeros(nb); np.add.at(msum, hb[zone], rho[zone]); mrho = msum / np.maximum(S.sum(1), 1)
    # per-tree normalisation of counts: divide each radial channel by its mean over the populated height range, log1p
    pop = C.sum(1) > 0
    scale = C[pop].mean(0) + 1e-3 if pop.any() else np.ones(C.shape[1])
    Cn = np.log1p(C / scale)
    tot = np.log1p(C.sum(1) / (C.sum(1)[pop].mean() + 1e-3))
    stem_r = np.interp(zb, st["z"], st["r"]); rel = np.interp(zb, st["z"], st["reliable"].astype(float))
    X = np.column_stack([Cn, tot, occ, conc, mrho, zb / max(H, 1.0), np.clip(H - zb, 0, 40) / 10.0, stem_r * 10, rel,
                         np.full(nb, np.log10(max(len(P), 1)) / 7.0)]).astype(np.float32)
    return zb.astype(np.float32), X


def one(args):
    split, tid = args
    out = ROOT / "cache" / split / f"{tid}.npz"
    if out.exists():
        return tid, "cached"
    import laspy
    from stem import fit_stem_multi as fit_stem
    t0 = time.time()
    f = laspy.read(str(DATA / split / "laz" / f"{tid}.laz")); P = np.vstack([f.x, f.y, f.z]).T
    st = fit_stem(P)
    if not np.isfinite(st["r"]).any():
        return tid, "NO STEM"
    zb, X = features(P, st)
    np.savez_compressed(out, zb=zb, X=X, sz=st["z"], sx=st["x"], sy=st["y"], sr=st["r"], srel=st["reliable"], H=st["H"],
                        n_points=len(P), z_crown=st.get("z_crown", np.nan))
    return tid, f"{time.time() - t0:.1f}s bins {len(zb)}"


def main():
    split = sys.argv[1] if len(sys.argv) > 1 else "all"; workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 4
    jobs = []
    for sp in (["train", "test"] if split == "all" else [split]):
        (ROOT / "cache" / sp).mkdir(parents=True, exist_ok=True)
        jobs += [(sp, f[:-4]) for f in sorted(os.listdir(DATA / sp / "laz")) if f.endswith(".laz")]
    os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    with Pool(workers) as pool:
        for i, (tid, msg) in enumerate(pool.imap_unordered(one, jobs), 1):
            print(f"[{i}/{len(jobs)}] {tid}: {msg}", flush=True)


if __name__ == "__main__":
    main()
