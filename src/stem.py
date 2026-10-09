"""Stem axis + radius tracking for single-tree point clouds (FOR-wood).

fit_stem(P) -> dict(z, x, y, r, ok, n): per 5-cm slab centre z, the smoothed stem-axis centre (x, y), the smoothed stem radius r,
whether the slab had an accepted circle fit, and the number of points used. RANSAC circle fits in slabs, tracked upward and
downward from a robust start near breast height, adaptive slab thickness for sparse (UAV) clouds, then robust smoothing."""
import numpy as np

STEP = 0.05


def circle_lsq(Q):
    """Kasa algebraic fit, then 5 Gauss-Newton steps on geometric distance."""
    x, y = Q[:, 0], Q[:, 1]
    A = np.column_stack([x, y, np.ones_like(x)]); bvec = x * x + y * y
    try:
        c = np.linalg.lstsq(A, bvec, rcond=None)[0]
    except np.linalg.LinAlgError:
        return None
    cx, cy = c[0] / 2, c[1] / 2; r = np.sqrt(max(c[2] + cx * cx + cy * cy, 1e-12))
    for _ in range(5):
        dx, dy = x - cx, y - cy; d = np.hypot(dx, dy) + 1e-12
        J = np.column_stack([-dx / d, -dy / d, -np.ones_like(d)]); res = d - r
        try:
            step = np.linalg.lstsq(J, -res, rcond=None)[0]
        except np.linalg.LinAlgError:
            break
        cx, cy, r = cx + step[0], cy + step[1], r + step[2]
        if np.abs(step).max() < 1e-5:
            break
    return cx, cy, abs(r)


def ransac_circle(Q, rmin, rmax, tol, rng, iters=300, center_hint=None, max_shift=None):
    n = len(Q)
    if n < 6:
        return None
    i = rng.integers(0, n, size=(iters, 3))
    p1, p2, p3 = Q[i[:, 0]], Q[i[:, 1]], Q[i[:, 2]]
    ax, ay, bx, by, cx_, cy_ = p1[:, 0], p1[:, 1], p2[:, 0], p2[:, 1], p3[:, 0], p3[:, 1]
    d = 2 * (ax * (by - cy_) + bx * (cy_ - ay) + cx_ * (ay - by))
    ok = np.abs(d) > 1e-9
    a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx_ * cx_ + cy_ * cy_
    ux = np.where(ok, (a2 * (by - cy_) + b2 * (cy_ - ay) + c2 * (ay - by)) / np.where(ok, d, 1), np.nan)
    uy = np.where(ok, (a2 * (cx_ - bx) + b2 * (ax - cx_) + c2 * (bx - ax)) / np.where(ok, d, 1), np.nan)
    rr = np.hypot(ax - ux, ay - uy)
    good = ok & (rr >= rmin) & (rr <= rmax)
    if center_hint is not None and max_shift is not None:
        good &= np.hypot(ux - center_hint[0], uy - center_hint[1]) <= max_shift
    if not good.any():
        return None
    ux, uy, rr = ux[good], uy[good], rr[good]
    # inlier counts for all candidates (chunked to bound memory)
    best, best_n = None, -1
    for s in range(0, len(ux), 64):
        D = np.abs(np.hypot(Q[None, :, 0] - ux[s:s + 64, None], Q[None, :, 1] - uy[s:s + 64, None]) - rr[s:s + 64, None])
        cnt = (D < tol).sum(1); k = int(cnt.argmax())
        if cnt[k] > best_n:
            best_n, best = int(cnt[k]), (ux[s + k], uy[s + k], rr[s + k])
    inl = np.abs(np.hypot(Q[:, 0] - best[0], Q[:, 1] - best[1]) - best[2]) < tol
    if inl.sum() < 6:
        return None
    f = circle_lsq(Q[inl])
    if f is None or not (rmin <= f[2] <= rmax):
        return None
    inl = np.abs(np.hypot(Q[:, 0] - f[0], Q[:, 1] - f[1]) - f[2]) < tol
    # angular coverage of the inliers (a stem seen from one side only gives a poorly constrained radius)
    ang = np.arctan2(Q[inl, 1] - f[1], Q[inl, 0] - f[0]); cov = len(np.unique(((ang + np.pi) / (2 * np.pi) * 12).astype(int) % 12)) / 12
    return dict(x=f[0], y=f[1], r=f[2], inliers=int(inl.sum()), n=n, frac=float(inl.mean()), cov=cov)


def _start(P, rng, hint=None):
    """Robust first stem fit between 0.8 and 3 m above the lowest points. The hint is the XY median of the low points (a segmented
    single tree's base); slabs widen until they hold enough points (sparse UAV clouds); circles must sit near the hint."""
    best = None
    zmin = np.percentile(P[:, 2], 0.5)
    low = P[(P[:, 2] > zmin + 0.3) & (P[:, 2] < zmin + 1.8), :2]
    if hint is None:
        hint = np.median(low, 0) if len(low) >= 10 else np.median(P[:, :2], 0)
    for z0 in np.arange(zmin + 0.8, zmin + 3.0, 0.2):
        for half in (0.1, 0.2, 0.3):
            m = np.abs(P[:, 2] - z0) < half
            Q = P[m, :2]; Q = Q[np.hypot(Q[:, 0] - hint[0], Q[:, 1] - hint[1]) < 0.6]
            if len(Q) >= 40:
                break
        if len(Q) < 15:
            continue
        if len(Q) > 20000:
            Q = Q[rng.choice(len(Q), 20000, replace=False)]
        sparse = len(Q) < 400
        f = ransac_circle(Q, 0.025, 0.45, 0.025 if sparse else 0.012, rng, iters=500, center_hint=hint, max_shift=0.45)
        if f and (best is None or f["inliers"] * f["cov"] > best["inliers"] * best["cov"]):
            best = dict(f, z=z0, sparse=sparse)
    return best


def candidate_hints(P, k=3):
    """Start hints: low-point XY median, treetop XY median, and the densest XY cells of the 0.5-2.5 m slab."""
    zmin = np.percentile(P[:, 2], 0.5); H = np.percentile(P[:, 2], 99.95)
    low = P[(P[:, 2] > zmin + 0.3) & (P[:, 2] < zmin + 1.8), :2]
    hints = []
    if len(low) >= 10: hints.append(np.median(low, 0))
    top = P[P[:, 2] > H - 1.0, :2]
    if len(top) >= 5: hints.append(np.median(top, 0))
    sl = P[(P[:, 2] > zmin + 0.5) & (P[:, 2] < zmin + 2.5), :2]
    if len(sl) >= 50:
        lo = sl.min(0); idx = ((sl - lo) / 0.1).astype(int); nx, ny = idx.max(0) + 1
        Hh = np.zeros((nx, ny)); np.add.at(Hh, (idx[:, 0], idx[:, 1]), 1)
        from scipy.ndimage import maximum_filter, uniform_filter
        Hs = uniform_filter(Hh, 3); pk = (Hs == maximum_filter(Hs, 7)) & (Hs > 0)
        cand = np.argwhere(pk); vals = Hs[pk]
        for j in np.argsort(-vals)[:k]:
            hints.append(lo + (cand[j] + 0.5) * 0.1)
    out = []
    for h in hints:
        if all(np.hypot(*(h - o)) > 0.25 for o in out): out.append(h)
    return out


def fit_stem_multi(P, seed=0):
    """Run fit_stem from each candidate hint; keep the stem with the most reliable 5-cm circle fits."""
    best, best_s = None, -1
    for h in candidate_hints(P):
        st = fit_stem(P, seed=seed, hint=h)
        sc = int(st["reliable"].sum()) if "reliable" in st else 0
        if sc > best_s: best, best_s = st, sc
    return best if best is not None else fit_stem(P, seed=seed)


def fit_stem(P, seed=0, zmax=None, hint=None):
    rng = np.random.default_rng(seed)
    top = P[:, 2].max() if zmax is None else zmax
    zs = np.arange(max(P[:, 2].min(), 0.0) + 0.1, top, STEP)
    order = np.argsort(P[:, 2]); Pz = P[order]; zsort = Pz[:, 2]
    st = _start(P, rng, hint)
    out = {z: None for z in zs}
    if st is None:
        return _finish(zs, out, H=float(np.percentile(P[:, 2], 99.95)))

    def slab(zc, half):
        a, b = np.searchsorted(zsort, [zc - half, zc + half]); return Pz[a:b, :2]

    def track(z_list, cx, cy, r):
        miss = 0
        for z in z_list:
            half = 0.05
            Q = slab(z, half)
            Q = Q[np.hypot(Q[:, 0] - cx, Q[:, 1] - cy) < r + 0.12] if len(Q) else Q
            while len(Q) < 25 and half < 0.3:          # sparse clouds: widen the slab
                half += 0.05; Q = slab(z, half); Q = Q[np.hypot(Q[:, 0] - cx, Q[:, 1] - cy) < r + 0.12] if len(Q) else Q
            if len(Q) > 8000:
                Q = Q[rng.choice(len(Q), 8000, replace=False)]
            tol = max(0.006, 0.08 * r) if len(Q) >= 400 else max(0.02, 0.12 * r)
            f = ransac_circle(Q, max(0.012, 0.6 * r), 1.1 * r + 0.005, tol, rng, iters=250, center_hint=(cx, cy), max_shift=0.03 + 0.4 * r)
            if f is not None and f["frac"] > 0.15 and f["inliers"] >= 8:
                out[z] = dict(f, half=half); cx, cy = f["x"], f["y"]; r = 0.7 * r + 0.3 * f["r"]; miss = 0
            else:
                miss += 1
                if miss > 60:          # 3 m without a fit: stop tracking (crown top)
                    break

    up = [z for z in zs if z >= st["z"]]; down = [z for z in zs[::-1] if z < st["z"]]
    track(up, st["x"], st["y"], st["r"]); track(down, st["x"], st["y"], st["r"])
    return _finish(zs, out, H=float(np.percentile(P[:, 2], 99.95)))


def _robust_smooth(z, v, w, frac_m=0.6, it=3):
    """Tricube-weighted local linear fit with robustness iterations (LOWESS in metres)."""
    zs = z.copy(); est = np.interp(zs, z, v)
    rw = np.ones_like(v)
    for _ in range(it):
        out = np.empty_like(z)
        for i, z0 in enumerate(z):
            d = np.abs(z - z0) / frac_m; k = (d < 1)
            ww = w[k] * rw[k] * (1 - d[k] ** 3) ** 3
            if ww.sum() < 1e-9:
                out[i] = est[i]; continue
            A = np.column_stack([np.ones(k.sum()), z[k] - z0])
            try:
                beta = np.linalg.lstsq(A * ww[:, None] ** 0.5, v[k] * ww ** 0.5, rcond=None)[0]; out[i] = beta[0]
            except np.linalg.LinAlgError:
                out[i] = est[i]
        res = v - out; s = np.median(np.abs(res)) + 1e-9
        u = res / (6 * s); rw = np.where(np.abs(u) < 1, (1 - u ** 2) ** 2, 0.0)
        est = out
    return est


def _taper_fit(z, r, w, H):
    """Weighted fit r = a * (H - z)^p on reliable slabs (p clipped to [0.4, 1.4]); returns a callable."""
    k = (H - z) > 0.3
    if k.sum() < 5:
        return None
    lx, ly = np.log(H - z[k]), np.log(r[k]); ww = w[k]
    A = np.column_stack([np.ones(k.sum()), lx])
    beta = np.linalg.lstsq(A * ww[:, None] ** 0.5, ly * ww ** 0.5, rcond=None)[0]
    pexp = float(np.clip(beta[1], 0.4, 1.4)); la = float(np.average(ly - pexp * lx, weights=ww))
    return lambda zz: np.exp(la) * np.clip(H - np.asarray(zz), 0.0, None) ** pexp


def _finish(zs, out, H=None):
    ok = np.array([out[z] is not None for z in zs])
    res = dict(z=zs, ok=ok, x=np.full(len(zs), np.nan), y=np.full(len(zs), np.nan), r=np.full(len(zs), np.nan),
               cov=np.zeros(len(zs)), inl=np.zeros(len(zs)), r_raw=np.full(len(zs), np.nan), reliable=np.zeros(len(zs), bool))
    if ok.sum() < 3:
        return res
    H = float(zs.max()) if H is None else H
    zf = zs[ok]; X = np.array([out[z]["x"] for z in zf]); Y = np.array([out[z]["y"] for z in zf])
    R = np.array([out[z]["r"] for z in zf]); C = np.array([out[z]["cov"] for z in zf]); I = np.array([out[z]["inliers"] for z in zf])
    F = np.array([out[z]["frac"] for z in zf])
    w = np.clip(C, 0.1, 1) * np.sqrt(np.clip(I, 1, None))
    xs, ys = _robust_smooth(zf, X, w, 0.8), _robust_smooth(zf, Y, w, 0.8)
    rs = _robust_smooth(zf, R, w * np.clip(C, 0.1, 1), 1.2)
    # reliable slabs: good angular coverage and inlier fraction, and consistent with the smoothed profile
    rel = (C >= 0.4) & (F >= 0.25) & (np.abs(R - rs) < 0.15 * rs + 0.004)
    # the crown starts where reliable fits stop being dense: last height with >= 50% reliable slabs in the 1-m window below
    ztop = zf.min()
    for zz in zf:
        win = (zf > zz - 1.0) & (zf <= zz)
        if win.sum() >= 5 and rel[win].mean() >= 0.5:
            ztop = zz
    sel = rel & (zf >= min(1.0, ztop - 1.0)) & (zf <= ztop)
    tap = _taper_fit(zf[sel], R[sel], w[sel], H) if sel.sum() >= 5 else None
    r_final = np.interp(zs, zf, rs)
    if tap is not None:
        model = tap(zs)
        below = zs <= ztop
        # below the crown: smoothed fits, kept within 15% of the taper model; above: taper model
        r_final = np.where(below, np.clip(r_final, 0.85 * model, 1.15 * model), model)
        # butt swell is real near the ground: trust the smoothed fits below 1.3 m
        r_final = np.where(zs < 1.3, np.interp(zs, zf, rs), r_final)
    # never let the radius grow with height above 1.3 m (running minimum with a 3-mm allowance)
    up = zs >= 1.3
    if up.any():
        idx = np.where(up)[0]; cur = r_final[idx[0]]
        for j in idx:
            cur = min(cur + 0.003, r_final[j]) if np.isfinite(r_final[j]) else cur
            r_final[j] = min(r_final[j], cur) if np.isfinite(r_final[j]) else cur
    res["r"] = np.maximum(r_final, 0.005)
    lo, hi = zf.min(), zf.max(); inside = (zs >= lo) & (zs <= hi)
    res["x"][inside] = np.interp(zs[inside], zf, xs); res["y"][inside] = np.interp(zs[inside], zf, ys)
    res["x"][zs > hi], res["y"][zs > hi] = xs[-1], ys[-1]
    res["x"][zs < lo], res["y"][zs < lo] = xs[0], ys[0]
    res["r_raw"][ok] = R; res["cov"][ok] = C; res["inl"][ok] = I
    res["reliable"][np.where(ok)[0][rel]] = True
    res["z_top_fit"] = hi; res["z_bot_fit"] = lo; res["z_crown"] = float(ztop); res["H"] = H
    res["taper_ok"] = tap is not None
    return res
