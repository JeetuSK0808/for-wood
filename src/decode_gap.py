"""Spacing-aware post-processing of the heatmap peaks (12 Nov), evaluated nested like compare_board.py.

Whorls are quasi-periodic along the stem. After the usual peaks (thr, dist) of final_v2.predict:
  1. local spacing m(z) = median gap between the peaks within +-1.5 m (global median if fewer than 3 gaps),
  2. suppress: while two neighbouring peaks are closer than rmin * m, drop the weaker one (rmin 0 = off),
  3. fill: while a gap is wider than rfill * m, add the strongest local maximum of the heatmap inside the gap (at least 0.35 m_gap from
     both ends) if its height is >= tlow (rfill 0 = off),
then the optional teeth above the highest peak (ext) as before.
For FI and SE the whole setting (thr, dist, ext, rmin, rfill, tlow) is chosen on 4 folds and scored on the 5th; the baseline is the
same nested choice with rmin = rfill = 0 (= v8b). CI: paired tree bootstrap of the per-source F1 difference, also height-weighted
(SE test trees are short; weights p_test(H)/p_train(H) as in se_robust.py).
usage: python src/decode_gap.py <labels.csv> <tags> [sources=FI,SE]
"""
import itertools, sys
import numpy as np
from scipy.signal import argrelmax
sys.path.insert(0, "src"); import score_v2 as S, eval_v2 as V, whorl_cnn as W, se_robust as SR
from ensemble import src_of

THR = (0.2, 0.25, 0.3, 0.35); DIST = (6, 8, 12, 16); EXT = {"FI": (0,), "SE": (0, 0.2), "NO": (0,)}
RMIN = (0, 0.4, 0.5, 0.6); RFILL = (0, 1.6, 1.8, 2.1); TLOW = (0.03, 0.06, 0.1)


def grid(s):
    return [g for g in itertools.product(THR, DIST, EXT[s], RMIN, RFILL, TLOW) if not (g[4] == 0 and g[5] != TLOW[0])]


def local_spacing(z, win=1.5):
    g = np.diff(z); mid = 0.5 * (z[1:] + z[:-1]); glob = np.median(g) if len(g) else 0.3
    def m(x):
        s = g[np.abs(mid - x) <= win]
        return float(np.median(s)) if len(s) >= 3 else float(glob)
    return m


def post(h, zb, z, rmin, rfill, tlow):
    z = np.sort(z)
    if len(z) < 3 or (rmin == 0 and rfill == 0):
        return z
    hz = np.interp(z, zb, h)
    if rmin > 0:
        while len(z) >= 3:
            m = local_spacing(z); g = np.diff(z); bad = [i for i in range(len(g)) if g[i] < rmin * m(0.5 * (z[i] + z[i + 1]))]
            if not bad: break
            i = min(bad, key=lambda i: g[i]); j = i if hz[i] < hz[i + 1] else i + 1
            z = np.delete(z, j); hz = np.delete(hz, j)
    if rfill > 0 and len(z) >= 3:
        mx = argrelmax(h, order=2)[0]; mx = mx[h[mx] >= tlow]; cz, ch = zb[mx], h[mx]; added = True; it = 0
        while added and it < 200:
            added = False; it += 1; m = local_spacing(z); g = np.diff(z)
            for i in np.argsort(-g):
                mm = m(0.5 * (z[i] + z[i + 1]))
                if g[i] <= rfill * mm: break
                sel = (cz > z[i] + 0.35 * mm) & (cz < z[i + 1] - 0.35 * mm)
                if sel.any():
                    k = np.argmax(np.where(sel, ch, -1)); z = np.sort(np.append(z, cz[k])); added = True; break
    return z


def predict(h, zb, top, thr, dist, ext, rmin, rfill, tlow):
    z = np.sort(W.peaks(h, zb, thr, dist))
    if len(z) == 0:
        z = np.array([float(zb[int(np.argmax(h))])])
    z = post(h, zb, z, rmin, rfill, tlow)
    if ext > 0:
        z = np.concatenate([z, S.comb(z.max() + ext, ext, top)])
    return z


def main():
    labels, tags = sys.argv[1], sys.argv[2]; srcs = (sys.argv[3] if len(sys.argv) > 3 else "FI,SE").split(",")
    refs, g2, fold, heat, top = V.setup(labels, tags)
    for s in srcs:
        ks = sorted(k for k in heat if src_of(k) == s and k in refs); fk = np.array([fold[k] for k in ks])
        w, _, _ = SR.h_weights(ks, s); one = np.ones(len(ks))
        _, ct = V.comb_table(ks, refs, top); comb, _ = V.nested_pick(ct, fk); cw = (comb * w).sum() / w.sum()
        sets = grid(s)
        tab = np.array([[S.tree_prf(predict(heat[k], g2[k]["zb"], top[k], *g), refs[k])[2] for k in ks] for g in sets])
        isb = np.array([g[3] == 0 and g[4] == 0 for g in sets])
        base, bp = V.nested_pick(np.where(isb[:, None], tab, -1), fk); new, npk = V.nested_pick(tab, fk); d = new - base
        ci, ciw = SR.boot(d, one), SR.boot(d, w)
        print(f"{s}: {len(ks)} trees | base F1 {base.mean():.4f} skill {S.skill(base.mean(), comb.mean()):+.4f} (weighted {S.skill((base * w).sum() / w.sum(), cw):+.4f}) | "
              f"gap-aware F1 {new.mean():.4f} skill {S.skill(new.mean(), comb.mean()):+.4f} (weighted {S.skill((new * w).sum() / w.sum(), cw):+.4f}) | "
              f"d F1 {d.mean():+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}] weighted [{ciw[0]:+.4f}, {ciw[1]:+.4f}]", flush=True)
        print(f"   picks {[sets[i] for i in npk]} | all-train {sets[int(np.argmax(tab.mean(1)))]}", flush=True)


if __name__ == "__main__":
    main()
