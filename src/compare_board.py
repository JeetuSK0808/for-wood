"""Head-to-head board Skill (9 Nov rules) of heatmap ensembles, each with the same nested decoding:
  FI, SE: (thr, dist, ext) chosen on 4 folds, scored on the 5th (ext = teeth above the highest peak)
  NO:     nested choice among the plain comb, the heatmap-filtered comb and model peaks
Board Skill = mean over FI, NO, SE of (F1 - F1_comb)/(1 - F1_comb), F1_comb = nested plain comb.
CI: paired per-source tree bootstrap of (cand - base).
usage: python src/compare_board.py <labels.csv> <base tags> <cand tags> [<cand tags> ...]
"""
import itertools, sys
import numpy as np
from scipy.ndimage import maximum_filter1d
sys.path.insert(0, "src"); import score_v2 as S, eval_v2 as V, final_v2 as F, eval_no as N
from ensemble import src_of


def per_tree(labels, spec):
    refs, g2, fold, heat, top = V.setup(labels, spec); out = {}
    for s in ("FI", "SE", "NO"):
        ks = sorted(k for k in heat if src_of(k) == s and k in refs); fk = np.array([fold[k] for k in ks])
        f1 = lambda fn: np.array([S.tree_prf(fn(k), refs[k])[2] for k in ks])
        if s == "NO":
            hm = {k: maximum_filter1d(heat[k], 11) for k in ks}
            cset = list(itertools.product(N.Z0S, N.SS, N.TAUS))
            ctab = np.array([f1(lambda k, c=c: N.fcomb(hm[k], g2[k]["zb"], c[0], c[1], top[k], c[2])) for c in cset])
            plain = np.array([c[2] == 0 for c in cset])
            comb, _ = V.nested_pick(np.where(plain[:, None], ctab, -1), fk)
            psets = list(itertools.product(V.THR, V.DIST))
            ptab = np.array([f1(lambda k, g=g: F.predict(heat[k], g2[k]["zb"], top[k], g[0], g[1], 0)) for g in psets])
            new, picks = V.nested_pick(np.vstack([ctab, ptab]), fk)
            names = [("comb" if cset[i][2] == 0 else "fcomb") if i < len(cset) else "peaks" for i in picks]
        else:
            _, ct = V.comb_table(ks, refs, top); comb, _ = V.nested_pick(ct, fk)
            sets = list(itertools.product(V.THR, V.DIST, F.EXT))
            new, picks = V.nested_pick(np.array([f1(lambda k, g=g: F.predict(heat[k], g2[k]["zb"], top[k], *g)) for g in sets]), fk)
            names = [sets[i] for i in picks]
        out[s] = (new, comb, names)
    return out


def board(per, idx):
    return np.mean([S.skill(per[s][0][idx[s]].mean(), per[s][1][idx[s]].mean()) for s in per])


def main():
    labels, specs = sys.argv[1], sys.argv[2:]; res = {sp: per_tree(labels, sp) for sp in specs}; base = specs[0]
    full = {s: np.arange(len(res[base][s][0])) for s in res[base]}; rng = np.random.default_rng(0)
    idxs = [{s: rng.integers(0, len(full[s]), len(full[s])) for s in full} for _ in range(2000)]
    for sp in specs:
        r = res[sp]; sk = {s: S.skill(r[s][0].mean(), r[s][1].mean()) for s in r}
        line = f"{sp:44s} FI {sk['FI']:+.4f} SE {sk['SE']:+.4f} NO {sk['NO']:+.4f} | board {board(r, full):+.4f} | NO picks {r['NO'][2]}"
        if sp != base:
            d = [board(r, i) - board(res[base], i) for i in idxs]
            line += f" | d {board(r, full) - board(res[base], full):+.4f} [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]"
        print(line, flush=True)


if __name__ == "__main__":
    main()
