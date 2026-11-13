"""Head-to-head board Skill (9 Nov rules) of heatmap ensembles, each with the same nested decoding:
  FI, SE: (thr, dist, ext) chosen on 4 folds, scored on the 5th (ext = teeth above the highest peak)
  NO:     nested choice among the plain comb, the heatmap-filtered comb and model peaks
Board Skill = mean over FI, NO, SE of (F1 - F1_comb)/(1 - F1_comb), F1_comb = nested plain comb.
CI: paired per-source tree bootstrap of (cand - base).
"hw" = the same with training trees weighted by p_test(H) / p_train(H) per source (se_robust.h_weights; label-free): the SE test trees
are much shorter than the SE training trees, and this weighting predicted the 10 Nov board for FI (0.689 vs 0.692) and SE better.
usage: python src/compare_board.py <labels.csv> <base tags> <cand tags> [<cand tags> ...] [--jobs N]
  tags may contain weighted groups ('n1:3+n1x:5'); a '@gapSE' suffix decodes SE with decode_gap.py (spacing-aware peaks)
  --stack: one more line where the spec itself is chosen per source on the 4 training folds (jointly with its setting), scored on the 5th
"""
import itertools, sys
import numpy as np
from scipy.ndimage import maximum_filter1d
sys.path.insert(0, "src"); import score_v2 as S, eval_v2 as V, final_v2 as F, eval_no as N, se_robust as SR, decode_gap as DG
from ensemble import src_of


def per_tree(labels, spec):
    spec, _, opt = spec.partition("@"); gap = opt[3:].split("+") if opt.startswith("gap") else []   # 'tags@gapSE' = decode_gap for SE
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
            tab = np.vstack([ctab, ptab]); new, picks = V.nested_pick(tab, fk)
            names = [("comb" if cset[i][2] == 0 else "fcomb") if i < len(cset) else "peaks" for i in picks]
        else:
            _, ct = V.comb_table(ks, refs, top); comb, _ = V.nested_pick(ct, fk)
            if s in gap:
                sets = DG.grid(s); fn = lambda k, g: DG.predict(heat[k], g2[k]["zb"], top[k], *g)
            else:
                sets = list(itertools.product(V.THR, V.DIST, F.EXT)); fn = lambda k, g: F.predict(heat[k], g2[k]["zb"], top[k], *g)
            tab = np.array([f1(lambda k, g=g: fn(k, g)) for g in sets]); new, picks = V.nested_pick(tab, fk)
            names = [sets[i] for i in picks]
        out[s] = (new, comb, names, ks, tab, fk)
    return out


def board(per, idx, w=None):
    if w is None:
        return np.mean([S.skill(per[s][0][idx[s]].mean(), per[s][1][idx[s]].mean()) for s in per])
    m = lambda x, s: (x[idx[s]] * w[s][idx[s]]).sum() / w[s][idx[s]].sum()
    return np.mean([S.skill(m(per[s][0], s), m(per[s][1], s)) for s in per])


def main():
    jobs = int(sys.argv[sys.argv.index("--jobs") + 1]) if "--jobs" in sys.argv else 1
    args = [a for i, a in enumerate(sys.argv[1:], 1) if a not in ("--jobs", "--stack") and sys.argv[i - 1] != "--jobs"]
    labels, specs = args[0], args[1:]; base = specs[0]
    if jobs > 1:   # per-spec work is independent; results are identical to the serial run
        from multiprocessing import Pool
        with Pool(min(jobs, len(specs))) as pool:
            res = dict(zip(specs, pool.starmap(per_tree, [(labels, sp) for sp in specs])))
    else:
        res = {sp: per_tree(labels, sp) for sp in specs}
    full = {s: np.arange(len(res[base][s][0])) for s in res[base]}; rng = np.random.default_rng(0)
    idxs = [{s: rng.integers(0, len(full[s]), len(full[s])) for s in full} for _ in range(2000)]
    hw = {s: SR.h_weights(res[base][s][3], s)[0] for s in res[base]}
    for sp in specs:
        r = res[sp]; sk = {s: S.skill(r[s][0].mean(), r[s][1].mean()) for s in r}
        skw = {s: S.skill((r[s][0] * hw[s]).sum() / hw[s].sum(), (r[s][1] * hw[s]).sum() / hw[s].sum()) for s in r}
        line = (f"{sp:44s} FI {sk['FI']:+.4f} SE {sk['SE']:+.4f} NO {sk['NO']:+.4f} | board {board(r, full):+.4f} | hw FI {skw['FI']:+.4f} SE {skw['SE']:+.4f} "
                f"NO {skw['NO']:+.4f} board {board(r, full, hw):+.4f} | NO picks {r['NO'][2]}")
        if sp != base:
            d = [board(r, i) - board(res[base], i) for i in idxs]; dw = [board(r, i, hw) - board(res[base], i, hw) for i in idxs]
            line += (f" | d {board(r, full) - board(res[base], full):+.4f} [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]"
                     f" hw d {board(r, full, hw) - board(res[base], full, hw):+.4f} [{np.percentile(dw, 2.5):+.4f}, {np.percentile(dw, 97.5):+.4f}]")
        print(line, flush=True)
    if "--stack" in sys.argv:   # the ensemble chosen per source and fold together with its decoding setting (rows of all specs stacked)
        st = {}
        for s in res[base]:
            T = np.vstack([res[sp][s][4] for sp in specs]); off = np.cumsum([0] + [len(res[sp][s][4]) for sp in specs])
            new, picks = V.nested_pick(T, res[base][s][5]); st[s] = (new, res[base][s][1], [specs[np.searchsorted(off, i, "right") - 1] for i in picks], res[base][s][3])
        sk = {s: S.skill(st[s][0].mean(), st[s][1].mean()) for s in st}
        d = [board(st, i) - board(res[base], i) for i in idxs]; dw = [board(st, i, hw) - board(res[base], i, hw) for i in idxs]
        print(f"{'stacked: spec per source, nested':44s} FI {sk['FI']:+.4f} SE {sk['SE']:+.4f} NO {sk['NO']:+.4f} | board {board(st, full):+.4f} | hw board "
              f"{board(st, full, hw):+.4f} | d {board(st, full) - board(res[base], full):+.4f} [{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]"
              f" hw d {board(st, full, hw) - board(res[base], full, hw):+.4f} [{np.percentile(dw, 2.5):+.4f}, {np.percentile(dw, 97.5):+.4f}]", flush=True)
        for s in st: print(f"   {s} picks: {st[s][2]}")


if __name__ == "__main__":
    main()
