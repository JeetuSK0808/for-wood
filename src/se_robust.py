"""Robustness of the per-source whorl decoding to the train -> test shift (12 Nov), on out-of-fold heatmaps.

The SE test trees are much shorter than the SE training trees (median H 12.7 m vs ~19 m; 6 of 12 under 13 m), so a setting that is
best on the pooled training trees may not transfer. For one source (default SE) this prints, for the decoding grid of final_v2
(thr x dist x ext, ext = evenly spaced teeth above the highest peak):
  - random-fold nested result (the v8b procedure), per-subgroup (TreeID prefix) and short/tall breakdowns of the picked setting,
  - leave-one-subgroup-out nested result (setting picked on the other subgroups),
  - test-height-weighted nested result: training trees weighted by p_test(H) / p_train(H) (H-bin histogram, label-free), the setting
    picked by the weighted mean F1 of the other folds, scored by the weighted mean on the held-out fold,
  - paired tree bootstrap of (candidate - v8b) for the no-teeth variant and the weighted pick.
usage: python src/se_robust.py <labels.csv> <tags> [source=SE]
"""
import itertools, re, sys
import numpy as np
sys.path.insert(0, "src"); import score_v2 as S, eval_v2 as V, final_v2 as F, stem_diam as SD
from ensemble import src_of

grp = lambda t: re.sub(r"[_\d]+$", "", t)
BINS = np.array([0, 9, 12, 15, 18, 21, 40.0])


def h_weights(ks, src):
    """p_test(H) / p_train(H) on coarse H bins, from the cached point-cloud heights (no labels)"""
    tr = SD.load("train"); te = SD.load("test")
    htr = np.array([float(tr[k]["H"]) for k in ks]); hte = np.array([float(v["H"]) for t, v in te.items() if SD.source_of(t) == src])
    ptr = np.histogram(htr, BINS)[0] / len(htr); pte = np.histogram(hte, BINS)[0] / len(hte)
    b = np.clip(np.digitize(htr, BINS) - 1, 0, len(BINS) - 2); w = pte[b] / np.maximum(ptr[b], 1e-9)
    return w / w.mean(), htr, hte


def wnested(tab, fk, w):
    out, picks = np.zeros(tab.shape[1]), []
    for f in range(5):
        tr, te = fk != f, fk == f
        i = int(np.argmax((tab[:, tr] * w[tr]).sum(1) / w[tr].sum())); out[te] = tab[i, te]; picks.append(i)
    return out, picks


def boot(d, w, n=4000, seed=0):
    rng = np.random.default_rng(seed); m = []
    for _ in range(n):
        i = rng.integers(0, len(d), len(d)); m.append((d[i] * w[i]).sum() / w[i].sum())
    return np.percentile(m, [2.5, 97.5])


def main():
    labels, tags = sys.argv[1], sys.argv[2]; src = sys.argv[3] if len(sys.argv) > 3 else "SE"
    refs, g2, fold, heat, top = V.setup(labels, tags)
    ks = sorted(k for k in heat if src_of(k) == src and k in refs); fk = np.array([fold[k] for k in ks]); gs = np.array([grp(k) for k in ks])
    w, htr, hte = h_weights(ks, src); one = np.ones(len(ks))
    print(f"{src}: {len(ks)} train trees, H median {np.median(htr):.1f} | test {len(hte)} trees, H median {np.median(hte):.1f} | weights {np.round(w, 2).tolist()}")
    _, ct = V.comb_table(ks, refs, top); comb, _ = V.nested_pick(ct, fk)
    sets = list(itertools.product(V.THR, V.DIST, F.EXT)); tab = np.array([[S.tree_prf(F.predict(heat[k], g2[k]["zb"], top[k], *g), refs[k])[2] for k in ks] for g in sets])
    sk = lambda f1, ww: S.skill((f1 * ww).sum() / ww.sum(), (comb * ww).sum() / ww.sum())
    base, bp = V.nested_pick(tab, fk); allpick = sets[int(np.argmax(tab.mean(1)))]
    print(f"random-fold nested (v8b procedure): F1 {base.mean():.4f} skill {sk(base, one):+.4f} | weighted skill {sk(base, w):+.4f} | picks {[sets[i] for i in bp]} | all-train {allpick}")
    # the same picks without teeth
    noext = np.array([tab[sets.index((sets[bp[f]][0], sets[bp[f]][1], 0)), i] for i, f in enumerate(fk)])
    ne_sets = [g for g in sets if g[2] == 0]; ne_tab = np.array([tab[sets.index(g)] for g in ne_sets]); ne_best, ne_p = V.nested_pick(ne_tab, fk)
    for name, v in (("same thr/dist, teeth off", noext), ("nested, teeth never", ne_best)):
        d = v - base; ci = boot(d, one); ciw = boot(d, w)
        print(f"  {name:26s} F1 {v.mean():.4f} skill {sk(v, one):+.4f} (d {d.mean():+.4f} F1 [{ci[0]:+.4f}, {ci[1]:+.4f}]) | weighted skill {sk(v, w):+.4f} (d [{ciw[0]:+.4f}, {ciw[1]:+.4f}])")
    print("  per subgroup / height (held-out F1: v8b-nested | teeth off | teeth never | comb):")
    for g in sorted(set(gs)) + ["H<13", "H>=13"]:
        m = (gs == g) if not g.startswith("H") else ((htr < 13) if g == "H<13" else (htr >= 13))
        print(f"    {g:7s} n {m.sum():2d}: {base[m].mean():.3f} | {noext[m].mean():.3f} | {ne_best[m].mean():.3f} | {comb[m].mean():.3f}")
    # leave-one-subgroup-out
    lo = np.zeros(len(ks)); lp = {}
    for g in sorted(set(gs)):
        i = int(np.argmax(tab[:, gs != g].mean(1))); lo[gs == g] = tab[i, gs == g]; lp[g] = sets[i]
    d = lo - base; ci = boot(d, one)
    print(f"leave-one-subgroup-out: F1 {lo.mean():.4f} skill {sk(lo, one):+.4f} (d vs v8b {d.mean():+.4f} [{ci[0]:+.4f}, {ci[1]:+.4f}]) | picks {lp}")
    # test-height-weighted pick
    wv, wp = wnested(tab, fk, w); d = wv - base; ci = boot(d, one); ciw = boot(d, w)
    print(f"height-weighted nested: F1 {wv.mean():.4f} skill {sk(wv, one):+.4f} | weighted skill {sk(wv, w):+.4f} (weighted d [{ciw[0]:+.4f}, {ciw[1]:+.4f}], unweighted d [{ci[0]:+.4f}, {ci[1]:+.4f}]) | picks {[sets[i] for i in wp]}")
    wall = sets[int(np.argmax((tab * w).sum(1) / w.sum()))]; print(f"  all-train weighted pick {wall}")
    # teeth effect per tree
    ia, ib = sets.index(allpick), sets.index((allpick[0], allpick[1], 0))
    print("  per tree (H, refs, F1 with teeth -> without):", ", ".join(f"{k} {h:.0f}m {len(refs[k])} {tab[ia, j]:.2f}->{tab[ib, j]:.2f}" for j, (k, h) in enumerate(zip(ks, htr))))


if __name__ == "__main__":
    main()
