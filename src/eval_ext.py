"""FI/SE under the 9 Nov rules: model peaks, optionally extended above the highest model peak with comb teeth (spacing s_ext),
since predictions more than 5 cm above a tree's top label are ignored. (thr, dist, s_ext) chosen nested over the tree folds.
usage: python src/eval_ext.py <labels.csv> <tags> <source FI|SE>
"""
import sys, itertools
import numpy as np
sys.path.insert(0, "src"); import whorl_cnn as W, score_v2 as S, eval_v2 as V
from ensemble import src_of

EXT = (0, 0.10, 0.15, 0.20, 0.30, 0.40)


def main():
    labels, tags, src = sys.argv[1:4]
    refs, g2, fold, heat, top = V.setup(labels, tags)
    ks = sorted(k for k in heat if src_of(k) == src and k in refs); fk = np.array([fold[k] for k in ks])
    pk = {}; rows = []; sets = list(itertools.product(V.THR, V.DIST, EXT))
    for thr, dist, e in sets:
        r = []
        for k in ks:
            if (thr, dist) not in pk:
                pass
            z = pk.setdefault((thr, dist, k), np.sort(W.peaks(heat[k], g2[k]["zb"], thr, dist)))
            if len(z) == 0:
                z = np.array([float(g2[k]["zb"][int(np.argmax(heat[k]))])])
            if e > 0:
                z = np.concatenate([z, S.comb(z.max() + e, e, top[k])])
            r.append(S.tree_prf(z, refs[k])[2])
        rows.append(r)
    tab = np.array(rows); noext = np.array([s[2] == 0 for s in sets])
    base_f, bp = V.nested_pick(np.where(noext[:, None], tab, -1), fk); ext_f, ep = V.nested_pick(tab, fk)
    combos, ct = V.comb_table(ks, refs, top); comb_f, _ = V.nested_pick(ct, fk)
    d = ext_f - base_f; rng = np.random.default_rng(0); boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    print(f"{src}: comb {comb_f.mean():.4f} | peaks {base_f.mean():.4f} (skill {S.skill(base_f.mean(), comb_f.mean()):+.4f}) picks {[sets[i] for i in bp]}")
    print(f"   peaks+ext {ext_f.mean():.4f} (skill {S.skill(ext_f.mean(), comb_f.mean()):+.4f}) picks {[sets[i] for i in ep]} | d {d.mean():+.4f} [{np.percentile(boots, 2.5):+.4f}, {np.percentile(boots, 97.5):+.4f}]")


if __name__ == "__main__":
    main()
