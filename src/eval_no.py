"""Strategies for a weak source (Norway) under the 9 Nov rules, all nested over the 5 tree folds:
  comb      the organisers' blind baseline (start, spacing) tuned on the training folds  -> Skill 0 by definition
  fcomb     the same dense comb, but teeth are dropped where the model heatmap (max within +-5 cm) is below tau
  model     model peaks (thr, dist)
usage: python src/eval_no.py <labels.csv> <tags> [source=NO]
"""
import sys, itertools
import numpy as np
from scipy.ndimage import maximum_filter1d
sys.path.insert(0, "src"); import score_v2 as S, eval_v2 as V
from ensemble import src_of

Z0S = np.arange(0.0, 8.01, 0.2); SS = (0.10, 0.11, 0.12, 0.13, 0.15, 0.20, 0.25, 0.30)
TAUS = (0.0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.16, 0.2, 0.25, 0.3)


def fcomb(heat_max, zb, z0, s, top, tau):
    t = S.comb(z0, s, top)
    if tau <= 0 or len(t) == 0:
        return t
    i = np.clip(np.round((t - zb[0]) / (zb[1] - zb[0])).astype(int), 0, len(zb) - 1)
    return t[heat_max[i] >= tau]


def main():
    labels, tags = sys.argv[1], sys.argv[2]; src = sys.argv[3] if len(sys.argv) > 3 else "NO"
    refs, g2, fold, heat, top = V.setup(labels, tags)
    ks = sorted(k for k in heat if src_of(k) == src and k in refs); fk = np.array([fold[k] for k in ks])
    hm = {k: maximum_filter1d(heat[k], 11) for k in ks}
    combos = list(itertools.product(Z0S, SS, TAUS))
    tab = np.array([[S.tree_prf(fcomb(hm[k], g2[k]["zb"], z0, s, top[k], tau), refs[k])[2] for k in ks] for z0, s, tau in combos])
    plain = np.array([c[2] == 0 for c in combos])
    comb_f, cp = V.nested_pick(np.where(plain[:, None], tab, -1), fk)
    fc_f, fp = V.nested_pick(tab, fk)
    sets, pt = V.peaks_table(ks, heat, g2, refs); mod_f, mp = V.nested_pick(pt, fk)
    best_all = combos[int(np.argmax(tab.mean(1)))]
    print(f"{src}: {len(ks)} trees | comb F1 {comb_f.mean():.4f} picks {[tuple(round(x, 2) for x in combos[i]) for i in cp]}")
    print(f"   filtered comb F1 {fc_f.mean():.4f} picks {[tuple(round(x, 2) for x in combos[i]) for i in fp]} | skill {S.skill(fc_f.mean(), comb_f.mean()):+.4f}")
    print(f"   model peaks F1 {mod_f.mean():.4f} | skill {S.skill(mod_f.mean(), comb_f.mean()):+.4f} | all-train best filtered comb {best_all}")
    d = fc_f - comb_f; rng = np.random.default_rng(0)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(2000)]
    print(f"   filtered - comb per-tree F1: {d.mean():+.4f} [95% boot {np.percentile(boots, 2.5):+.4f}, {np.percentile(boots, 97.5):+.4f}]")


if __name__ == "__main__":
    main()
