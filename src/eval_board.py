"""Board-style comparison under the 9 Nov rules: mean over FI, NO, SE of Skill = (F1 - F1_comb)/(1 - F1_comb), nested over folds.

Strategies (per source), all on the same heatmaps:
  v7     the settings in the submitted v7 zip (FI 0.3/12, SE 0.35/4, NO 0.6/4 = abstain)
  new    FI/SE: (thr, dist, ext) chosen nested (ext = teeth above the highest peak); NO: the comb chosen nested
CI: per-source tree bootstrap of the difference in board Skill (comb F1 recomputed on each resample).
usage: python src/eval_board.py <labels.csv> <tags>
"""
import itertools, sys
import numpy as np
sys.path.insert(0, "src"); import score_v2 as S, eval_v2 as V, final_v2 as F
from ensemble import src_of

V7 = {"FI": (0.3, 12, 0), "SE": (0.35, 4, 0), "NO": (0.6, 4, 0)}


def main():
    labels, tags = sys.argv[1], sys.argv[2]
    refs, g2, fold, heat, top = V.setup(labels, tags); per = {}
    for s in ("FI", "SE", "NO"):
        ks = sorted(k for k in heat if src_of(k) == s and k in refs); fk = np.array([fold[k] for k in ks])
        f1 = lambda g: np.array([S.tree_prf(F.predict(heat[k], g2[k]["zb"], top[k], *g), refs[k])[2] for k in ks])
        v7 = f1(V7[s]); _, ct = V.comb_table(ks, refs, top); comb, _ = V.nested_pick(ct, fk)
        if s == "NO":
            new = comb.copy()
        else:
            sets = list(itertools.product(V.THR, V.DIST, F.EXT)); new, _ = V.nested_pick(np.array([f1(g) for g in sets]), fk)
        per[s] = (v7, new, comb)
        print(f"{s}: F1 v7 {v7.mean():.4f} new {new.mean():.4f} comb {comb.mean():.4f} | skill v7 {S.skill(v7.mean(), comb.mean()):+.4f} new {S.skill(new.mean(), comb.mean()):+.4f}", flush=True)
    board = lambda idx, j: np.mean([S.skill(per[s][j][idx[s]].mean(), per[s][2][idx[s]].mean()) for s in per])
    full = {s: np.arange(len(per[s][0])) for s in per}; rng = np.random.default_rng(0)
    d = [board(i, 1) - board(i, 0) for i in ({s: rng.integers(0, len(per[s][0]), len(per[s][0])) for s in per} for _ in range(2000))]
    print(f"board Skill v7 {board(full, 0):+.4f} -> new {board(full, 1):+.4f} | d {board(full, 1) - board(full, 0):+.4f} [95% boot {np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]")


if __name__ == "__main__":
    main()
