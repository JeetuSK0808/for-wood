"""Out-of-fold evaluation under the 9 Nov 2026 rules (score_v2): per-source mean tree F1, Skill vs a nested comb, board-style mean.

Everything is nested over the 5 tree folds (grouped by tree, stratified by source): the comb (start, spacing) and every
strategy's settings are chosen on the other 4 folds and scored on the held-out fold, like the comb the organisers tune on train.

usage: python src/eval_v2.py <labels.csv> <tags,comma,sep>      e.g. data_v2/.../train/branch_references.csv v1s8_src,v2s8_src,d1es,d2es
"""
import sys, itertools
import numpy as np, pandas as pd
sys.path.insert(0, "src"); import whorl_cnn as W, evalcv as E, score_v2 as S
from ensemble import src_of

THR = [round(t, 2) for t in np.arange(0.05, 0.61, 0.05)]
DIST = (4, 6, 8, 10, 12, 16, 20, 30, 40)
Z0S = np.arange(0.0, 12.01, 0.1); SS = np.arange(0.10, 1.001, 0.01)


def setup(labels, tags):
    b = pd.read_csv(labels); b["TreeID"] = b.TreeID.astype(str)
    refs = {k: np.sort(g.Z.values) for k, g in b.groupby("TreeID")}
    g2 = E.grids()["g2"]; fold = E.grids()["fold"]
    heat = E.combine_many([load_group(t) for t in tags.split(",")])
    top = {k: float(g2[k]["zb"][-1]) for k in heat}
    return refs, g2, fold, heat, top


def parse_group(g):
    """'a:3+b:5' -> [('a', 3.0), ('b', 5.0)] (weights = seed counts, so the group is the mean over all seeds); 'a' -> [('a', 1.0)]"""
    return [(m.split(":")[0], float(m.split(":")[1]) if ":" in m else 1.0) for m in g.split("+")]


def load_group(g):
    """out-of-fold heatmaps of one ensemble member; a plain tag is E.load_tags(tag) unchanged"""
    mem = parse_group(g)
    if len(mem) == 1:
        return E.load_tags(mem[0][0])
    hs = [(E.load_tags(t), w) for t, w in mem]; tw = sum(w for _, w in hs)
    return {k: sum(w * h[k] for h, w in hs) / tw for k in hs[0][0]}


def comb_table(ks, refs, top):
    """F1 of every (start, spacing) comb on every tree: array [n_combos, n_trees]"""
    combos = list(itertools.product(Z0S, SS))
    return combos, np.array([[S.tree_prf(S.comb(z0, s, top[k]), refs[k])[2] for k in ks] for z0, s in combos])


def nested_pick(table, fk):
    """for each outer fold pick the row with the best mean on the other folds; returns held-out per-tree values and picks"""
    out = np.zeros(table.shape[1]); picks = []
    for f in range(5):
        tr, te = fk != f, fk == f
        i = int(np.argmax(table[:, tr].mean(1))); out[te] = table[i, te]; picks.append(i)
    return out, picks


def peaks_table(ks, heat, g2, refs, extra=None):
    """F1 of model peaks for every (thr, dist) on every tree; extra(k, z) may add predictions (e.g. comb teeth)"""
    rows, settings = [], list(itertools.product(THR, DIST))
    for thr, dist in settings:
        r = []
        for k in ks:
            z = W.peaks(heat[k], g2[k]["zb"], thr, dist)
            if len(z) == 0:
                z = np.array([float(g2[k]["zb"][int(np.argmax(heat[k]))])])
            if extra is not None:
                z = extra(k, np.sort(z))
            r.append(S.tree_prf(z, refs[k])[2])
        rows.append(r)
    return settings, np.array(rows)


def main():
    labels, tags = sys.argv[1], sys.argv[2]
    refs, g2, fold, heat, top = setup(labels, tags)
    res = {}
    for s in ("FI", "SE", "NO"):
        ks = sorted(k for k in heat if src_of(k) == s and k in refs); fk = np.array([fold[k] for k in ks])
        combos, ct = comb_table(ks, refs, top); comb_f, cp = nested_pick(ct, fk)
        sets, pt = peaks_table(ks, heat, g2, refs); model_f, mp = nested_pick(pt, fk)
        fixed = {"FI": (0.3, 12), "SE": (0.35, 4), "NO": (0.6, 4)}[s]
        v7_f = pt[sets.index(fixed)] if fixed in sets else None
        cb = combos[int(np.argmax(ct.mean(1)))]
        res[s] = dict(comb=comb_f.mean(), model=model_f.mean(), v7=v7_f.mean() if v7_f is not None else np.nan, comb_all=cb)
        print(f"{s}: trees {len(ks)} | comb F1 {comb_f.mean():.4f} (all-train pick start {cb[0]:.1f} m, spacing {cb[1]:.2f} m) | "
              f"model nested F1 {model_f.mean():.4f} (picks {[sets[i] for i in mp]}) | v7 setting {res[s]['v7']:.4f}", flush=True)
        for name in ("model", "v7"):
            print(f"    skill {name}: {S.skill(res[s][name], res[s]['comb']):+.4f}")
    for name in ("model", "v7"):
        print(f"board-style mean skill ({name}): {np.mean([S.skill(res[s][name], res[s]['comb']) for s in ('FI', 'SE', 'NO')]):+.4f}")


if __name__ == "__main__":
    main()
