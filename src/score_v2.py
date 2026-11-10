"""Local copy of the 9 Nov 2026 FOR-wood whorl scorer (Skill), plus the blind comb baseline it is measured against.

Rules (Codabench page, update 2026-11-09):
  - per tree, predictions are matched one-to-one to reference whorls within +-5 cm: the largest number of pairs, and among
    those the smallest total height error (here: Hungarian assignment with a large cost for pairs beyond 5 cm)
  - predictions more than 5 cm above the tree's highest reference whorl are ignored; there is no lower limit
  - per-tree precision / recall / F1; a missing tree counts as F1 0
  - Skill per source = (F1 - F1_comb) / (1 - F1_comb), F1 = mean tree F1 over the source's trees; board Skill = mean of FI, NO, SE
  - the comb ignores the point cloud: whorls evenly spaced from a fixed start height to the top of the tree, spacing and start
    tuned per source on the training data
"""
import numpy as np
from scipy.optimize import linear_sum_assignment

TOL = 0.05


def match(pred, ref, tol=TOL):
    """max-cardinality, then min-total-error one-to-one matching; returns (tp, errors)"""
    if len(pred) == 0 or len(ref) == 0:
        return 0, []
    D = np.abs(np.asarray(pred, float)[:, None] - np.asarray(ref, float)[None, :])
    C = np.where(D <= tol, D, 1e3)
    r, c = linear_sum_assignment(C); ok = D[r, c] <= tol
    return int(ok.sum()), list(D[r, c][ok])


def tp_count(pred, ref, tol=TOL):
    """maximum number of one-to-one pairs within tol; on a line the two-pointer sweep over sorted heights is optimal
    (same count as match(), which additionally minimises the total error)"""
    i = j = tp = 0
    while i < len(pred) and j < len(ref):
        if pred[i] < ref[j] - tol:
            i += 1
        elif ref[j] < pred[i] - tol:
            j += 1
        else:
            tp += 1; i += 1; j += 1
    return tp


def tree_prf(pred, ref, tol=TOL):
    """precision, recall, F1 of one tree under the new rules (predictions > top label + tol are ignored)"""
    ref = np.sort(np.asarray(ref, float)); pred = np.sort(np.asarray(pred, float))
    if len(ref):
        pred = pred[pred <= ref[-1] + tol]
    tp = tp_count(pred, ref, tol)
    P = tp / len(pred) if len(pred) else 0.0; R = tp / len(ref) if len(ref) else 0.0
    return P, R, (2 * P * R / (P + R) if P + R > 0 else 0.0)


def comb(z0, s, top):
    """the blind baseline: evenly spaced heights from z0 to the tree top"""
    return np.arange(z0, top + 1e-9, s) if top >= z0 else np.zeros(0)


def fit_comb(trees, z0s=np.arange(0.0, 12.01, 0.1), ss=np.arange(0.10, 1.001, 0.01)):
    """trees: list of (refs, top). Grid-search the (start, spacing) that maximises mean tree F1; returns (z0, s, F1)"""
    best = (0.0, 0.5, -1.0)
    for s in ss:
        for z0 in z0s:
            f = np.mean([tree_prf(comb(z0, s, top), ref)[2] for ref, top in trees])
            if f > best[2]:
                best = (float(z0), float(s), float(f))
    return best


def skill(f1, f1_comb):
    return (f1 - f1_comb) / (1 - f1_comb)
