"""Whorl detection on the 2-cm height grid with a dilated 1-D CNN (heatmap regression) + peak picking.

usage: python src/whorl_cnn.py cv [--epochs 60] [--tag v1]      5-fold grouped CV, out-of-fold F1 at +-5 cm (and 3/10/20 cm)
       python src/whorl_cnn.py fit [--epochs 60] [--tag v1]     train on all trees, predict test -> preds/whorls_<tag>.csv
"""
import argparse, json, sys, time, zlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.signal import find_peaks

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/ST_WhorlDetection_StemEstimation"
SIGMA = 0.03
CROP = 512
CACHE = "cache"
DISTS = (4, 5, 6, 7, 8, 10)


SRC_ONEHOT = False


def _src(tid):
    import re
    if re.fullmatch(r"\d+", tid): return 0
    if re.fullmatch(r"plot\d+_tree\d+", tid): return 2
    return 1


def load(split):
    trees = {}
    for f in sorted((ROOT / CACHE / split).glob("*.npz")):
        d = np.load(f); X = d["X"]
        if SRC_ONEHOT:
            oh = np.zeros((len(X), 3), np.float32); oh[:, _src(f.stem)] = 1.0; X = np.hstack([X, oh])
        trees[f.stem] = dict(zb=d["zb"], X=X, H=float(d["H"]), n=int(d["n_points"]))
    return trees


def targets(trees, refs):
    for tid, t in trees.items():
        Z = refs.get(tid, np.zeros(0))
        y = np.zeros(len(t["zb"]), np.float32)
        for z in Z:
            y = np.maximum(y, np.exp(-0.5 * ((t["zb"] - z) / SIGMA) ** 2))
        t["y"] = y.astype(np.float32); t["Z"] = np.sort(Z)


DIL = None


class Block(nn.Module):
    def __init__(self, w, d):
        super().__init__()
        self.c1 = nn.Conv1d(w, w, 3, padding=d, dilation=d); self.n1 = nn.GroupNorm(8, w)
        self.c2 = nn.Conv1d(w, w, 3, padding=d, dilation=d); self.n2 = nn.GroupNorm(8, w)

    def forward(self, x):
        h = torch.nn.functional.gelu(self.n1(self.c1(x))); h = self.n2(self.c2(h)); return torch.nn.functional.gelu(x + h)


class Net(nn.Module):
    def __init__(self, c, w=64, dil=(1, 2, 4, 8, 16, 1, 2, 4)):
        dil = DIL if DIL else dil
        super().__init__()
        self.inp = nn.Conv1d(c, w, 5, padding=2); self.blocks = nn.Sequential(*[Block(w, d) for d in dil]); self.out = nn.Conv1d(w, 1, 1)

    def forward(self, x):
        return self.out(self.blocks(self.inp(x))).squeeze(1)


def batches(trees, ids, mu, sd, rng, bs=16):
    xs, ys = [], []
    for _ in range(bs):
        t = trees[ids[rng.integers(len(ids))]]; n = len(t["zb"])
        if n > CROP:
            s = rng.integers(0, n - CROP + 1); X = t["X"][s:s + CROP]; y = t["y"][s:s + CROP]
        else:
            X = np.zeros((CROP, t["X"].shape[1]), np.float32); X[:n] = t["X"]; y = np.zeros(CROP, np.float32); y[:n] = t["y"]
        X = (X - mu) / sd
        X = X * (1 + 0.05 * rng.standard_normal((1, X.shape[1]))).astype(np.float32)      # mild per-channel gain jitter
        xs.append(X.T); ys.append(y)
    return torch.from_numpy(np.stack(xs).astype(np.float32)), torch.from_numpy(np.stack(ys).astype(np.float32))


def train(trees, ids, mu, sd, epochs, dev, seed=0, steps_per_epoch=40, width=64):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    net = Net(len(mu), w=width).to(dev); opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * steps_per_epoch)
    for ep in range(epochs):
        net.train()
        for _ in range(steps_per_epoch):
            xb, yb = batches(trees, ids, mu, sd, rng); xb, yb = xb.to(dev), yb.to(dev)
            logit = net(xb)
            w = 1 + 4 * (yb > 0.3).float()
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(logit, yb, reduction="none") * w).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    return net


@torch.no_grad()
def predict(net, t, mu, sd, dev):
    net.eval(); X = torch.from_numpy(np.ascontiguousarray(((t["X"] - mu) / sd).T[None]).astype(np.float32)).to(dev)
    return torch.sigmoid(net(X)).cpu().numpy()[0]


def peaks(p, zb, thr, dist_bins):
    k, _ = find_peaks(p, height=thr, distance=dist_bins)
    z = []
    for i in k:
        if 0 < i < len(p) - 1:
            a, b, c = p[i - 1], p[i], p[i + 1]; den = a - 2 * b + c
            off = 0.5 * (a - c) / den if abs(den) > 1e-9 else 0.0
            z.append(zb[i] + np.clip(off, -0.5, 0.5) * (zb[1] - zb[0]))
        else:
            z.append(zb[i])
    return np.array(z)


def match(pred, ref, tol):
    """Greedy one-to-one matching by increasing distance within tol; returns TP, FP, FN, abs errors."""
    if len(pred) == 0 or len(ref) == 0:
        return 0, len(pred), len(ref), []
    D = np.abs(pred[:, None] - ref[None, :]); pairs = np.argwhere(D <= tol)
    order = np.argsort(D[pairs[:, 0], pairs[:, 1]]); used_p, used_r, err = set(), set(), []
    for i in order:
        a, b = pairs[i]
        if a in used_p or b in used_r:
            continue
        used_p.add(a); used_r.add(b); err.append(D[a, b])
    tp = len(used_p); return tp, len(pred) - tp, len(ref) - tp, err


def evaluate(oof, trees, thr, dist_bins, tol=0.05, by=None):
    agg = {}
    for tid, p in oof.items():
        key = by(tid) if by else "all"
        z = peaks(p, trees[tid]["zb"], thr, dist_bins); tp, fp, fn, err = match(z, trees[tid]["Z"], tol)
        a = agg.setdefault(key, [0, 0, 0, []]); a[0] += tp; a[1] += fp; a[2] += fn; a[3] += err
    out = {}
    for k, (tp, fp, fn, err) in agg.items():
        P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1); out[k] = dict(F1=2 * P * R / max(P + R, 1e-9), P=P, R=R, mae=float(np.mean(err)) if err else np.nan)
    return out


def folds_for(ids, src, k=5):
    f = {}
    for s in sorted(set(src.values())):
        members = sorted([i for i in ids if src[i] == s], key=lambda x: zlib.crc32(x.encode()))
        for j, m in enumerate(members):
            f[m] = j % k
    return f


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=["cv", "fit"]); ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--tag", default="v1"); ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--cache", default="cache"); ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--src-onehot", action="store_true")
    a = ap.parse_args()
    global CACHE, CROP, DIL, DISTS, SRC_ONEHOT
    SRC_ONEHOT = a.src_onehot
    CACHE = a.cache
    binw = 0.01 if a.cache == "cache2" else 0.02
    CROP = int(round(10.24 / binw / 2)) * 2 if binw == 0.01 else 512
    DIL = (1, 2, 4, 8, 16, 32, 1, 2, 4, 8) if binw == 0.01 else None
    DISTS = sorted({max(2, int(round(m / binw))) for m in (0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.20)})
    dev = torch.device("mps" if torch.backends.mps.is_available() else "cpu"); torch.set_num_threads(4)
    b = pd.read_csv(DATA / "train/branch_references.csv"); b["TreeID"] = b.TreeID.astype(str)
    refs = {k: g.Z.values for k, g in b.groupby("TreeID")}; src = dict(zip(b.TreeID, b.source))
    trees = load("train"); targets(trees, refs); ids = sorted(trees)
    allX = np.concatenate([trees[i]["X"] for i in ids]); mu = allX.mean(0).astype(np.float32); sd = (allX.std(0) + 1e-6).astype(np.float32)
    (ROOT / "preds").mkdir(exist_ok=True)
    if a.mode == "cv":
        fold = folds_for(ids, src); oof = {}; t0 = time.time()
        for k in range(5):
            tr = [i for i in ids if fold[i] != k]; te = [i for i in ids if fold[i] == k]
            ps = {i: 0 for i in te}
            for s in range(a.seeds):
                net = train(trees, tr, mu, sd, a.epochs, dev, seed=s, width=a.width)
                for i in te:
                    ps[i] = ps[i] + predict(net, trees[i], mu, sd, dev) / a.seeds
            oof.update(ps); print(f"fold {k} done {time.time() - t0:.0f}s", flush=True)
        np.savez_compressed(ROOT / "preds" / f"oof_{a.tag}.npz", **oof)
        best = None
        for thr in (0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6):
            for dist in DISTS:
                r = evaluate(oof, trees, thr, dist)["all"]
                if best is None or r["F1"] > best[0]["F1"]:
                    best = (r, thr, dist)
        r, thr, dist = best
        print(f"BEST oof F1@5cm {r['F1']:.4f} (P {r['P']:.3f} R {r['R']:.3f} mae {r['mae']:.4f}) thr {thr} dist {dist}")
        for tol in (0.03, 0.05, 0.10, 0.20):
            print(f"  tol {tol:.2f}: F1 {evaluate(oof, trees, thr, dist, tol)['all']['F1']:.4f}")
        bys = evaluate(oof, trees, thr, dist, by=lambda t: src[t])
        print("  by source:", {k: round(v["F1"], 4) for k, v in bys.items()})
        json.dump(dict(F1=r["F1"], P=r["P"], R=r["R"], thr=thr, dist=dist, epochs=a.epochs, seeds=a.seeds, by_source={k: v["F1"] for k, v in bys.items()}),
                  open(ROOT / "preds" / f"cv_{a.tag}.json", "w"), indent=1)
    else:
        cfg = json.load(open(ROOT / "preds" / f"cv_{a.tag}.json")); test = load("test")
        ps = {i: 0 for i in test}
        for s in range(a.seeds):
            net = train(trees, ids, mu, sd, a.epochs, dev, seed=s, width=a.width)
            for i in test:
                ps[i] = ps[i] + predict(net, test[i], mu, sd, dev) / a.seeds
        np.savez_compressed(ROOT / "preds" / f"test_heat_{a.tag}.npz", **{k: np.asarray(v, np.float32) for k, v in ps.items()})
        rows = []
        for tid, p in ps.items():
            z = np.sort(peaks(p, test[tid]["zb"], cfg["thr"], cfg["dist"]))
            rows += [(tid, round(float(v), 4), j + 1) for j, v in enumerate(z)]
        pd.DataFrame(rows, columns=["TreeID", "Z", "whorl_number"]).to_csv(ROOT / "preds" / f"whorls_{a.tag}.csv", index=False)
        print("test trees", len(ps), "whorls", len(rows))


if __name__ == "__main__":
    main()
