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
ARCH = "cnn"          # cnn | gru (BiGRU context layer after the conv blocks)
EMA = 0.0             # >0: evaluate an exponential moving average of the weights
STRETCH = 0.0         # >0: random height-stretch augmentation by a factor in [1-STRETCH, 1+STRETCH]
EXTRA = False         # extra position channels: height relative to the crown base and to the tree top
DROP = 0.0            # channel dropout inside each residual block
WD = 1e-4             # AdamW weight decay
RMAP = False          # append the flattened relief map R (bins x 36 sectors x 3 bands, cache6/cache7) and learn a 2-D front-end on it
NS_R, NB_R = 36, 3


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
        if EXTRA:
            zb = d["zb"]; H = max(float(d["H"]), 1.0); zc = float(d["z_crown"]) if np.isfinite(d["z_crown"]) else H
            X = np.hstack([X, np.column_stack([(zb - zc) / H, (zb - zc) / 5.0, np.clip((H - zb) / H, 0, 1)]).astype(np.float32)])
        if RMAP:
            X = np.hstack([X, np.log1p(d["R"].astype(np.float32) * 10).reshape(len(X), NS_R * NB_R)])
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
        self.drop = nn.Dropout1d(DROP) if DROP > 0 else nn.Identity()

    def forward(self, x):
        h = self.drop(torch.nn.functional.gelu(self.n1(self.c1(x)))); h = self.n2(self.c2(h)); return torch.nn.functional.gelu(x + h)


class CircConv2d(nn.Module):
    """conv over (height, azimuth): zero padding along height, circular padding along azimuth"""
    def __init__(self, ci, co, kz, ka, dz=1):
        super().__init__()
        self.pa = ka // 2; self.c = nn.Conv2d(ci, co, (kz, ka), padding=(dz * (kz // 2), 0), dilation=(dz, 1)); self.n = nn.GroupNorm(4, co)

    def forward(self, x):
        if self.pa:
            x = torch.cat([x[..., -self.pa:], x, x[..., :self.pa]], -1)
        return torch.nn.functional.gelu(self.n(self.c(x)))


class Net(nn.Module):
    def __init__(self, c, w=64, dil=(1, 2, 4, 8, 16, 1, 2, 4)):
        dil = DIL if DIL else dil
        super().__init__()
        self.c1 = c - NS_R * NB_R if RMAP else c; self.r2d = None
        if RMAP:
            self.r2d = nn.Sequential(CircConv2d(NB_R, 16, 5, 3), CircConv2d(16, 16, 5, 3, 2), CircConv2d(16, 16, 5, 3, 4))
            c = self.c1 + 32
        self.inp = nn.Conv1d(c, w, 5, padding=2); self.blocks = nn.Sequential(*[Block(w, d) for d in dil]); self.out = nn.Conv1d(w, 1, 1)
        self.gru = nn.GRU(w, w // 2, batch_first=True, bidirectional=True) if ARCH == "gru" else None
        if self.gru is not None:
            self.gn = nn.GroupNorm(8, w)

    def forward(self, x):
        if self.r2d is not None:
            B, _, L = x.shape; r = x[:, self.c1:].reshape(B, NS_R, NB_R, L).permute(0, 2, 3, 1)
            h2 = self.r2d(r.contiguous()); x = torch.cat([x[:, :self.c1], h2.amax(3), h2.mean(3)], 1)
        h = self.blocks(self.inp(x))
        if self.gru is not None:
            g, _ = self.gru(h.transpose(1, 2)); h = torch.nn.functional.gelu(h + self.gn(g.transpose(1, 2)))
        return self.out(h).squeeze(1)


def batches(trees, ids, mu, sd, rng, bs=16):
    xs, ys = [], []
    for _ in range(bs):
        t = trees[ids[rng.integers(len(ids))]]; n = len(t["zb"])
        if STRETCH > 0:
            f = 1 + STRETCH * (2 * rng.random() - 1); binw = float(t["zb"][1] - t["zb"][0])
            span = CROP / f; s0 = rng.uniform(0, max(n - 1 - span, 0)) if n - 1 > span else 0.0
            u = s0 + np.arange(CROP) / f; valid = u <= n - 1; u = np.minimum(u, n - 1)
            i0 = np.minimum(u.astype(int), n - 2); w_ = (u - i0)[:, None].astype(np.float32)
            X = (t["X"][i0] * (1 - w_) + t["X"][i0 + 1] * w_) * valid[:, None]
            zc = (t["Z"] - float(t["zb"][0])) / binw; jz = (zc - s0) * f
            jj = np.arange(CROP)[:, None]
            y = (np.exp(-0.5 * ((jj - jz[None, :]) * binw / SIGMA) ** 2).max(1) if len(jz) else np.zeros(CROP)) * valid
            X = (X - mu) / sd * valid[:, None]
            X = X * (1 + 0.05 * rng.standard_normal((1, X.shape[1]))).astype(np.float32)
            xs.append(X.T); ys.append(y.astype(np.float32)); continue
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
    net = Net(len(mu), w=width).to(dev); opt = torch.optim.AdamW(net.parameters(), lr=2e-3, weight_decay=WD)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=2e-3, total_steps=epochs * steps_per_epoch)
    ema = [q.detach().clone() for q in net.parameters()] if EMA > 0 else None; step = 0
    for ep in range(epochs):
        net.train()
        for _ in range(steps_per_epoch):
            xb, yb = batches(trees, ids, mu, sd, rng); xb, yb = xb.to(dev), yb.to(dev)
            logit = net(xb)
            w = 1 + 4 * (yb > 0.3).float()
            loss = (torch.nn.functional.binary_cross_entropy_with_logits(logit, yb, reduction="none") * w).mean()
            opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            step += 1
            if ema is not None:
                dec = min(EMA, (1 + step) / (10 + step))
                with torch.no_grad():
                    for e_, q in zip(ema, net.parameters()):
                        e_.mul_(dec).add_(q.detach(), alpha=1 - dec)
    if ema is not None:
        with torch.no_grad():
            for e_, q in zip(ema, net.parameters()):
                q.copy_(e_)
    return net


TTA = ()              # height-stretch factors for test-time augmentation, e.g. (0.94, 1.0, 1.06)


@torch.no_grad()
def predict(net, t, mu, sd, dev):
    net.eval()
    if TTA:
        n = len(t["zb"]); out = np.zeros(n)
        for f in TTA:
            m = int(round((n - 1) * f)) + 1; u = np.arange(m) / f; i0 = np.minimum(u.astype(int), n - 2); w_ = (u - i0)[:, None]
            Xs = t["X"][i0] * (1 - w_) + t["X"][i0 + 1] * w_
            X = torch.from_numpy(np.ascontiguousarray(((Xs - mu) / sd).T[None]).astype(np.float32)).to(dev)
            ps = torch.sigmoid(net(X)).cpu().numpy()[0]
            out += np.interp(np.arange(n) * f, np.arange(m), ps) / len(TTA)
        return out.astype(np.float32)
    X = torch.from_numpy(np.ascontiguousarray(((t["X"] - mu) / sd).T[None]).astype(np.float32)).to(dev)
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
    ap.add_argument("--src-onehot", action="store_true"); ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--arch", default="cnn"); ap.add_argument("--ema", type=float, default=0.0); ap.add_argument("--stretch", type=float, default=0.0)
    ap.add_argument("--sigma", type=float, default=0.03); ap.add_argument("--extra", action="store_true")
    ap.add_argument("--dropout", type=float, default=0.0); ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--tta", default=""); ap.add_argument("--rmap", action="store_true")
    a = ap.parse_args()
    global CACHE, CROP, DIL, DISTS, SRC_ONEHOT, ARCH, EMA, STRETCH, SIGMA, EXTRA, DROP, WD, TTA, RMAP
    RMAP = a.rmap
    TTA = tuple(float(x) for x in a.tta.split(",")) if a.tta else ()
    SRC_ONEHOT = a.src_onehot; ARCH = a.arch; EMA = a.ema; STRETCH = a.stretch; SIGMA = a.sigma; EXTRA = a.extra
    DROP = a.dropout; WD = a.wd
    CACHE = a.cache
    _zb = np.load(next((ROOT / a.cache / "train").glob("*.npz")))["zb"]; binw = round(float(_zb[1] - _zb[0]), 3)
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
        fold = folds_for(ids, src); oof = {}; t0 = time.time(); per_seed = {}; plain = {}
        for k in range(5):
            tr = [i for i in ids if fold[i] != k]; te = [i for i in ids if fold[i] == k]
            ps = {i: 0 for i in te}
            for s in range(a.seed0, a.seed0 + a.seeds):
                net = train(trees, tr, mu, sd, a.epochs, dev, seed=s, width=a.width)
                for i in te:
                    q = predict(net, trees[i], mu, sd, dev); ps[i] = ps[i] + q / a.seeds; per_seed.setdefault(s, {})[i] = q.astype(np.float32)
                    if TTA:     # also keep the plain (no-TTA) prediction of the same model under <tag>_plain
                        global_tta = TTA; globals()["TTA"] = ()
                        q0 = predict(net, trees[i], mu, sd, dev); globals()["TTA"] = global_tta
                        plain.setdefault(s, {})[i] = q0.astype(np.float32)
            oof.update(ps); print(f"fold {k} done {time.time() - t0:.0f}s", flush=True)
        np.savez_compressed(ROOT / "preds" / f"oof_{a.tag}.npz", **oof)
        for s, d_ in per_seed.items():
            np.savez_compressed(ROOT / "preds" / f"oof_{a.tag}_s{s}.npz", **d_)
        for s, d_ in plain.items():
            np.savez_compressed(ROOT / "preds" / f"oof_{a.tag}plain_s{s}.npz", **d_)
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
        cj = ROOT / "preds" / f"cv_{a.tag}.json"
        cfg = json.load(open(cj)) if cj.exists() else dict(thr=0.3, dist=8); test = load("test")
        ps = {i: 0 for i in test}
        for s in range(a.seed0, a.seed0 + a.seeds):
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
