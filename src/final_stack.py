"""v9 (13 Nov 2026): the whorl ensemble chosen per source = the final-model version of compare_board.py --stack.

compare_board.py --stack chooses, per source and outer fold, the (ensemble spec, decoding setting) pair with the best mean tree F1
on the other 4 folds, and scores it on the 5th. Here the same choice is made once on all training trees: every spec is run through
final_v2.py (its setting chosen on the spec's out-of-fold heatmaps; NO with --no-rule auto), and per source the spec whose chosen
setting has the highest all-train mean tree F1 is kept. Those sources' rows are merged line by line (text level, so the output is
byte-identical to the per-spec files) -> preds/whorls_<out>.csv, preds/final_<out>.json.
usage: python src/final_stack.py <out_tag> <labels.csv> <spec1> <spec2> ...
  spec = comma-separated tags as in compare_board.py (weighted groups 'a:3+b:5'; '@gapSE' = final_v2.py --gap SE)
"""
import json, subprocess, sys
sys.path.insert(0, "src"); from ensemble import src_of

SRC = ("FI", "SE", "NO")


def main():
    out, labels, specs = sys.argv[1], sys.argv[2], sys.argv[3:]; cfgs, lines = [], []
    for i, sp in enumerate(specs):
        tags, _, opt = sp.partition("@"); gap = opt[3:].replace("+", ",") if opt.startswith("gap") else ""
        cmd = [sys.executable, "src/final_v2.py", f"{out}_s{i}", labels, *tags.split(","), "--no-rule", "auto"] + (["--gap", gap] if gap else [])
        print(f"[{i}] {sp}", flush=True); subprocess.run(cmd, check=True)
        cfgs.append(json.load(open(f"preds/final_{out}_s{i}.json"))); lines.append(open(f"preds/whorls_{out}_s{i}.csv").read().splitlines())
    f1 = lambda c: c["oof_tree_F1"] if "oof_tree_F1" in c else c["train_tree_F1"]
    table = {s: [round(float(f1(c["per_source"][s])), 5) for c in cfgs] for s in SRC}
    pick = {s: max(range(len(specs)), key=lambda i: table[s][i]) for s in SRC}       # ties -> the earlier spec
    head = lines[0][0]; assert all(L[0] == head for L in lines)
    rows = {s: [ln for ln in lines[pick[s]][1:] if src_of(ln.split(",")[0]) == s] for s in SRC}
    ids = sorted({ln.split(",")[0] for L in lines for ln in L[1:]})
    assert all(sorted({ln.split(",")[0] for ln in L[1:]}) == ids for L in lines)
    by_id = {}
    for s in SRC:
        for ln in rows[s]: by_id.setdefault(ln.split(",")[0], []).append(ln)
    merged = [head] + [ln for k in sorted(by_id, key=ids.index) for ln in by_id[k]]
    open(f"preds/whorls_{out}.csv", "w").write("\n".join(merged) + "\n")
    cfg = {"specs": specs, "labels": labels, "all_train_F1": table,
           "per_source": {s: dict(spec=specs[pick[s]], **cfgs[pick[s]]["per_source"][s]) for s in SRC}}
    json.dump(cfg, open(f"preds/final_{out}.json", "w"), indent=1)
    print(json.dumps(cfg["per_source"]), "| test whorls", len(merged) - 1, {s: len(rows[s]) for s in SRC})


if __name__ == "__main__":
    main()
