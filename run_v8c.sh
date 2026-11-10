#!/usr/bin/env bash
# forwood_v8c (10 Nov 2026): v8b's whorls unchanged; only the stems change. v8b's GBM correction is blended per source with a
# height-only taper GBM (Norway: 0.6 / 0.4) and fitted non-increasing with height (Finland, Norway). Weight and switch were chosen
# per source on out-of-fold profiles scored like the 9 Nov board (src/eval_stem.py). Needs run_v8b.sh first (preds/whorls_v8b.csv).
set -e
PY=.venv/bin/python
OMP_NUM_THREADS=4 $PY src/eval_stem.py            # v7 vs v8b stems scored like the board (out-of-fold, test-mix estimate)
OMP_NUM_THREADS=4 $PY src/eval_stem_var.py        # blend / isotonic / both, chosen nested; paired bootstrap vs v8b
OMP_NUM_THREADS=4 $PY src/stem_final.py v8c       # -> preds/stem_v8c.csv, preds/stem_v8c.json
$PY src/package.py v8b v8c forwood_v8c
