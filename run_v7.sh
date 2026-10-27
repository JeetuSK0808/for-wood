#!/usr/bin/env bash
# forwood_v7 (the submitted model): v5's 8-seed source-one-hot CNN pair (run_v5_seeds.sh first) plus a second pair
# trained with dropout 0.1, EMA weights (0.999) and +-10% height stretch; one averaged heatmap, per-source peak settings.
set -e
PY=.venv/bin/python
for mode in cv fit; do        # cv -> out-of-fold heatmaps (for tuning), fit -> all training trees (for the test set)
  nice -n 10 $PY -u src/whorl_cnn.py $mode --seeds 3 --src-onehot --epochs 100 --tag d1es --dropout 0.1 --ema 0.999 --stretch 0.1
  nice -n 10 $PY -u src/whorl_cnn.py $mode --seeds 3 --src-onehot --epochs 100 --tag d2es --cache cache2 --dropout 0.1 --ema 0.999 --stretch 0.1
done
$PY src/final_ens.py v7ens v1s8_src v2s8_src d1es d2es    # -> preds/final_v7ens.json, preds/whorls_v7ens.csv
OMP_NUM_THREADS=2 $PY src/stem_diam.py fit --tag v2
$PY src/package.py v7ens v2 forwood_v7
