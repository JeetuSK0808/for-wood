#!/usr/bin/env bash
# forwood_v8b (9 Nov 2026 rules): the CNNs retrained on the corrected FOR-wood v2 labels (2-cm, 1-cm and 1-cm relief-map models,
# 3 seeds each), decoded per source as in v8a; Norway = the blind comb filtered by the heatmap (teeth kept only where the heatmap
# max within +-5 cm reaches tau), which beat the plain comb in nested CV. Needs prep.py/prep2.py caches (and run_v7.sh for the
# comparison against v8a's heatmaps).
set -e
PY=.venv/bin/python; L=data_v2/ST_WhorlDetection_StemEstimation/train
$PY src/prep6.py all --workers 3                     # azimuth-resolved stem-relief features -> cache6/
for mode in cv fit; do                               # cv -> out-of-fold heatmaps (tuning), fit -> all training trees (test set)
  nice -n 10 $PY -u src/whorl_cnn.py $mode --epochs 100 --seeds 3 --src-onehot --labels $L/branch_references.csv --tag n1
  nice -n 10 $PY -u src/whorl_cnn.py $mode --epochs 100 --seeds 3 --src-onehot --labels $L/branch_references.csv --tag n2 --cache cache2
  nice -n 10 $PY -u src/whorl_cnn.py $mode --epochs 100 --seeds 3 --src-onehot --labels $L/branch_references.csv --tag nrm2 --cache cache6 --rmap
done
OMP_NUM_THREADS=2 $PY src/compare_board.py $L/branch_references.csv v1s8_src,v2s8_src,d1es,d2es n1,n2,nrm2   # nested CV: v8a vs v8b
$PY src/final_v2.py v8b $L/branch_references.csv n1 n2 nrm2 --no-rule auto      # -> preds/final_v8b.json, preds/whorls_v8b.csv
OMP_NUM_THREADS=2 $PY src/stem_diam.py fit --tag v2l --labels $L/stem_references.csv
$PY src/package.py v8b v2l forwood_v8b
