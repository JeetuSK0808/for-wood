#!/usr/bin/env bash
# forwood_v9 (13 Nov 2026): more seeds and members for the v2-label CNNs, and the whorl ensemble chosen per source.
#  - 5 more seeds (3-7) for each v8b CNN (n1x, n2x, nrm2x), averaged with v8b's 3 seeds as weighted groups ('n1:3+n1x:5')
#  - the dropout/EMA/height-stretch pair (d1v, d2v) and a 2-cm relief-map model (nrm1, cache7), all on the v2 labels
#  - optional spacing-aware peak decoding for Sweden (src/decode_gap.py)
# Per source, the ensemble and its decoding setting are chosen together on 4 folds and scored on the 5th (compare_board.py --stack;
# also height-weighted to the test trees, which are shorter in Sweden), then chosen once on all training trees for the submission
# (src/final_stack.py). Stems = v8c. Needs run_v8b.sh and run_v8c.sh first.
set -e
PY=.venv/bin/python; L=data_v2/ST_WhorlDetection_StemEstimation/train/branch_references.csv
$PY src/prep6.py all --workers 3 --base cache --out cache7          # relief map on the 2-cm grid -> cache7/
for mode in cv fit; do                                             # cv -> out-of-fold heatmaps (tuning), fit -> all training trees (test set)
  nice -n 10 $PY -u src/whorl_cnn.py $mode --epochs 100 --src-onehot --labels $L --tag n1x --seed0 3 --seeds 5
  nice -n 10 $PY -u src/whorl_cnn.py $mode --epochs 100 --src-onehot --labels $L --tag n2x --cache cache2 --seed0 3 --seeds 5
  nice -n 10 $PY -u src/whorl_cnn.py $mode --epochs 100 --src-onehot --labels $L --tag nrm2x --cache cache6 --rmap --seed0 3 --seeds 5
  nice -n 10 $PY -u src/whorl_cnn.py $mode --epochs 100 --src-onehot --labels $L --tag d1v --seeds 3 --dropout 0.1 --ema 0.999 --stretch 0.1
  nice -n 10 $PY -u src/whorl_cnn.py $mode --epochs 100 --src-onehot --labels $L --tag d2v --cache cache2 --seeds 3 --dropout 0.1 --ema 0.999 --stretch 0.1
  nice -n 10 $PY -u src/whorl_cnn.py $mode --epochs 100 --src-onehot --labels $L --tag nrm1 --cache cache7 --rmap --seeds 3
done
G8="n1:3+n1x:5,n2:3+n2x:5,nrm2:3+nrm2x:5"
SPECS="n1,n2,nrm2 n1,n2,nrm2@gapSE $G8 $G8@gapSE $G8,d1v,d2v $G8,d1v,d2v@gapSE $G8,d1v,d2v,nrm1 $G8,d1v,d2v,nrm1@gapSE"
OMP_NUM_THREADS=1 $PY src/compare_board.py $L $SPECS --jobs 4 --stack   # nested CV; the first spec = v8b/v8c whorls
$PY src/final_stack.py v9 $L $SPECS                                     # -> preds/final_v9.json, preds/whorls_v9.csv
$PY src/package.py v9 v8c forwood_v9
