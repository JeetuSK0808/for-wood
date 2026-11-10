#!/usr/bin/env bash
# forwood_v8a (9 Nov 2026 rules: whorls ranked by Skill per source vs a blind comb; FOR-wood v2 labels).
# Uses the v7 CNN heatmaps (run_v5_seeds.sh + run_v7.sh first; those CNNs were trained on the v1 labels) and re-tunes the
# per-source decoding on the v2 labels: FI/SE = heatmap peaks (+ teeth above the highest peak where that helps, SE),
# NO = the blind comb (the model is no better than it there in nested CV). Stems: GBM correction refit on the v2 stem labels.
set -e
PY=.venv/bin/python; L=data_v2/ST_WhorlDetection_StemEstimation/train
OMP_NUM_THREADS=2 $PY src/eval_board.py $L/branch_references.csv v1s8_src,v2s8_src,d1es,d2es   # nested CV report (v7 vs v8a)
$PY src/final_v2.py v8a $L/branch_references.csv v1s8_src v2s8_src d1es d2es                   # -> preds/final_v8a.json, preds/whorls_v8a.csv
OMP_NUM_THREADS=2 $PY src/stem_diam.py fit --tag v2l --labels $L/stem_references.csv
$PY src/package.py v8a v2l forwood_v8a
