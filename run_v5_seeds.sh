#!/usr/bin/env bash
# 8-seed versions of the two source-aware CNNs, then per-tree-F1 tuning for FI/SE (NO stays at v3's (0.6, 4)), fit, package v5.
cd "$(dirname "$0")"
nice -n 5 .venv/bin/python -u src/whorl_cnn.py cv --epochs 100 --seeds 8 --tag v1s8_src --src-onehot > logs/cv_v1s8_src.log 2>&1
nice -n 5 .venv/bin/python -u src/whorl_cnn.py cv --epochs 100 --seeds 8 --tag v2s8_src --cache cache2 --src-onehot > logs/cv_v2s8_src.log 2>&1
nice -n 10 .venv/bin/python src/tune_macro.py v1s8_src v2s8_src 0.5 > logs/tune_macro_s8.log 2>&1
nice -n 10 .venv/bin/python src/tune_macro.py v1s3m_src v2s3m_src 0.5 > logs/tune_macro_s3.log 2>&1
nice -n 5 .venv/bin/python -u src/whorl_cnn.py fit --epochs 100 --seeds 8 --tag v1s8_src --src-onehot > logs/fit_v1s8_src.log 2>&1
nice -n 5 .venv/bin/python -u src/whorl_cnn.py fit --epochs 100 --seeds 8 --tag v2s8_src --cache cache2 --src-onehot > logs/fit_v2s8_src.log 2>&1
echo V5FITDONE >> logs/package_v5.log
