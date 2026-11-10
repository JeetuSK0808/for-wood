#!/usr/bin/env bash
# FOR-wood dataset (CC BY-SA 4.0). v1: Zenodo 10.5281/zenodo.23012506 -> data/ (point clouds + v1 labels).
# v2 (9 Nov 2026, corrected labels; point clouds unchanged): Zenodo 10.5281/zenodo.24083541 -> data_v2/
set -uo pipefail; cd "$(dirname "$0")"
Z=data/ST_WhorlDetection_StemEstimation.zip
[ -f $Z ] || { curl -sSL --retry 5 -A "curl/8.7.1" -o $Z.part "https://zenodo.org/api/records/23012506/files/ST_WhorlDetection_StemEstimation.zip/content" && mv $Z.part $Z; }
unzip -q -o $Z -d data/ && echo "OK v1 $(du -sh data | cut -f1)"
mkdir -p data_v2; Z2=data_v2/ST_WhorlDetection_StemEstimation_v2.zip
[ -f $Z2 ] || { curl -sSL --retry 5 -A "curl/8.7.1" -o $Z2.part "https://zenodo.org/api/records/24083541/files/ST_WhorlDetection_StemEstimation_v2.zip/content" && mv $Z2.part $Z2; }
unzip -q -o $Z2 "*/train/branch_references.csv" "*/train/stem_references.csv" -d data_v2/ && echo "OK v2 labels"; echo DONE
