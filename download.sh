#!/usr/bin/env bash
# FOR-wood dataset (Zenodo 10.5281/zenodo.23012506, CC BY-SA 4.0) -> data/
set -uo pipefail; cd "$(dirname "$0")"
Z=data/ST_WhorlDetection_StemEstimation.zip
[ -f $Z ] || { curl -sSL --retry 5 -A "curl/8.7.1" -o $Z.part "https://zenodo.org/api/records/23012506/files/ST_WhorlDetection_StemEstimation.zip/content" && mv $Z.part $Z; }
unzip -q -o $Z -d data/ && echo "OK $(du -sh data | cut -f1)"; echo DONE
