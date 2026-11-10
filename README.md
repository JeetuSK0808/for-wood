# FOR-wood benchmark: whorl detection and stem profiles from single-tree point clouds

Satyajeeth (Jeetu) Suresh Kannan, Innovation Academy, Alpharetta, Georgia, USA (team BHopper, Codabench user jeetu_sk). Submission code for the FOR-wood benchmark on Codabench (competition 18295, EU SingleTree project).
The dataset (Zenodo, doi 10.5281/zenodo.23012506, CC BY-SA 4.0) is not included in this repository.

## Method

**1. Stem axis and radius (`src/stem.py`).**
- **Slab fitting:** the stem is fitted in 5-cm slabs with RANSAC circles, tracked upward and downward from a start near breast height.
- **Start search:** several start points are tried: the median of the low points, the tree-top XY, and the densest low cells. The stem with the most reliable fits is kept, which stops the tracker locking onto a neighbouring stem.
- **Sparse clouds:** slabs widen until they hold enough points.
- **Radius growth:** capped at ×1.1 per slab.
- **Smoothing:** a robust (LOWESS) smooth, then a taper model r = a(H − z)^p fitted on reliable slabs.
- **Crown:** the taper model takes over in the crown, and above 1.3 m the radius is never allowed to grow with height.

**2. Height-resolved features (`src/prep.py`, `src/prep2.py`).** Distances to the fitted axis minus the stem radius are binned per height (2 cm in v1, 1 cm in v2):
- radial histograms of the space just outside the stem
- angular occupancy and concentration of the branch zone (v2: rotation-invariant sorted sector shares in three radial bands)
- point intensity near the stem surface (v2)
- relative height, stem radius, fit reliability and point density

**3. Whorl detection (`src/whorl_cnn.py`, `src/ensemble.py`).**
- **Model:** a dilated 1-D CNN regresses a Gaussian heatmap (σ 3 cm) of whorl heights. The data source (Finland, Sweden, Norway,
  read from the TreeID pattern) is given to the network as one-hot input channels.
- **Ensemble (v7):** 8 seeds per feature version, plus a second pair of the two versions trained with dropout 0.1,
  EMA weights (0.999) and ±10% height-stretch augmentation (3 seeds each). All four heatmaps are averaged (2-cm ones
  interpolated to the 1-cm grid) in `src/final_ens.py`.
- **Peaks:** whorls are local maxima above a threshold with a minimum spacing, refined to sub-bin height. Every tree gets at least
  its most confident whorl.
- **Tuning:** threshold and spacing are tuned per data source on out-of-fold predictions (5-fold CV grouped by tree and
  stratified by source), for the official metric (F1 at ±5 cm averaged over trees). v7: Finland thr 0.3 / spacing 12, Sweden 0.35 / 4. The Norwegian clouds are very sparse
  (about 1/200 of the Finnish point density), the model is close to chance there, so only confident Norwegian whorls are kept.

**4. Stem profile (`src/stem_diam.py`).** The fitted diameter plus a gradient-boosted correction of (reference − fit). Its features are the fitted profile, relative height, the crown base, point density, and the data source, read from the TreeID pattern. Output is every 10 cm plus 0.6, 1.3 and 2.0 m.

## Update for the 9 Nov 2026 rules (v8a)

The organisers corrected the labels (FOR-wood v2, Zenodo 10.5281/zenodo.24083541; point clouds unchanged) and now rank whorl
detection by **Skill = (F1 − F1_comb) / (1 − F1_comb)** per source, averaged over Finland, Sweden and Norway. The comb is a blind,
evenly spaced guess. Predictions more than 5 cm above a tree's top label are ignored. v8a keeps the v7 CNN heatmaps (trained on the
v1 labels) and re-tunes only the decoding on the v2 labels (`src/final_v2.py`; local copy of the scorer in `src/score_v2.py`):
- **Finland:** heatmap peaks, threshold 0.25, minimum spacing 16 cm.
- **Sweden:** heatmap peaks (threshold 0.15, spacing 6 cm) plus evenly spaced guesses every 15 cm above the highest peak.
- **Norway:** the blind comb itself (from 0.8 m, every 12 cm). On the sparse Norwegian clouds our model was no better than the comb
  in nested cross-validation, and the earlier choice to predict almost nothing there now scores far below the comb.
- **Stems:** the gradient-boosted correction is refit on the v2 stem labels, and profiles run to 0.5 m above the fitted stem top.

Nested cross-validation under the new rules (v2 labels; settings chosen on 4 folds, scored on the 5th; `src/eval_board.py`):

| | Skill FI | Skill SE | Skill NO | mean |
|---|---|---|---|---|
| v7 decoding | +0.650 | +0.481 | −0.508 | +0.208 |
| v8a decoding | +0.650 | +0.552 | 0.000 | +0.401 |

Stem diameter RMSE with the v2-label correction: 0.018 m (FI 0.012, SE 0.007, NO 0.033).

## Results (out-of-fold, 189 training trees, mean per-tree F1 at ±5 cm)

| | FI | SE | NO |
|---|---|---|---|
| Whorl F1, 8-seed ensemble | 0.670 | 0.746 | ~0.01 (abstains) |
| Whorl F1, v7 ensemble (nested CV: settings re-tuned inside each fold) | 0.678 | 0.741 | abstains |
| Stem diameter RMSE (m) | 0.015 | 0.007 | 0.036 |

Public benchmark board (64 hidden test trees): whorl F1 0.506 with the 3-seed version (v3), which stays our leaderboard entry. v7 scored 0.490 there (27 Oct), so the nested-CV gain (Finland + Sweden combined: v3 0.675, v5 0.684, v7 0.692) did not carry over to the 64 test trees. v3 is this pipeline with 3 seeds per CNN.

## Reproduce

```bash
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python "laspy[lazrs]" numpy scipy pandas scikit-learn torch
bash download.sh                                   # Zenodo data -> data/ST_WhorlDetection_StemEstimation/
.venv/bin/python src/prep.py all --workers 4 && .venv/bin/python src/prep2.py all --workers 4
bash run_v5_seeds.sh                               # 8-seed CV + fit of both CNNs (source one-hot)
bash run_v7.sh                                     # dropout/EMA pair, v7 ensemble, stems, submissions/forwood_v7.zip
bash run_v8a.sh                                    # 9 Nov rules: v2 labels, per-source decoding, submissions/forwood_v8a.zip
```
Settings and nested-CV numbers of the submitted ensemble: `preds/final_v7ens.json`. The older v5 recipe is still reproducible with `src/ensemble.py` and `preds/ens_v5.json`.


Everything runs on a laptop CPU (the CNN uses Apple MPS if available). There are no external models or data beyond the benchmark.

## License
MIT. The FOR-wood data is CC BY-SA 4.0 and is not redistributed here.
