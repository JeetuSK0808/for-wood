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

**v8b** retrains the CNNs on the corrected v2 labels: a 2-cm and a 1-cm model as before, plus a 1-cm model that also sees an
azimuth-resolved map of the stem surface (`src/prep6.py`, `whorl_cnn.py --rmap`). That map is the bark's deviation from a smooth
per-direction surface, so it shows the small dead-branch stubs that mark whorls below the live crown. Norway uses the comb, minus
teeth where the heatmap shows nothing within ±5 cm (`final_v2.py --no-rule auto`). Nested cross-validation (`src/compare_board.py`):

| | Skill FI | Skill SE | Skill NO | mean |
|---|---|---|---|---|
| v8a (v7 heatmaps) | +0.650 | +0.552 | −0.004 | +0.399 |
| v8b (v2-label CNNs) | +0.686 | +0.581 | +0.050 | +0.439 |

**v8c** keeps v8b's whorls (the same `branch_references.csv`) and changes only the stems. The new scorer pools the squared
diameter error over every evaluation height (0.6 m, 1.3 m, 2 m, then every metre) of every tree, and most of our error came from
the sparse Norwegian clouds, where a circle fitted to a few dozen noisy points per slab can be off by 20-50 %. v8c blends the
Norwegian profiles 60/40 with a taper model that uses only the tree's height and crown base (no point-cloud diameter), and makes
the Finnish and Norwegian profiles non-increasing with height (`src/stem_final.py`). The weight and the switch were chosen per
source on out-of-fold profiles. Nested cross-validation, scored like the board (`src/eval_stem.py`, `src/eval_stem_var.py`;
"test mix" weights each source by its expected share of evaluation heights in the 64 test trees):

| stem RMSE (m) | FI | SE | NO | all training trees | test mix |
|---|---|---|---|---|---|
| v8b | 0.0120 | 0.0076 | 0.0332 | 0.0189 | 0.0167 |
| v8c | 0.0119 | 0.0076 | 0.0247 | 0.0156 | 0.0142 |

Test-mix difference: −0.0025 m [95% tree bootstrap −0.0046, −0.0006]. A robust per-slab circle fit and a median-radius estimate
were also tried for Norway and did not help.

**v9** adds members to the v2-label ensemble and lets each source choose its own ensemble. Each v8b CNN gets 5 more seeds (seeds
3-7, averaged with its 3 v8b seeds), the dropout/EMA/height-stretch pair of v7 is retrained on the v2 labels (`d1v`, `d2v`), and a
2-cm version of the relief-map model is added (`nrm1`, `cache7`). Sweden can also be decoded with spacing-aware peaks
(`src/decode_gap.py`: drop the weaker of two peaks much closer than the local whorl spacing, fill gaps much wider than it from weaker
heatmap maxima). For each source the ensemble and its decoding setting are chosen together on 4 folds and scored on the 5th
(`src/compare_board.py --stack`), and for the submission once on all training trees (`src/final_stack.py`): Finland uses all
members, Sweden the 8-seed groups with spacing-aware peaks, Norway the 8-seed groups plus the dropout pair (filtered comb).
Nested cross-validation; "height-weighted" weights the training trees by the test trees' height mix per source (label-free;
the Swedish test trees are much shorter than the training trees):

| | Skill FI | Skill SE | Skill NO | mean | height-weighted mean |
|---|---|---|---|---|---|
| v8b / v8c (n1, n2, nrm2) | +0.686 | +0.581 | +0.050 | +0.439 | +0.426 |
| + 5 seeds per model | +0.692 | +0.590 | +0.061 | +0.448 | +0.430 |
| + dropout pair | +0.694 | +0.596 | +0.072 | +0.454 | +0.436 |
| + 2-cm relief-map model | +0.699 | +0.583 | +0.054 | +0.445 | +0.427 |
| v9: ensemble chosen per source | +0.696 | +0.593 | +0.067 | +0.452 | +0.441 |

v9 − v8b: +0.0129 [95% paired tree bootstrap +0.0025, +0.0252]; height-weighted +0.0146 [+0.0005, +0.0295]. Spacing-aware
decoding alone adds +0.0048 [+0.0002, +0.0104] to v8b's ensemble, +0.0041 to the 8-seed groups and nothing once the dropout
pair is in. Stems are v8c's.


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
bash run_v8b.sh                                    # CNNs retrained on the v2 labels (+ relief-map model), submissions/forwood_v8b.zip
bash run_v8c.sh                                    # v8b whorls + Norway taper blend and monotone stems, submissions/forwood_v8c.zip
bash run_v9.sh                                     # more seeds and members, ensemble chosen per source, submissions/forwood_v9.zip
```
Settings and nested-CV numbers of the submitted ensemble: `preds/final_v7ens.json`. The older v5 recipe is still reproducible with `src/ensemble.py` and `preds/ens_v5.json`.


Everything runs on a laptop CPU (the CNN uses Apple MPS if available). There are no external models or data beyond the benchmark.

## License
MIT. The FOR-wood data is CC BY-SA 4.0 and is not redistributed here.
