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
- **Ensemble:** 8 seeds per feature version, and the two versions' heatmaps averaged.
- **Peaks:** whorls are local maxima above a threshold with a minimum spacing, refined to sub-bin height. Every tree gets at least
  its most confident whorl.
- **Tuning:** threshold and spacing are tuned per data source on out-of-fold predictions (5-fold CV grouped by tree and
  stratified by source), for the official metric (F1 at ±5 cm averaged over trees). The Norwegian clouds are very sparse
  (about 1/200 of the Finnish point density), the model is close to chance there, so only confident Norwegian whorls are kept.

**4. Stem profile (`src/stem_diam.py`).** The fitted diameter plus a gradient-boosted correction of (reference − fit). Its features are the fitted profile, relative height, the crown base, point density, and the data source, read from the TreeID pattern. Output is every 10 cm plus 0.6, 1.3 and 2.0 m.

## Results (out-of-fold, 189 training trees, mean per-tree F1 at ±5 cm)

| | FI | SE | NO |
|---|---|---|---|
| Whorl F1, 8-seed ensemble | 0.670 | 0.746 | ~0.01 (abstains) |
| Stem diameter RMSE (m) | 0.015 | 0.007 | 0.036 |

Public benchmark board (64 hidden test trees): whorl F1 0.506 with the 3-seed version (v3).

## Reproduce

```bash
uv venv .venv --python 3.12 && uv pip install --python .venv/bin/python "laspy[lazrs]" numpy scipy pandas scikit-learn torch
bash download.sh                                   # Zenodo data -> data/ST_WhorlDetection_StemEstimation/
.venv/bin/python src/prep.py all --workers 4 && .venv/bin/python src/prep2.py all --workers 4
bash run_v5_seeds.sh                               # 8-seed CV + fit of both CNNs (source one-hot)
.venv/bin/python src/ensemble.py testsrc v1s8_src v2s8_src ens_v5 preds/ens_v5.json
OMP_NUM_THREADS=2 .venv/bin/python src/stem_diam.py fit --tag v2
.venv/bin/python src/package.py ens_v5 v2 forwood_v5
```
`preds/ens_v5.json`: `{"w1": 0.5, "per_source": {"FI": {"thr": 0.2, "dist": 8}, "SE": {"thr": 0.3, "dist": 5}, "NO": {"thr": 0.6, "dist": 4}}}`


Everything runs on a laptop CPU (the CNN uses Apple MPS if available). There are no external models or data beyond the benchmark.

## License
MIT. The FOR-wood data is CC BY-SA 4.0 and is not redistributed here.
