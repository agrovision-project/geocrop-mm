# geocrop-mm

Code, inputs specification and per-seed results for the temporal out-of-distribution experiment
(Tables 9 to 12) and Figure 1 of

> C. Paris, M. F. Celik, S. Maurogiovanni, G. Tseng, V. Marsocci, R. Cartuyvels, G. Cavallaro,
> R. Sedona. *Multimodal Crop Type Classification with Geospatial Foundation Models and Geo-tagged
> Field Photos.* Submitted to the International Journal of Applied Earth Observation and
> Geoinformation.

## Scope

Reproduced here: every cell of Tables 9 to 12 (models trained on LUCAS 2018, tested on LUCAS 2022)
except the OlmoEarth cells, and Figure 1.

Not in this repository: Tables 2 to 8 (in-domain and few-shot experiments on 2018) and the
OlmoEarth cells of Tables 9 to 12, including the OlmoEarth+CLIP column of Table 12. These were
produced with separate pipelines by co-authors and external partners. `run_ood.py` also computes
OlmoEarth rows from the OlmoEarth file listed below, because that file is part of the common point
pool; those rows are not the published values.

## Contents

| path | content |
|---|---|
| `run_ood.py` | experiment and aggregation into Tables 9 to 12 |
| `make_fig1.py` | Figure 1 |
| `crop_classes.py` | LUCAS LC1 codes to the 17 crop classes |
| `slurm/run_ood.sbatch` | SLURM array job, one seed per GPU |
| `data/SHA256SUMS` | checksums of every input file |
| `data/clip_photos_{2018,2022}.csv.gz` | the photo behind every CLIP embedding: `point_id`, `url` |
| `results/ood/results_seed{17..21}_{overall,perclass}.csv` | per-seed results of the published run |
| `requirements.txt` | package versions of the published run |

## Input data

All inputs sit in one directory, `GEOCROP_DATA`. Arrays are indexed by the 8-digit LUCAS
`POINT_ID`. Before running, check the inputs against the published run:
`cd "$GEOCROP_DATA" && sha256sum -c <repository>/data/SHA256SUMS`.

| file (`{y}` = 2018, 2022) | content | keys and shape (N for 2018 / 2022) |
|---|---|---|
| `{y}/LUCAS_{y}_LULC.csv` | LUCAS survey points; used columns `POINT_ID`, `Country`, `LC1`, and `LAT_USED`, `LON_USED` for Figure 1 | 55,220 / 49,833 points of the 17 crop classes |
| `{y}/AE_{y}_Xy_v2.npz` | AlphaEarth annual embedding | `X` (N,1,64,3,3) float32, `point_ids`; 123,483 / 115,844 |
| `{y}/PE_{y}_X.npz` | Presto embedding | `embeddings` (N,128,3,3) float32, `point_ids`; 123,483 / 115,744 |
| `{y}/TE_{y}_data.npz` | Tessera annual embedding | `X` (N,1,128,3,3) float32, `point_ids`; 107,642 / 103,841 |
| `{y}/OE_{y}_AI2_v2.npz` | OlmoEarth embedding, int8-quantized, provided by external partners | `data` (N,768,3,3) int8, `point_ids`; 105,738 / 103,468 |
| `{y}/CLIP_{y}_point.npz` | image embedding of the LUCAS Point photo | `data` (N,1024) float32, `point_id`, `url`; 55,081 / 49,282 |
| `{y}/modalities/S1_{y}_Xy_v2.npz` | Sentinel-1 monthly composites, VV and VH | `X` (N,12,2,3,3) float32, `point_ids`, `fill_value`; 120,032 / 113,214 |
| `{y}/modalities/S2_{y}_Xy_v2.npz` | Sentinel-2 L2A monthly composites, B2 B3 B4 B5 B6 B7 B8 B8A B11 B12 | `X` (N,12,10,3,3) float32; 123,483 / 115,744 |
| `{y}/modalities/ERA5_{y}_Xy_v2.npz` | ERA5-Land monthly, 18 variables | `X` (N,12,18,1,1) float32; 123,483 / 115,844 |
| `{y}/modalities/DEM_{y}_Xy_v2.npz` | Copernicus DEM GLO-30 height, slope, aspect | `X` (N,1,3,1,1) float32; 123,483 / 115,844 |
| `{y}/modalities/SOIL_{y}_Xy_v2.npz` | AI4SoilHealth soil properties, 32 features | `X` (N,1,32,1,1) float32; 107,642 / 103,841 |
| `ne_50m_admin_0_countries.zip` | Natural Earth 1:50m Admin 0 countries (public domain), Figure 1 only, from https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip | |

The satellite, climate, terrain and soil inputs cover a 3 x 3 pixel window centred on each LUCAS
point and were generated following Celik et al. (2026b) from the data archive of Celik et al.
(2026a, doi:10.5281/zenodo.17988209); Section 2 of the paper describes each source. Each CLIP
embedding encodes one photo, the LUCAS Point photo of that location as served by Eurostat GISCO; the
two photo lists in `data/` give its URL for every embedding (2018 files end in `P.jpg`, 2022 files
in `LCLU_Point.jpg`).

## Method, as implemented in `run_ood.py`

- **Labels.** LUCAS `LC1` codes are mapped to 17 crop classes (`crop_classes.py`); other codes are
  discarded. The point pool of each year is the set of labelled points present in all five
  embedding files: 54,059 in 2018, 49,041 in 2022.
- **Split, per seed.** 2018 points are stratified by class and country; strata with fewer than 5
  points are dropped; 70 % train, the remaining 30 % halved into validation and an unused 2018
  test part (stratified by stratum, or by class if a stratum has fewer than 2 points). This gives
  37,752 training and 8,090 validation points. The test set is the full 2022 pool, 49,041 points.
  Seeds: 17, 18, 19, 20, 21.
- **Embeddings.** The 3 x 3 window is averaged. OlmoEarth values x are dequantized as
  sign(x) (|x| / 127.5)^2. Non-finite values and values at or below -9990 are imputed with the
  training median; features are then standardized with training statistics.
- **Classifiers.** Linear probe: one linear layer, Adam, learning rate 1e-3, batch 256,
  class-weighted cross-entropy (weights n / (K n_c), normalized to mean 1), learning rate halved
  after 10 epochs without validation-loss improvement (minimum 1e-6), early stopping after 20
  epochs without an improvement larger than 1e-4, at most 200 epochs, weights of the best
  validation loss. Random forest: 100 trees,
  `class_weight='balanced_subsample'`, `max_features='sqrt'`, `random_state` = seed.
- **Fusion.** Each modality is standardized separately, then Naive: concatenation; PCA: 32
  principal components of the concatenation, fitted on training data; Proj: the concatenation is
  mapped to 32 dimensions by a linear layer trained jointly with a linear classification head
  (same recipe as the linear probe), and the head is discarded. The fused features are
  standardized again before the classifier.
- **Temporal Transformers.** Inputs: 12 monthly steps of S1 (2 channels) and S2 (10 channels),
  window-averaged with fill values excluded; ERA5 (18 monthly variables); DEM (3) and soil (32) as
  static inputs; each channel imputed with the training median and standardized with training
  statistics. Encoder:
  linear embedding to 64, sinusoidal positional encoding, 2 Transformer layers, 4 heads,
  feed-forward 256, GELU, dropout 0.1, temporal mean pooling, layer normalization. S1 only and S2
  only: encoder and linear head. S1+S2: one encoder per modality, concatenated, head
  LayerNorm-Linear(64)-GELU-Dropout-Linear. S1+S2+TMS, S1+S2+CLIP, S1+S2+TMS+CLIP: encoders for
  S1, S2 (and ERA5); static inputs DEM, soil and the 32-d CLIP projection (trained as in Proj,
  standardized) each through Linear(64)-LayerNorm-GELU-Dropout-Linear(64)-LayerNorm; same head.
  Training as the linear probe, except learning rate halved after 5 epochs without improvement and
  early stopping after 15. Each model uses the points present in all of its raw files: S2 only
  keeps all 37,752 / 8,090 / 49,041 points; every model with S1 keeps 37,043 to 37,052 training
  and 7,931 to 7,960 validation points depending on the seed, and 48,607 test points.
- **Metrics.** Cohen's kappa, overall accuracy, macro and weighted F1, per-class F1
  (scikit-learn); tables report mean and standard deviation (ddof = 1) over the five seeds.

## Reproduction

```bash
pip install -r requirements.txt
export GEOCROP_DATA=<input directory>

GEOCROP_SEED=17 python run_ood.py      # one seed; 18 to 21 likewise, or: sbatch slurm/run_ood.sbatch
GEOCROP_AGGREGATE=1 python run_ood.py  # results/ood/*.csv -> results/ood/tables/
python make_fig1.py                    # -> results/figures/
```

A seed run overwrites `results/ood/results_seed<seed>_*.csv`, so `git diff results/` compares a
reproduction with the published run. The aggregation step needs only these CSVs: no input data and
no GPU.

| output in `results/ood/tables/` | paper |
|---|---|
| `table9.tex` | Table 9, single modalities, overall |
| `table10.tex` | Table 10, single modalities, per-class F1 (embeddings with the linear probe) |
| `table11.tex` | Table 11, fusion, overall |
| `table12.tex` | Table 12, per-class F1 of S1+S2+TMS+CLIP and of Tessera+CLIP with the best mean overall accuracy (random forest, Proj) |
| `overall_meanstd.csv`, `perclass_meanstd.csv` | all rows, mean and standard deviation |

Per-seed CSV columns: `Data, Fusion, Classifier, Seed, Kappa, OA, MacroF1, WeightedF1` (overall) and
`Data, Fusion, Classifier, Seed, Class, F1` (per class); OA and F1 in percent. `LPFusion32` and
`PCAFusion32` in `Fusion` are Proj and PCA of the paper.

The published run used one NVIDIA A100 GPU per seed (JUWELS Booster, Jülich Supercomputing Centre)
with the versions in `requirements.txt`. Rows involving gradient training (linear probe, Proj,
Transformers) ran on the GPU, which is not bitwise deterministic, and can differ in the last
digits on other hardware or library versions; all other rows depend only on inputs, seed and
library versions.

## License and acknowledgement

Apache License 2.0 (`LICENSE`). This work is supported by the AgroVision project, funded by the
European Space Agency (ESA) Phi-Lab under contract No. 4000147606.
