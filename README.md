# geocrop-mm

Code and results for Tables 9 to 12 and Figure 1 of

C. Paris, M. F. Celik, S. Maurogiovanni, G. Tseng, V. Marsocci, R. Cartuyvels, G. Cavallaro,
R. Sedona, *Multimodal Crop Type Classification with Geospatial Foundation Models and Geo-tagged
Field Photos*, submitted to the International Journal of Applied Earth Observation and
Geoinformation.

Tables 9 to 12 report models trained on LUCAS 2018 and tested on LUCAS 2022. All their cells are
computed here except the OlmoEarth cells, which, like Tables 2 to 8, were computed outside this
code. `run_ood.py` still reads the OlmoEarth file, because the evaluated points are those present
in all five embedding files; the OlmoEarth rows it writes are not the values in the paper.

## Files

| path | |
|---|---|
| `run_ood.py` | experiment (one seed per run) and aggregation into Tables 9 to 12 |
| `make_fig1.py` | Figure 1 |
| `crop_classes.py` | LUCAS LC1 codes of the 17 crop classes |
| `slurm/run_ood.sbatch` | SLURM array job, seeds 17 to 21 |
| `data/SHA256SUMS` | checksums of the input files |
| `data/clip_photos_{2018,2022}.csv.gz` | for each CLIP embedding, the `point_id` and the `url` of the encoded photo |
| `results/ood/` | per-seed results of the runs reported in the paper |
| `requirements.txt` | package versions of those runs |

## Input data

Both scripts read the directory `$GEOCROP_DATA`. The files below are not distributed with this
repository; `cd "$GEOCROP_DATA" && sha256sum -c <repo>/data/SHA256SUMS` checks that they are
the ones used in the paper. Arrays are indexed by the 8-digit LUCAS point id.

| file | content | arrays | N 2018 | N 2022 |
|---|---|---|---|---|
| `{y}/LUCAS_{y}_LULC.csv` | LUCAS survey points; columns used: `POINT_ID`, `Country`, `LC1`, `LAT_USED`, `LON_USED` | | 55,220 | 49,833 |
| `{y}/AE_{y}_Xy_v2.npz` | AlphaEarth | `X` (N,1,64,3,3), `point_ids` | 123,483 | 115,844 |
| `{y}/PE_{y}_X.npz` | Presto | `embeddings` (N,128,3,3), `point_ids` | 123,483 | 115,744 |
| `{y}/TE_{y}_data.npz` | Tessera | `X` (N,1,128,3,3), `point_ids` | 107,642 | 103,841 |
| `{y}/OE_{y}_AI2_v2.npz` | OlmoEarth, int8, from external partners | `data` (N,768,3,3), `point_ids` | 105,738 | 103,468 |
| `{y}/CLIP_{y}_point.npz` | CLIP embedding of the LUCAS Point photo | `data` (N,1024), `point_id`, `url` | 55,081 | 49,282 |
| `{y}/modalities/S1_{y}_Xy_v2.npz` | Sentinel-1 monthly, VV, VH | `X` (N,12,2,3,3), `point_ids` | 120,032 | 113,214 |
| `{y}/modalities/S2_{y}_Xy_v2.npz` | Sentinel-2 L2A monthly, B2 to B8A, B11, B12 | `X` (N,12,10,3,3), `point_ids` | 123,483 | 115,744 |
| `{y}/modalities/ERA5_{y}_Xy_v2.npz` | ERA5-Land monthly, 18 variables | `X` (N,12,18,1,1), `point_ids` | 123,483 | 115,844 |
| `{y}/modalities/DEM_{y}_Xy_v2.npz` | Copernicus DEM GLO-30: height, slope, aspect | `X` (N,1,3,1,1), `point_ids` | 123,483 | 115,844 |
| `{y}/modalities/SOIL_{y}_Xy_v2.npz` | AI4SoilHealth, 32 properties | `X` (N,1,32,1,1), `point_ids` | 107,642 | 103,841 |
| `ne_50m_admin_0_countries.zip` | [Natural Earth](https://naciscdn.org/naturalearth/50m/cultural/ne_50m_admin_0_countries.zip) 1:50m countries, Figure 1 only | | | |

`{y}` is 2018 or 2022; the LUCAS counts are points of the 17 classes. Satellite, ERA5, DEM and
soil values cover the 3 x 3 pixels around each point and were produced as described in Section 2
of the paper (Celik et al. 2026a, doi:10.5281/zenodo.17988209; Celik et al. 2026b). The photos
behind the CLIP embeddings are the LUCAS Point photos on Eurostat GISCO, listed in
`data/clip_photos_{y}.csv.gz`.

## Method

What `run_ood.py` does:

- Labels: LC1 codes of `crop_classes.py`, all other points dropped. Evaluated points are those
  present in all five embedding files: 54,059 in 2018, 49,041 in 2022.
- Split (seeds 17 to 21): the 2018 points are stratified by class and country, groups with fewer
  than 5 points are dropped, 70 % go to training and the rest is split in half into validation and
  an unused test part. Result: 37,752 training and 8,090 validation points. The test set is all
  49,041 points of 2022.
- Embeddings: mean over the 3 x 3 window; OlmoEarth dequantized as sign(x) (|x|/127.5)^2; values
  that are non-finite or at most -9990 replaced by the training median; standardized with training
  statistics.
- Linear probe: one linear layer, Adam (learning rate 1e-3), batch 256, cross-entropy with class
  weights n / (K n_c) normalized to mean 1, learning rate halved after 10 epochs without improvement
  of the validation loss, stop after 20 (at most 200 epochs), weights of the lowest validation loss.
- Random forest: 100 trees, `class_weight='balanced_subsample'`, `max_features='sqrt'`,
  `random_state` = seed.
- Fusion, after standardizing each input: Naive concatenates; PCA keeps 32 components of the
  concatenation; Proj maps the concatenation to 32 dimensions with a linear layer trained together
  with a linear classifier (as the linear probe), which is then dropped. The result is standardized
  again.
- Transformers: S1 and S2 as 12 monthly steps (window mean, fill values ignored), ERA5 as 12 steps,
  DEM and soil as static vectors, each channel imputed and standardized as above. Encoder: linear
  embedding to 64, sinusoidal positions, 2 layers, 4 heads, feed-forward 256, GELU, dropout 0.1,
  mean over time, LayerNorm. S1 only and S2 only add a linear classifier. S1+S2 concatenates one
  encoder per sensor, classifier LayerNorm, Linear(64), GELU, Dropout, Linear. S1+S2+TMS,
  S1+S2+CLIP and S1+S2+TMS+CLIP add an encoder for ERA5 (TMS rows) and, for each static input
  (DEM, soil, 32-d CLIP projection trained as in Proj), Linear(64), LayerNorm, GELU, Dropout,
  Linear(64), LayerNorm, with the same classifier. Training as the linear probe, but the learning
  rate is halved after 5 epochs and training stops after 15. Each model keeps only points present
  in all its input files: 48,607 test points for every model with S1 (training and validation
  37,043 to 37,052 and 7,931 to 7,960, depending on the seed), all points for S2 only.
- Metrics: Cohen's kappa, overall accuracy, macro and weighted F1, per-class F1; the tables give
  mean and standard deviation (ddof = 1) over the five seeds.

## Usage

```bash
pip install -r requirements.txt
export GEOCROP_DATA=<input directory>
GEOCROP_SEED=17 python run_ood.py       # likewise 18 to 21, or sbatch slurm/run_ood.sbatch
GEOCROP_AGGREGATE=1 python run_ood.py   # tables from results/ood/*.csv, into results/ood/tables/
python make_fig1.py                     # into results/figures/
```

A seed run overwrites its file in `results/ood/`, so `git diff` shows any difference from the
paper's runs. The aggregation needs neither the input data nor a GPU.

| `results/ood/tables/` | paper |
|---|---|
| `table9.tex`, `table10.tex`, `table11.tex` | Tables 9, 10, 11 (Table 10: embeddings with the linear probe) |
| `table12.tex` | Table 12: S1+S2+TMS+CLIP, and Tessera+CLIP with the highest mean accuracy (random forest, Proj) |
| `overall_meanstd.csv`, `perclass_meanstd.csv` | mean and standard deviation of every row |

Columns of the per-seed files: `Data, Fusion, Classifier, Seed, Kappa, OA, MacroF1, WeightedF1`
and `Data, Fusion, Classifier, Seed, Class, F1`, with OA and F1 in percent. `LPFusion32` is Proj.

The paper's runs used one NVIDIA A100 per seed on JUWELS Booster (Jülich Supercomputing Centre).
Training on GPU is not bitwise deterministic, so linear probe, Proj and Transformer rows can differ
in the last digits on other hardware or package versions.

## License

Apache License 2.0. Supported by the AgroVision project, funded by the European Space Agency
(ESA) Phi-Lab under contract No. 4000147606.
