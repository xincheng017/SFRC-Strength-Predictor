# SFRC Strength Database (CS & STS)

Consolidated experimental database of steel fiber-reinforced concrete (SFRC) used in the companion paper:

> *Machine learning-based prediction of compressive and splitting tensile strength of steel fiber-reinforced concrete: A comparative study of optimization algorithms and interpretability analysis.* Submitted to **Journal of Materials in Civil Engineering (ASCE)**.

## Files

| File | Description |
|---|---|
| `SFRC_cleaned_datasets_v2.xlsx` | Master database (two sheets: `CS_cleaned_v2`, `STS_cleaned_v2`) |
| `CS_dataset_n431.csv` | Compressive-strength dataset (n = 431), CSV export of sheet `CS_cleaned_v2` |
| `STS_dataset_n240.csv` | Splitting-tensile-strength dataset (n = 240), CSV export of sheet `STS_cleaned_v2` |
| `data_sources.md` | Full list of the **47 source publications** from which the database was compiled |

## Data dictionary

### Model input features (as used in the paper)

| Column | Unit | Description |
|---|---|---|
| `Water [kg]` | kg/m³ | Water content |
| `Cement [kg]` | kg/m³ | Cement content |
| `Sand [kg]` | kg/m³ | Fine aggregate (sand) content |
| `CA [kg]` | kg/m³ | Coarse aggregate content |
| `smax [mm]` | mm | Maximum aggregate size |
| `SP [kg]` | kg/m³ | Superplasticizer dosage |
| `Fiber type` | – | Categorical steel-fiber type (6 classes; one-hot encoded for modeling) |
| `Vf [%]` | % | Fiber volume fraction |
| `df [mm]` | mm | Fiber diameter |
| `Lf [mm]` | mm | Fiber length |

### Target variables

| Column | Sheet | Unit | Description |
|---|---|---|---|
| `fcy [MPa]` | CS | MPa | Measured compressive strength (CS) |
| `fst [MPa]` | STS | MPa | Measured splitting tensile strength (STS) |

### Auxiliary / derived columns (provided for convenience; not model inputs)

`ρf` (fiber density), `ff [MPa]` (fiber tensile strength; STS sheet only), `w/c`, `s/a`, `a/c`, `Lf/df`, `ρf·Lf/df`, `Lf_df_ratio`, `rho_f_x_Lf_df`, `fiber_factor`.

## Compilation protocol (summary)

- Compiled from **47 independent published studies** (journal articles and theses) spanning diverse countries, steel fiber types, and mix proportions; see `data_sources.md` for the complete reference list in ASCE author–date format.
- A standardized extraction protocol required every retained specimen to report complete mix-design parameters, fiber characteristics, and the corresponding mechanical test result.
- Samples missing critical features were excluded (listwise deletion); no imputation was applied.
- Resulting sizes: **CS: n = 431**, **STS: n = 240** (671 samples in total).

## Reproducing the paper's split

Stratified 80:20 train/test split (`StratifiedShuffleSplit`, fixed random seed) using tertile-discretized strength combined with fiber type as the stratification key, yielding **344/87 (CS)** and **192/48 (STS)**. See the training scripts in this repository for the exact seed and code.

## Citation

If you use this database, please cite the companion paper (citation will be updated upon publication) and the original experimental studies listed in `data_sources.md`.

## License / provenance note

All data points were digitized or transcribed from the published literature listed in `data_sources.md`; copyright of the original experiments remains with the respective authors. The compiled dataset is shared for research reproducibility.
