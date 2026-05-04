# PTP1B Pipeline Run

Dataset: PTP1B fragment screening (Keedy et al.)
Reference structure: 1NWL (apo)

## Data

| Directory | Contents |
|-----------|----------|
| `original_data/` | 1,679 MTZ files, already standardized to `####.mtz` naming |
| `refine_1nwl/refine_output/` | PHENIX apo-refined PDB and MTZ files (phases) |
| `bound_models/all_superposed_clean_new-hits-removed_pdb+mtz/` | 142 bound ligand structures for blob tagging |

---

## Step 1: Standardize

Copies and renames raw MTZ files to the `####.mtz` naming scheme expected by the
rest of the pipeline. For this dataset this step is already done.

To run on a new dataset, generate a config with `valdo.pipeline init standardize`,
fill in `source_path`, `destination_path`, `mtz_file_pattern` (a regex with one
capture group for the numeric ID), and `expcolumns`, then:

- **Config:** `configs/config_standardize.yaml`
- **Command:** `valdo.pipeline standardize configs/config_standardize.yaml`
- **Input:** Raw MTZ files in `source_path/`
- **Output:** `destination_path/####.mtz`

---

## Step 2: Reindex

Spacegroup `P 31 2 1` has indexing ambiguity, so this step is required.
Reference dataset: `0001.mtz`.

- **Config:** `configs/config_reindex.yaml`
- **Command:** `valdo.pipeline reindex configs/config_reindex.yaml`
- **Input:** `original_data/*.mtz`
- **Output:** `reindexed/####_{symop}.mtz`, `reindexed/reindex_record.pkl`, `reindex_cc_diff_histogram.png`, `reindex_cc_diff_scatter.png`

If `reindex_record.pkl` already exists, re-running the command skips reindexing and regenerates the plots only.

22 datasets had unresolved ambiguity (`num_duplicates > 1`) and both symop variants were written to disk. These should be filtered out after refinement using R-factor comparison.

---

## Step 3: Scale

Anisotropic Debye-Waller scaling of all reindexed datasets to the reference (`0001_0.mtz`).
All reindexed files are passed including both symop variants for the 22 ambiguous datasets;
those will be filtered after refinement using R-factor comparison.

- **Config:** `configs/config_scale.yaml`
- **Command:** `valdo.pipeline scale configs/config_scale.yaml`
- **Input:** `reindexed/*.mtz` (1,702 files)
- **Output:** `scaled/####_{symop}.mtz` (1,702 files), `scaled/ptp1b_scaling_metrics.pkl`

A small number of datasets produced `RuntimeWarning: invalid value in multiply` during scaling — these are low-quality datasets expected to be filtered by R-factor before the VAE steps.

Validation plots are saved automatically to `scaled/`:
- `ptp1b_scaling_end_corr_histogram.png` — distribution of per-dataset correlation after scaling
- `ptp1b_scaling_LS_scatter.png` — start vs final least-squares error; points above the diagonal are datasets where scaling made things worse

If `scaling_metrics.pkl` already exists, re-running the command skips scaling and regenerates the plots only.

### Post-scaling filtering

Before preprocessing, datasets are filtered using `refine_1nwl/refine_summary.csv`:

- Remove datasets with `Rf_final > 0.45` (50 datasets)
- For the 22 ambiguous datasets, keep only the symop with the lower R-free

The filtered file list (1,650 paths) is saved to `configs/scaled_filtered_files.txt` and used as input to the next stage. The `.txt` file format is supported by all pipeline stages as an alternative to a glob pattern.

Two datasets were removed from the initial filtered list:
- `0003_1.mtz`: appeared in `refine_summary.csv` but `original_data/0003.mtz` does not exist, so no scaled file was ever produced.
- `0110_1.mtz`: scaling diverged silently — `F-obs` is fully valid but `F-obs-scaled` is entirely NaN. The `valdo.pipeline scale` command now warns about such files automatically after scaling completes.

---

## Step 4: Preprocess

Builds the intersection and union Miller index sets across all filtered scaled datasets, then
Z-score normalizes the structure factor amplitudes to produce VAE input/output arrays.

- **Config:** `configs/config_preprocess.yaml`
- **Command:** `valdo.pipeline preprocess configs/config_preprocess.yaml`
- **Input:** `configs/scaled_filtered_files.txt` (1,650 paths)
- **Output:**
  - `vae/intersection.pkl` — Miller indices common to all datasets (2,283 reflections)
  - `vae/union.pkl` — Miller indices in any dataset (77,821 reflections)
  - `vae/sigF.pkl` — error estimates on the union set
  - `vae/union_mean.pkl`, `vae/union_sd.pkl` — per-reflection mean and SD used for Z-scoring
  - `vae/vae_input.npy` — shape (1650, 2283), intersection amplitudes (Z-scored)
  - `vae/vae_output.npy` — shape (1650, 77821), union amplitudes (Z-scored)
  - `vae/vae_sigF.npy` — shape (1650, 77821), union error estimates (Z-scored)

---

## Step 5: Train VAE

Trains a Variational Autoencoder to reconstruct the full union set of structure factor
amplitudes from the intersection (complete) subset.

- **Config:** `configs/config_train.yaml`
- **Command:** `valdo.pipeline train configs/config_train.yaml`
- **Input:** `vae/vae_input.npy`, `vae/vae_output.npy`, `vae/vae_sigF.npy`
- **Output:** `vae/trained_vae.pkl` (32 MB), `vae/vae_loss_curves.png`

Key hyperparameters used (matching the pipeline notebook):
- `latent_dim: 7`, `n_hidden_layers: [3, 6]`, `n_hidden_size: 100`
- `activation: tanh`, `stdof: 128` (Student-t loss), `include_errors: true`
- `epochs: 500`, `batch_size: 100`, `learning_rate: 0.001`, `w_kl: 1.0`, `eps: 0.02`

Loss decreased from ~5.1×10⁶ (epoch 1) to ~2.6×10⁶ (epoch 500) with no NaN.

### Debugging notes

Training initially produced NaN in the first step. Two issues were identified and fixed
in `valdo.pipeline train` (no changes to the core library):

1. **All-NaN input row**: sample index 92 (`0110_1.mtz`) in `vae_input.npy` was entirely NaN —
   the scaling for this dataset diverged silently, producing all-NaN `F-obs-scaled`. It passed
   the R-factor filter but should have been excluded before preprocess. The `scale` stage now
   warns about such files automatically. The `train` runner also guards against this by dropping
   all-NaN rows before training.

2. **Logvar overflow**: a forward hook was registered on the encoder to clamp the log-variance
   output to `[-10, 10]`, preventing `exp(logvar)` from overflowing in the KL divergence term.
   Gradient clipping (`max_norm=1.0`) was also added as an additional safeguard.

---
