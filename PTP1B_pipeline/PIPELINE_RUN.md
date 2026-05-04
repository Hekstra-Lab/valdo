# PTP1B Pipeline — Worked Example

This document records a complete run of the `valdo.pipeline` CLI on the PTP1B fragment
screening dataset (Keedy et al., apo reference: PDB 1NWL). It doubles as a step-by-step
guide for adapting the pipeline to a new dataset: each step explains what it does, what
to configure, what output to expect, and notes on common pitfalls.

**Dataset:** 1,679 MTZ files from a PTP1B fragment screen  
**Reference structure:** 1NWL (apo form)  
**Final result:** AUC = 0.94 on 8,316 filtered blobs (78 positive ligand blobs)

---

## Prerequisites

Before starting, you need:

1. **MTZ files** — diffraction data from your fragment screen, one file per dataset.
2. **Apo reference model** — a refined PDB + MTZ for a ligand-free crystal of the same
   protein (used for phases in steps 8–9). For PTP1B this was produced by `valdo.refine`
   running PHENIX on each dataset independently; results are in `refine_1nwl/refine_output/`.
3. **`valdo` installed** — `pip install -e .` from the repo root.
4. **A GPU** — required for step 5 (VAE training). Steps 1–4 and 6–9 run on CPU.

**Directory layout used in this run:**

| Directory | Contents |
|-----------|----------|
| `original_data/` | 1,679 raw MTZ files, standardized to `####.mtz` naming |
| `refine_1nwl/refine_output/` | PHENIX-refined per-dataset PDB and MTZ files (phases) |
| `bound_models/all_superposed_clean_new-hits-removed/` | 142 known bound-state PDB files (optional, for evaluation only) |
| `configs/` | YAML config files for each pipeline stage |
| `vae/` | All VAE intermediate files and outputs |

---

## Step 1: Standardize

**What it does:** Copies raw MTZ files into a working directory and renames them to the
`####.mtz` scheme (e.g. `0001.mtz`, `0002.mtz`) that all downstream steps expect.

> **For this dataset:** Already done — `original_data/` contains files in the correct
> naming scheme. Skip to Step 2.

**To run on a new dataset:**

```bash
valdo.pipeline init standardize > configs/config_standardize.yaml
# Edit: set source_path, destination_path, mtz_file_pattern, expcolumns
valdo.pipeline standardize configs/config_standardize.yaml
```

- `mtz_file_pattern` is a regex with one capture group that extracts the numeric ID
  from your filenames (e.g. `".*?(\d{4})\.mtz"` matches `experiment_0042.mtz` → `0042`).
- `expcolumns` lists the column names to carry over (e.g. `[F, SIGF]`).

- **Config:** `configs/config_standardize.yaml`
- **Input:** Raw MTZ files in `source_path/`
- **Output:** `destination_path/####.mtz`

---

## Step 2: Reindex

**What it does:** Corrects indexing ambiguity — datasets in your screen may have been
indexed under different but equivalent conventions. This step correlates each dataset
against a reference and applies whichever reindexing operator gives the highest correlation,
ensuring all datasets share a consistent indexing.

> **When to skip:** If your space group has no indexing ambiguity, the function prints
> `"No ambiguity for this spacegroup!"` and exits immediately. You can verify by running
> the stage once and checking whether any output files are produced.

For PTP1B (`P 31 2 1`), reindexing is required. Reference dataset: `0001.mtz`.

```bash
valdo.pipeline reindex configs/config_reindex.yaml
```

- **Config:** `configs/config_reindex.yaml`
- **Input:** `original_data/*.mtz` (1,679 files)
- **Output:**
  - `reindexed/####_{symop}.mtz` — reindexed MTZ files (`_0` = identity, `_1` = alternate)
  - `reindexed/reindex_record.pkl` — per-dataset reindexing result
  - `reindex_cc_diff_histogram.png`, `reindex_cc_diff_scatter.png` — diagnostic plots

Re-running when `reindex_record.pkl` already exists skips reindexing and regenerates the
plots only.

**This run:** 22 datasets had unresolved ambiguity (`num_duplicates > 1`); both symop
variants were written to disk. These are resolved after refinement using R-factor comparison
(see Post-scaling filtering in Step 3).

---

## Step 3: Scale

**What it does:** Applies anisotropic Debye-Waller scaling to bring all datasets onto a
common amplitude scale using the reference dataset as a target. This corrects for
per-crystal radiation damage, lattice imperfections, and beam variation.

```bash
valdo.pipeline scale configs/config_scale.yaml
```

- **Config:** `configs/config_scale.yaml`
- **Input:** `reindexed/*.mtz` (1,702 files — includes both symop variants for ambiguous datasets)
- **Output:**
  - `scaled/####_{symop}.mtz` (1,702 files)
  - `scaled/ptp1b_scaling_metrics.pkl` — per-dataset scaling statistics
  - `scaled/ptp1b_scaling_end_corr_histogram.png` — distribution of end-of-scaling correlations
  - `scaled/ptp1b_scaling_LS_scatter.png` — start vs. final least-squares loss (points above the diagonal are datasets where scaling made things worse)

Re-running when `scaling_metrics.pkl` already exists skips scaling and regenerates plots only.

> **Watch out for:** A small number of datasets may produce
> `RuntimeWarning: invalid value in multiply`. These are low-quality datasets that will
> be caught by the R-factor filter below. After scaling completes, the pipeline also
> warns about any output file where `F-obs-scaled` is entirely NaN — exclude those
> files from subsequent steps.

### Apo refinement

Before filtering, you need per-dataset apo refinement statistics (R-factors) and phase
files for Step 8. Run PHENIX refinement against the apo model for all scaled datasets
using the `valdo.refine` CLI:

```bash
valdo.refine \
  --pdbpath  refine_1nwl/1nwl_apo.pdb \
  --mtzpath  "scaled/*.mtz" \
  --output   refine_1nwl/refine_output/ \
  --eff      notebooks/refine_drug.eff
```

This produces four files per dataset in `refine_output/`:
- `refine_####_{symop}_001.pdb` — refined model
- `refine_####_{symop}_001.mtz` — refined MTZ with phases (`PH2FOFCWT`, `PHFOFCWT`)
- `refine_####_{symop}_001.log` — PHENIX log with R-factors
- `refine_####_{symop}_001.eff` — complete PHENIX configuration used

> Refinement takes ~3 minutes per dataset with PHENIX. Run it on a cluster.
> An example `refine_drug.eff` is provided in `notebooks/`.

Parse the R-factors from the log files into a summary CSV using the provided script:

```bash
python parse_refine_logs.py refine_1nwl/refine_output/ refine_1nwl/refine_summary.csv
```

This produces `refine_summary.csv` with columns `file_idx`, `symop`, `Rw_start`,
`Rf_start`, `Rw_final`, `Rf_final`, `time(s)`.

### Post-scaling filtering

This is a **manual step** before preprocessing. Use `filter_datasets.py` to remove
poor-quality datasets, resolve reindexing ambiguity, and write a filtered file list:

```bash
python filter_datasets.py \
  --refine-summary refine_1nwl/refine_summary.csv \
  --reindex-record reindexed/reindex_record.pkl \
  --scaled-dir     scaled/ \
  --max-rfree      0.45 \
  --output         configs/scaled_filtered_files.txt
```

Pass `--min-cc 0.55` to additionally remove datasets whose post-scaling correlation
with the reference falls below 0.55.

All subsequent pipeline stages accept a `.txt` file list as `input_files` / `file_list`,
as an alternative to a glob pattern.

**This run:** 1,650 paths after filtering (50 removed for `Rf_final > 0.45`, 22 symop
duplicates resolved, 2 additional files removed automatically):
- `0003_1.mtz` — `original_data/0003.mtz` never existed so no scaled file was produced;
  `filter_datasets.py` only picks up files present on disk.
- `0110_1.mtz` — scaling diverged silently: `F-obs-scaled` is entirely NaN despite a
  valid R-free (0.22). `filter_datasets.py` detects and drops all-NaN scaled files
  automatically.

---

## Step 4: Preprocess

**What it does:** Finds the intersection (reflections present in every dataset) and union
(reflections present in any dataset) of Miller indices, then Z-score normalizes the
structure factor amplitudes. The intersection becomes the VAE input; the union becomes
the VAE output.

```bash
valdo.pipeline preprocess configs/config_preprocess.yaml
```

- **Config:** `configs/config_preprocess.yaml`
- **Input:** `configs/scaled_filtered_files.txt` (1,650 paths)
- **Output:**
  - `vae/intersection.pkl` — Miller indices common to all datasets (2,283 reflections)
  - `vae/union.pkl` — Miller indices present in any dataset (77,821 reflections)
  - `vae/sigF.pkl` — error estimates on the union set
  - `vae/union_mean.pkl`, `vae/union_sd.pkl` — per-reflection mean and SD for Z-scoring
  - `vae/vae_input.npy` — shape (1650, 2283), Z-scored intersection amplitudes
  - `vae/vae_output.npy` — shape (1650, 77821), Z-scored union amplitudes
  - `vae/vae_sigF.npy` — shape (1650, 77821), Z-scored union error estimates

---

## Step 5: Train VAE

**What it does:** Trains a Variational Autoencoder (VAE) to reconstruct the full union set
of structure factor amplitudes from the smaller intersection subset. The VAE learns a
low-dimensional latent representation of the per-dataset diffraction signal, enabling
it to impute missing reflections and denoise the data.

```bash
valdo.pipeline train configs/config_train.yaml
```

- **Config:** `configs/config_train.yaml`
- **Input:** `vae/vae_input.npy`, `vae/vae_output.npy`, `vae/vae_sigF.npy`
- **Output:** `vae/trained_vae.pkl` (32 MB), `vae/vae_loss_curves.png`

**Hyperparameters used** (matching the published pipeline notebook):

| Parameter | Value | Notes |
|-----------|-------|-------|
| `latent_dim` | 7 | Dimensionality of latent space |
| `n_hidden_layers` | [3, 6] | Encoder / decoder depth |
| `n_hidden_size` | 100 | Units per hidden layer |
| `activation` | tanh | |
| `stdof` | 128 | Student-t degrees of freedom for the loss; `null` = Gaussian |
| `include_errors` | true | Weights loss by experimental σ(F) |
| `epochs` | 500 | |
| `batch_size` | 100 | |
| `learning_rate` | 0.001 | |
| `w_kl` | 1.0 | KL divergence weight |
| `eps` | 0.02 | Noise floor added to σ(F) |

Loss decreased from ~5.1×10⁶ (epoch 1) to ~2.6×10⁶ (epoch 500) with no NaN.

### Troubleshooting: NaN loss

If training produces NaN loss, two root causes are most likely:

1. **All-NaN input row** — one or more datasets in `vae_input.npy` are entirely NaN
   because scaling diverged for that dataset. `filter_datasets.py` catches these
   automatically (Step 3), so this should not occur if you used it to build your file
   list. As an extra safeguard, the `train` runner also drops any remaining all-NaN
   rows and warns you.

2. **Logvar overflow** — the encoder log-variance output becomes very large, causing
   `exp(logvar)` to overflow in the KL term. The `train` runner guards against this by
   clamping logvar to `[-10, 10]` via a forward hook, and applies gradient clipping
   (`max_norm=1.0`). If NaN persists, try reducing `learning_rate` or adding more
   `eps`.

---

## Step 6: Reconstruct

**What it does:** Passes all 1,650 inputs through the trained VAE to produce MAP-estimated
reconstructed structure factor amplitudes for the full union set.

```bash
valdo.pipeline reconstruct configs/config_reconstruct.yaml
```

- **Config:** `configs/config_reconstruct.yaml`
- **Input:** `vae/trained_vae.pkl`, `vae/vae_input.npy`
- **Output:** `vae/recons/recons.npy` — shape (1650, 77821), all finite, range [−9.0, 9.3]σ

> `ml_recon: true` uses the MAP (maximum a posteriori) reconstruction rather than a
> stochastic sample. Set `repeats > 1` to estimate reconstruction uncertainty.

---

## Step 7: Rescale

**What it does:** Reverses the Z-score normalization applied in Step 4 to recover
original-scale amplitudes, and computes two new per-dataset columns:
- `recons` — VAE-reconstructed amplitude in the original scale
- `diff` — difference between the observed and reconstructed amplitude

```bash
valdo.pipeline rescale configs/config_rescale.yaml
```

- **Config:** `configs/config_rescale.yaml`
- **Input:** `vae/recons/recons.npy`, `vae/intersection.pkl`, `vae/union.pkl`, `configs/scaled_filtered_files.txt`
- **Output:** `vae/recons/####_{symop}.mtz` — 1,650 MTZ files with `recons` and `diff` columns added

---

## Step 8: Add Phases and Blobs

**What it does:** For each reconstructed dataset, this step:
1. Transfers phases from the corresponding PHENIX-refined MTZ in `refine_output/`
   (columns `PH2FOFCWT`, `PHFOFCWT`).
2. Computes sigma-weighted difference map weights (`WT`, `WDF` columns).
3. Extrapolates structure factor amplitudes at several scales (`ESF_2`, `ESF_4`, etc.)
   to amplify weak ligand signals.
4. Detects electron density blobs via Gaussian-blurred flood-fill and records their
   peak value, score, centroid, volume, and radius.

```bash
valdo.pipeline add_phases_and_blobs configs/config_add_phases_and_blobs.yaml
```

- **Config:** `configs/config_add_phases_and_blobs.yaml`
- **Input:** `vae/recons/*.mtz` (1,650 files), `refine_1nwl/refine_output/`
- **Output:**
  - `vae/recons_phased/####_{symop}.mtz` — phased MTZ files with `WT`, `WDF`, `ESF_2`, `ESF_4`, `ESF_6`, `ESF_8`, `ESF_16` columns
  - `vae/blobs/blob_stats.pkl` — 9,318 blobs (columns: `sample`, `peakz`, `peak`, `score`, `cenx`, `ceny`, `cenz`, `volume`, `radius`)

> **Why 802 phased files instead of 1,650?** Each dataset is matched to a phase file by
> sample ID. Datasets whose refinement failed or was not run will not have a phase file
> and are silently skipped. Check `refine_output/` coverage if you expect more output.

---

## Step 9: Tag Blobs

**What it does:** Annotates each blob with contextual labels used for filtering and
evaluation:
- **`cys215`** — 1 if the blob centroid is within `focal_radius` Å of the Cys215
  active-site residue. These are excluded from the final hit list (expected false positives
  near the catalytic site).
- **`ligand`** — 1 if the blob overlaps with a known bound ligand atom from a reference
  structure (requires `bound_models_folder`; set to 0 for all blobs if omitted).
- **`bound`** — 1 if this dataset has a corresponding known bound-state model.
- **`duplicate`** — 1 if a symmetry-equivalent blob already appears elsewhere in the table.

Fractional coordinates and all symmetry-equivalent positions are also computed.

```bash
valdo.pipeline tag_blobs configs/config_tag_blobs.yaml
```

- **Config:** `configs/config_tag_blobs.yaml`
- **Input:**
  - `vae/blobs/blob_stats.pkl`
  - `refine_1nwl/refine_output/` — per-dataset PDB files for residue proximity search
  - `vae/recons_phased/` — phased MTZ files for fractional coordinate computation
  - `bound_models/all_superposed_clean_new-hits-removed/` — known bound-state PDB files *(optional)*
- **Output:**
  - `vae/blobs/blob_stats_tagged.pkl` — 9,318 blobs with all annotation columns added
  - `vae/blobs/filtered_blob_stats_tagged.pkl` — 8,316 blobs passing `cys215 == 0` and `duplicate == 0`

**This run:** 624 blobs (from 141 known-bound datasets) were flagged as `bound`. After
excluding active-site and duplicate blobs, 8,316 blobs remain for evaluation.

> **No known bound structures?** Omit `bound_models_folder` from the config. The `ligand`
> and `bound` columns will be set to 0 for all blobs and the ligand-tagging step is
> skipped. The filtered output is still useful for visual inspection and hit prioritization.

Key parameters: `focal_seqid: 215`, `focal_radius: 5.0 Å`, `focal_tag_name: cys215`, `ncpu: 8`

---

## Step 10: Evaluate (AUC)

If you have known bound structures, evaluate hit-calling performance with the provided
script:

```bash
python plot_auc.py vae/blobs/filtered_blob_stats_tagged.pkl
```

This plots the ROC curve using blob `score` as the ranking metric and saves
`roc_curve.png` next to the input file.

**This run:** AUC = **0.9399** (78 positive blobs, 8,238 negative blobs).

> The `score` column is the integrated blob intensity above the contour threshold,
> computed by gemmi's flood-fill. Higher score = larger / stronger blob = more likely
> to be a real ligand event.
