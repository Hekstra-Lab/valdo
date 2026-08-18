# valdo.pipeline CLI Reference

## What's New

Each stage of the crystallographic ML pipeline demonstrated in `notebooks/pipeline.ipynb`
is now available as a standalone command-line tool. Configuration is handled through
YAML (or JSON) files, making it straightforward to adapt the pipeline to new datasets
without modifying notebook cells.

```
valdo.pipeline <stage_name> <config.yaml>
valdo.pipeline init <stage_name>     # print a ready-to-edit template to stdout
```

---

## Quick Start

```bash
# 1. Generate a template config for the stage you want to run
valdo.pipeline init standardize > config_standardize.yaml

# 2. Edit the config with your actual paths and parameters
#    (all paths in the template are placeholders)

# 3. Run the stage
valdo.pipeline standardize config_standardize.yaml
```

---

## Reruns and `--force`

Every stage except `standardize` checks for its own output before doing any work,
so re-running a finished stage is cheap and safe:

```
Found existing vae_input.npy — skipping preprocess (use --force to rerun).
```

Pass `--force` to recompute anyway:

```bash
valdo.pipeline preprocess config_preprocess.yaml --force
```

The output each stage looks for is its main product — `reindex_record.pkl`,
`scaling_metrics.pkl`, `vae_input.npy`, the trained VAE, `recons.npy`, any MTZ in
the rescale output folder, `blob_stats.pkl`, `filtered_blob_stats_tagged.pkl`.

> **Rerunning with different parameters:** `--force` overwrites, but does not clean
> up. Stages that write one MTZ per dataset will leave files from the previous run
> behind if the new run covers fewer datasets, and the stale files will be picked
> up by the next stage. Empty the output folder yourself when changing the input set.

---

## Stage Overview

| Stage | Key Inputs | Key Outputs |
|-------|-----------|-------------|
| `standardize` | Raw MTZ directory | `####.mtz` files |
| `reindex` *(optional)* | Standardized MTZs, reference MTZ | Reindexed MTZs, `reindex_record.pkl` |
| `scale` | MTZ file list, reference MTZ | Scaled MTZs, `scaling_metrics.pkl` |
| `filter` *(optional)* | Refinement summary CSV, `scaling_metrics.pkl` | Text file listing the datasets that passed |
| `preprocess` | Scaled MTZs | `vae_input.npy`, `vae_output.npy`, `vae_sigF.npy`, `intersection.pkl`, `union.pkl`, `union_mean.pkl`, `union_sd.pkl` |
| `train` | `vae_input.npy`, `vae_output.npy` | `trained_vae.pkl` |
| `reconstruct` | Trained VAE, `vae_input.npy` | `recons.npy` |
| `rescale` | `recons.npy`, scaled MTZs | MTZs with `recons` and `diff` columns |
| `add_phases_and_blobs` | Rescaled MTZs, refined MTZs/PDBs | MTZs with phases+weights, `blob_stats.pkl` |
| `tag_blobs` | `blob_stats.pkl`, refined PDBs | `blob_stats_tagged.pkl`, `filtered_blob_stats_tagged.pkl` |

### Stage Dependency Diagram

```
standardize → [reindex →] scale → [filter →] preprocess → train → reconstruct → rescale ──┐
                    │                                                                        ↓
                    └──────────── valdo.refine (PHENIX) ─────────────────→ add_phases_and_blobs → tag_blobs
```

Square brackets indicate an **optional** stage. `valdo.refine` can run in parallel
with the VAE stages (`preprocess` through `rescale`) since they use independent inputs.

---

## Skipping the Reindex Stage

The `reindex` stage corrects **indexing ambiguity**, which may or may not be
present depending on the dataset. It can be skipped entirely when no ambiguity exists.

### How to check

Run the reindex stage once. If no ambiguity is detected, the function prints:

```
No ambiguity for this spacegroup! No need to reindex!
```

and produces no output MTZ files (`reindex_record.pkl` will be absent or have zero rows).
You can then skip this stage and route the `standardize` output directly into `scale`.

If reindexing *does* run successfully it produces a `reindex_record.pkl` with columns
`file_idx`, `best_symop`, and `num_duplicates`. Datasets with `num_duplicates > 1`
have unresolved ambiguity and should be inspected or excluded before scaling.

### Config paths when skipping reindex

```yaml
# scale config — reindex was skipped
file_list: "/path/to/input_mtzs/*.mtz"        # standardize output
reference_mtz: "/path/to/input_mtzs/0001.mtz" # standardized reference
```

### Config paths when reindex was run

```yaml
# scale config — reindex was run
file_list: "/path/to/reindexed/*.mtz"          # reindex output
# reference_mtz is the reindexed version of your reference dataset.
# Look up best_symop for your reference in reindex_record.pkl, e.g.:
#   df[df.file_idx == "0001"]["best_symop"]  →  0
# then the reindexed reference is 0001_0.mtz
reference_mtz: "/path/to/reindexed/0001_0.mtz"
```

---

## Per-Stage Config Reference

Generate any template with `valdo.pipeline init <stage> > config.yaml`.
Below is the full reference for each stage.

---

### `standardize`

Copies raw MTZ files to a standard `####.mtz` naming scheme. Rows with NaN
in `expcolumns` are dropped.

```yaml
source_path: "/path/to/original/mtzs/"
destination_path: "/path/to/pipeline/input_mtzs/"
mtz_file_pattern: ".*([0-9]{4}).*.mtz"   # regex; capture group 1 → 4-digit ID
expcolumns:
  - "F-obs"
  - "SIGF-obs"
ncpu: 1                                   # optional, default 1
```

---

### `reindex` *(optional)*

Corrects indexing ambiguity. Returns early with no output if no ambiguity is detected.

```yaml
input_files: "/path/to/input_mtzs/*.mtz"
reference_file: "/path/to/input_mtzs/0001.mtz"
output_folder: "/path/to/reindexed/"
columns:
  - "F-obs"
  - "SIGF-obs"
wcorr: true          # use weighted Pearson CC (recommended)
cc_min_dif: 0.2      # minimum CC gap to resolve ambiguity; lower = more permissive
ncpu: 1
```

---

### `scale`

Anisotropic Debye-Waller scaling of all datasets to a reference.

> **Filtering note:** To drop poorly-behaved datasets *after* scaling, use the
> `filter` stage below. `file_list` here should still be curated manually if you
> already know some datasets are unusable.

```yaml
file_list: "/path/to/reindexed/*.mtz"
reference_mtz: "/path/to/reindexed/0001_0.mtz"
columns:
  - "F-obs"
  - "SIGF-obs"
output_folder: "/path/to/scaled/"
prefix: ""           # prefix for scaling_metrics.pkl filename only (not for MTZ names)
when_opt: 0.2        # "all" | "never" | float [0,1] threshold for numerical optimisation
ncpu: 1
```

---

### `filter` *(optional)*

Selects which scaled datasets go into the VAE. Writes a plain text file with one
MTZ path per line, which `preprocess` (and later `rescale`) accepts directly as
its `file_list`. Four things are dropped:

1. the worse-refining copy of any dataset left ambiguous by `reindex` (higher `Rf_final` wins removal),
2. datasets whose refinement R-free exceeds `max_rfree`,
3. datasets whose `F-obs-scaled` column is entirely NaN (scaling diverged),
4. datasets whose post-scaling correlation is below `min_cc`.

```yaml
refine_summary: "/path/to/refine_summary.csv"
reindex_record: "/path/to/reindexed/reindex_record.pkl"   # null if reindex was skipped
scaled_dir: "/path/to/scaled/"
metrics: null        # scaling_metrics.pkl; auto-detected inside scaled_dir when null
max_rfree: 0.45      # drop datasets refining worse than this
min_cc: 0.55         # drop datasets scaling worse than this (0.0 disables the check)
output: "/path/to/configs/scaled_filtered_files.txt"
```

**The `refine_summary` CSV.** This stage reads refinement statistics that come
from your own refinement run, not from valdo. The CSV needs three columns:

| Column | Meaning |
|--------|---------|
| `file_idx` | Dataset ID, zero-padded to 4 digits (`0042`) |
| `symop` | Symmetry operator index chosen by `reindex`; use `0` throughout if reindex was skipped |
| `Rf_final` | R-free at the end of refinement |

Rows are matched back to scaled MTZ files by the filename `{file_idx}_{symop}.mtz`,
which is the naming `reindex` and `scale` produce. `PTP1B_pipeline/parse_refine_logs.py`
in this repository is a worked example that builds this CSV from PHENIX log files;
any other refinement pipeline works as long as the three columns are present.

Then point the next stage at the result:

```yaml
# preprocess config
file_list: "/path/to/configs/scaled_filtered_files.txt"
```

---

### `preprocess`

Computes intersection and union of Miller indices across all scaled datasets,
applies Z-score normalisation, and saves VAE input/output arrays.

```yaml
file_list: "/path/to/scaled/*.mtz"
output_folder: "/path/to/vae/"
intersection_path: "/path/to/vae/intersection.pkl"
union_path: "/path/to/vae/union.pkl"
sigF_path: "/path/to/vae/sigF.pkl"
amplitude_col: "F-obs-scaled"
error_col: "SIGF-obs-scaled"
include_errors: true
prefix: ""           # prefix for vae_input.npy, vae_output.npy, vae_sigF.npy
```

The `output_folder` also receives `union_mean.pkl` and `union_sd.pkl`, which
are required by the `rescale` stage (`info_folder` must point here).

---

### `train`

Trains the VAE. Supports Gaussian ELBO, error-weighted ELBO, and Student-t ELBO.

```yaml
vae_input_path: "/path/to/vae/vae_input.npy"
vae_output_path: "/path/to/vae/vae_output.npy"
vae_sigF_path: "/path/to/vae/vae_sigF.npy"   # required when include_errors: true
output_path: "/path/to/vae/trained_vae.pkl"
# Architecture
latent_dim: 7
n_hidden_layers: [3, 6]                        # [encoder_layers, decoder_layers]
n_hidden_size: 100
# Training
epochs: 500
train_fraction: 0.8
batch_size: 100
learning_rate: 0.001
w_kl: 1.0
eps: 0.02
stdof: null           # null = Gaussian; integer = Student-t degrees of freedom
include_errors: true
random_seed: 42
```

---

### `reconstruct`

Runs the trained VAE in reconstruction mode over all samples.

```yaml
vae_path: "/path/to/vae/trained_vae.pkl"
vae_input_path: "/path/to/vae/vae_input.npy"
output_path: "/path/to/vae/recons/recons.npy"
ml_recon: false   # true = MAP (deterministic); false = single sample
repeats: 1        # >1: saves shape [2, N_datasets, N_reflections] (mean + std)
```

---

### `rescale`

Reverses Z-score normalisation and writes per-dataset MTZ files with two new
columns: `recons` (VAE estimate) and `diff` (Fobs − Frecons).

```yaml
recons_path: "/path/to/vae/recons/recons.npy"
intersection_path: "/path/to/vae/intersection.pkl"
union_path: "/path/to/vae/union.pkl"
input_files: "/path/to/scaled/*.mtz"   # must match the order used in preprocess
info_folder: "/path/to/vae/"          # same as preprocess output_folder
output_folder: "/path/to/vae/recons/"
amplitude_col: "F-obs-scaled"
ncpu: 1
```

---

### Prerequisite: apo structure refinement

Before running `add_phases_and_blobs` you need apo-refined structures for every
dataset — one PDB and one MTZ (containing phases) per dataset. Use `valdo.refine`
to run PHENIX refinement in batch:

```bash
# If reindex was run, refine against the reindexed files:
valdo.refine --pdbpath /path/to/apo_model.pdb \
             --mtzpath "/path/to/reindexed/*.mtz" \
             --output /path/to/refined/ \
             --eff /path/to/refine_drug.eff

# If reindex was skipped, refine against the standardized files instead:
valdo.refine --pdbpath /path/to/apo_model.pdb \
             --mtzpath "/path/to/input_mtzs/*.mtz" \
             --output /path/to/refined/ \
             --eff /path/to/refine_drug.eff
```

This requires PHENIX to be installed and a `.eff` refinement config file.
The refinement output (`/path/to/refined/`) is what you point `phasing_path`
and `model_folder` to in the `add_phases_and_blobs` config.

> **Tip:** You can run refinement in parallel with the VAE training stages
> (`preprocess`, `train`, `reconstruct`, `rescale`) since they operate on
> independent inputs.

---

### `add_phases_and_blobs`

Runs four sequential steps on the rescaled MTZ files:
1. Copy phases from apo-refined MTZ files
2. Compute weights → adds `WT` and `WDF` (weighted difference) columns (in-place)
3. Compute extrapolated structure factors → adds `ESF_N` columns (in-place)
4. Detect electron density blobs via Gaussian blur + flood-fill

```yaml
file_list: "/path/to/vae/recons/*.mtz"
phasing_path: "/path/to/refined/"    # apo-refined MTZs with phases
output_folder: "/path/to/vae/recons_phased/"
blob_output_folder: "/path/to/vae/blobs/"
model_folder: "/path/to/refined/"    # PDB models for blob detection
# Phase columns (PHENIX default names)
phase_2FOFC_col_in: "PH2FOFCWT"
phase_FOFC_col_in: "PHFOFCWT"
phase_2FOFC_col_out: "PH2FOFCWT"
phase_FOFC_col_out: "PHFOFCWT"
rfree_label_in: null
# Weighting
sigF_col: "SIGF-obs-scaled"
diff_col: "diff"
sigdF_pct: 90.0
absdF_pct: 99.99
# Extrapolation
F_col: "F-obs-scaled"
recons_col: "recons"
extrapolate_factors: [2, 4, 8]
# Blob detection
blob_diff_col: "WDF"
phase_col: "PH2FOFCWT"
cutoff: 5.0        # blob significance threshold in sigma
radius_in_A: 5.0   # Gaussian blur radius in Angstroms
prefix: ""
ncpu: 1
```

---

### `tag_blobs`

Tags blobs by proximity to a focal residue (e.g. a reactive cysteine) and to
known bound ligands, identifies symmetry-equivalent positions, removes duplicates,
and writes both a fully tagged and a filtered DataFrame.

```yaml
blob_stats_path: "/path/to/vae/blobs/blob_stats.pkl"
model_folder: "/path/to/refined/"
mtz_folder: "/path/to/vae/recons_phased/"
# Focal residue — blobs within focal_radius Å are tagged and removed from
# the filtered output (e.g. Cys215 in PTP1B is a known false-positive source)
focal_seqid: 215
focal_tag_name: "cys215"
focal_radius: 5.0
output_folder: "/path/to/vae/blobs/"
ncpu: 1
```

Outputs:
- `blob_stats_tagged.pkl` — full DataFrame with tag columns (`focal_tag_name`, `ligand`, `duplicate`)
- `filtered_blob_stats_tagged.pkl` — blobs with `focal_tag_name == 0` and `duplicate == 0`
