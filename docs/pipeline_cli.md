# valdo.pipeline

`valdo.pipeline` runs the VALDO screening pipeline one stage at a time from the
command line. Each stage does the job of one section of `notebooks/pipeline.ipynb`,
but reads its settings from a YAML (or JSON) file, so adapting the pipeline to a new
dataset means editing a config rather than notebook cells.

There are only two commands to learn:

```bash
valdo.pipeline init <stage>          # print a commented config template to stdout
valdo.pipeline <stage> <config.yaml> # run that stage
```

## Contents

- [Before you start](#before-you-start)
- [Quick start](#quick-start)
- [The stages](#the-stages)
- [Writing config files](#writing-config-files)
- [Re-running a stage](#re-running-a-stage)
- [Stage reference](#stage-reference)
- [What you end up with](#what-you-end-up-with)

---

## Before you start

You need three things:

1. **Diffraction data** — one MTZ per crystal, with an amplitude column and its
   error column (`F-obs` / `SIGF-obs` by default). Hundreds to thousands of
   datasets is the normal scale.
2. **An apo model** — a PDB of the unbound structure, used to refine every dataset.
3. **PHENIX** — required by `valdo.refine`, which produces the per-dataset refined
   models and phases that the last two stages consume. Refinement is not part of
   `valdo.pipeline`; see [Refining your datasets](#refining-your-datasets) for where
   it fits.

A GPU is optional but makes the `train` stage substantially faster. Stages that
support `ncpu` use `multiprocessing` and benefit from a many-core machine.

---

## Quick start

Every stage follows the same three steps:

```bash
# 1. Write a template config, pre-filled with defaults and comments
valdo.pipeline init standardize > config_standardize.yaml

# 2. Edit it — every path in the template is a placeholder
$EDITOR config_standardize.yaml

# 3. Run it
valdo.pipeline standardize config_standardize.yaml
```

Working through a full screen means repeating that for each stage in order. A
complete run looks like this:

```bash
valdo.pipeline standardize          config_standardize.yaml
valdo.pipeline reindex              config_reindex.yaml       # optional
valdo.pipeline scale                config_scale.yaml
# → refine every dataset with valdo.refine (can run in parallel with the VAE stages)
valdo.pipeline filter               config_filter.yaml        # optional
valdo.pipeline preprocess           config_preprocess.yaml
valdo.pipeline train                config_train.yaml
valdo.pipeline reconstruct          config_reconstruct.yaml
valdo.pipeline rescale              config_rescale.yaml
valdo.pipeline add_phases_and_blobs config_add_phases_and_blobs.yaml
valdo.pipeline tag_blobs            config_tag_blobs.yaml
```

Run them one at a time and look at the output before moving on — `reindex`, `scale`
and `train` each write validation plots next to their results, and they are the
cheapest way to catch a problem before it propagates.

`PTP1B_pipeline/` in this repository is a real worked example: the configs from a
1,679-dataset PTP1B fragment screen, plus `PIPELINE_RUN.md` recording what each stage
produced and which datasets were dropped along the way.

---

## The stages

| Stage | Reads | Writes |
|-------|-------|--------|
| `standardize` | Raw MTZ directory | `####.mtz` files |
| `reindex` *(optional)* | Standardized MTZs, reference MTZ | Reindexed MTZs, `reindex_record.pkl`, 2 plots |
| `scale` | MTZ file list, reference MTZ | Scaled MTZs, `scaling_metrics.pkl`, 2 plots |
| `filter` *(optional)* | Refinement summary CSV, `scaling_metrics.pkl` | Text file listing the datasets that passed |
| `preprocess` | Scaled MTZs | `vae_input.npy`, `vae_output.npy`, `vae_sigF.npy`, `intersection.pkl`, `union.pkl`, `union_mean.pkl`, `union_sd.pkl` |
| `train` | `vae_input.npy`, `vae_output.npy` | `trained_vae.pkl`, loss curves |
| `reconstruct` | Trained VAE, `vae_input.npy` | `recons.npy` |
| `rescale` | `recons.npy`, scaled MTZs | MTZs with `recons` and `diff` columns |
| `add_phases_and_blobs` | Rescaled MTZs, refined MTZs/PDBs | MTZs with phases + weights, `blob_stats.pkl` |
| `tag_blobs` | `blob_stats.pkl`, refined PDBs | `blob_stats_tagged.pkl`, `filtered_blob_stats_tagged.pkl` |

```
standardize → [reindex →] scale → [filter →] preprocess → train → reconstruct → rescale ──┐
                    │                                                                       ↓
                    └────────── valdo.refine (PHENIX) ──────────────────→ add_phases_and_blobs → tag_blobs
```

Square brackets mark optional stages:

- **`reindex`** is only needed if your datasets suffer from indexing ambiguity.
  See [`reindex`](#reindex-optional) for how to tell.
- **`filter`** drops badly-behaved datasets before training. You can skip it and
  curate the file list by hand instead.

Refinement runs on the same inputs as `scale`, so it can proceed in parallel with
everything from `preprocess` through `rescale`.

---

## Writing config files

**Format.** Files ending in `.yaml` or `.yml` are parsed as YAML; anything else is
parsed as JSON. YAML is easier to read and lets you keep comments.

**Start from `init`.** `valdo.pipeline init <stage>` prints a template listing every
field with a comment explaining it, and it is guaranteed to match the version you
have installed. Note that a template carries *recommended* starting values, which
are not always the same as the fallback used when you omit a field — the templates
prefer the settings from the PTP1B reference run. The values in this document are
the fallbacks.

**Selecting input files.** The fields that pick datasets — `file_list` in most
stages, `input_files` in `reindex` and `rescale` — accept three forms:

```yaml
file_list: "/path/to/scaled/*.mtz"          # a glob, expanded and sorted
file_list: ["/path/a.mtz", "/path/b.mtz"]   # an explicit list
file_list: "/path/to/filtered_files.txt"    # a .txt file, one path per line
```

The `.txt` form is what the `filter` stage writes, and it is the easiest way to keep
a curated dataset selection consistent across stages.

**Ordering matters.** `rescale` maps rows of `recons.npy` back onto MTZ files by
position, so its `input_files` must select the same datasets, in the same order, as
`preprocess` did. Using the identical value for both is the safe habit.

**Missing fields** fall back to the documented default; missing *required* fields
stop the run with a message naming each one. Whole numbers are fine where a decimal
is expected — `w_kl: 1` and `w_kl: 1.0` both work.

**`ncpu`** enables `multiprocessing` where the underlying function supports it.
One exception: `scale` can only parallelise when `when_opt: "never"`, and will tell
you when it falls back to a single process.

---

## Re-running a stage

Every stage except `standardize` checks whether its main output already exists, and
skips the work if so:

```
Found existing vae_input.npy — skipping preprocess (use --force to rerun).
```

This makes it safe to re-run a whole sequence after a failure part-way through.
To recompute anyway, pass `--force`:

```bash
valdo.pipeline preprocess config_preprocess.yaml --force
```

> **When changing the input set:** `--force` overwrites, but it does not clean up.
> Stages that write one MTZ per dataset leave files from the previous run in place,
> and the next stage will pick those stale files up. Empty the output folder yourself
> whenever the new run covers fewer datasets than the old one.

---

## Stage reference

Values shown are the defaults. Fields with no default are required.

### `standardize`

Copies raw MTZ files into a consistent `####.mtz` naming scheme, dropping rows with
NaN in `expcolumns`. Everything downstream assumes this naming.

```yaml
source_path: "/path/to/original/mtzs/"
destination_path: "/path/to/pipeline/input_mtzs/"
mtz_file_pattern: ".*([0-9]{4}).*.mtz"   # regex; capture group 1 → the 4-digit ID
expcolumns:
  - "F-obs"
  - "SIGF-obs"
ncpu: 1
```

---

### `reindex` *(optional)*

Some space groups admit more than one valid indexing convention, so datasets that
are otherwise comparable can disagree about which reflection is which. This stage
tries each reindexing operator against a reference dataset and keeps the one with
the highest correlation, writing files named `{id}_{symop}.mtz`.

```yaml
input_files: "/path/to/input_mtzs/*.mtz"
reference_file: "/path/to/input_mtzs/0001.mtz"   # a high-quality dataset
output_folder: "/path/to/reindexed/"
columns:
  - "F-obs"
  - "SIGF-obs"
wcorr: false         # true = weighted Pearson CC, recommended
cc_min_dif: 0.2      # minimum CC gap needed to call a winner; lower = more permissive
ncpu: 1
```

**Do you need this stage?** Run it once. If your space group has no ambiguity it
prints

```
No ambiguity for this spacegroup! No need to reindex!
```

and writes nothing — skip it and feed the `standardize` output straight into `scale`.
Otherwise it writes `reindex_record.pkl` with columns `file_idx`, `best_symop` and
`num_duplicates`, plus two plots of the CC differences. Datasets with
`num_duplicates > 1` were too close to call and should be inspected or dropped.

**What this changes downstream.** The reference you pass to `scale` must be the
reindexed copy of your reference dataset — look up its `best_symop` in
`reindex_record.pkl`:

```yaml
# scale config, if reindex ran and 0001 came out with best_symop 0
file_list: "/path/to/reindexed/*.mtz"
reference_mtz: "/path/to/reindexed/0001_0.mtz"

# scale config, if reindex was skipped
file_list: "/path/to/input_mtzs/*.mtz"
reference_mtz: "/path/to/input_mtzs/0001.mtz"
```

---

### `scale`

Puts every dataset on a common scale using a global anisotropic Debye-Waller factor,
so that amplitudes are comparable across crystals. Writes scaled MTZs, a
`scaling_metrics.pkl` report and two plots: a histogram of the post-scaling
correlation, and starting vs. final least-squares residual.

```yaml
file_list: "/path/to/reindexed/*.mtz"
reference_mtz: "/path/to/reindexed/0001_0.mtz"
columns:
  - "F-obs"
  - "SIGF-obs"
output_folder: "/path/to/scaled/"
prefix: ""           # prefixes scaling_metrics.pkl and the two plots, not the MTZs
when_opt: 0.2        # "all" | "never" | float in [0,1]
ncpu: 1
```

`when_opt` controls when the analytical scale is refined numerically: `"all"` always,
`"never"` not at all, or a correlation threshold below which refinement kicks in.
Numerical refinement is single-process only — with any value other than `"never"`,
`ncpu` is ignored and the stage says so.

The stage warns about datasets whose scaled amplitudes came out entirely NaN, which
means scaling diverged. Drop those before `preprocess`; the `filter` stage does it
for you.

---

### `filter` *(optional)*

Chooses which scaled datasets go into the VAE, writing a text file of MTZ paths that
`preprocess` and `rescale` accept directly as their `file_list`. Four things are
dropped:

1. the worse-refining copy of any dataset `reindex` left ambiguous (higher `Rf_final` loses),
2. datasets whose refinement R-free exceeds `max_rfree`,
3. datasets whose `F-obs-scaled` column is entirely NaN — scaling diverged,
4. datasets whose post-scaling correlation is below `min_cc`.

```yaml
refine_summary: "/path/to/refine_summary.csv"
reindex_record: "/path/to/reindexed/reindex_record.pkl"   # null if reindex was skipped
scaled_dir: "/path/to/scaled/"
metrics: null        # scaling_metrics.pkl; auto-detected inside scaled_dir when null
max_rfree: 0.45
min_cc: 0.55         # 0.0 disables the correlation check
output: "/path/to/configs/scaled_filtered_files.txt"
```

**The `refine_summary` CSV** holds refinement statistics from your own refinement
run — valdo does not produce it. Three columns are needed:

| Column | Meaning |
|--------|---------|
| `file_idx` | Dataset ID, zero-padded to 4 digits (`0042`) |
| `symop` | Symmetry operator index chosen by `reindex`; use `0` throughout if reindex was skipped |
| `Rf_final` | R-free at the end of refinement |

Rows are matched to scaled MTZs by the filename `{file_idx}_{symop}.mtz`.
`PTP1B_pipeline/parse_refine_logs.py` is a worked example that builds this CSV from
PHENIX log files; any refinement pipeline works as long as those columns are present.

Then point the next stage at the result:

```yaml
# preprocess config
file_list: "/path/to/configs/scaled_filtered_files.txt"
```

---

### `preprocess`

Works out which Miller indices are common to all datasets (the intersection) and
which appear in any of them (the union), then Z-score normalises the amplitudes into
the arrays the VAE trains on. The VAE reads intersection reflections and predicts
union reflections.

```yaml
file_list: "/path/to/scaled/*.mtz"
output_folder: "/path/to/vae/"
intersection_path: "/path/to/vae/intersection.pkl"
union_path: "/path/to/vae/union.pkl"
sigF_path: "/path/to/vae/sigF.pkl"
amplitude_col: "F-obs-scaled"
error_col: "SIGF-obs-scaled"
include_errors: true
prefix: ""           # prefixes vae_input.npy, vae_output.npy, vae_sigF.npy
```

`output_folder` also receives `union_mean.pkl` and `union_sd.pkl`, the normalisation
constants `rescale` needs later — point its `info_folder` here.

---

### `train`

Trains the VAE and saves both the model and a plot of the training and validation
loss curves. Check that plot before continuing: a KL term that collapses or a
validation loss that turns upward means the reconstruction will not be trustworthy.

```yaml
vae_input_path: "/path/to/vae/vae_input.npy"
vae_output_path: "/path/to/vae/vae_output.npy"
vae_sigF_path: "/path/to/vae/vae_sigF.npy"   # required when include_errors: true
output_path: "/path/to/vae/trained_vae.pkl"
# Architecture
latent_dim: 7
n_hidden_layers: [3, 6]      # [encoder layers, decoder layers]
n_hidden_size: 100
activation: "tanh"           # "relu" | "tanh" | "sigmoid"
# Training
epochs: 500
train_fraction: 0.8
batch_size: 100
learning_rate: 0.001
w_kl: 1.0                    # weight on the KL term
eps: 0.02
stdof: null                  # null = Gaussian likelihood; integer = Student-t d.o.f.
include_errors: true         # weight the likelihood by the measurement errors
random_seed: 42
```

`stdof` selects the likelihood: leaving it `null` gives the Gaussian ELBO, while an
integer switches to a Student-t ELBO with that many degrees of freedom, which is more
tolerant of outlier reflections. The PTP1B run used `stdof: 128` with
`activation: "relu"`; the `init` template ships those values.

Samples whose input row is entirely NaN are dropped with a warning rather than
poisoning the loss.

---

### `reconstruct`

Runs the trained VAE over every dataset to produce its apo estimate — what each
crystal's amplitudes should look like with no ligand bound.

```yaml
vae_path: "/path/to/vae/trained_vae.pkl"
vae_input_path: "/path/to/vae/vae_input.npy"
output_path: "/path/to/vae/recons/recons.npy"
ml_recon: true    # true = maximum-likelihood, deterministic; false = draw a sample
repeats: 1        # >1 draws repeatedly and saves [2, N_datasets, N_reflections] (mean, std)
```

---

### `rescale`

Undoes the Z-score normalisation and writes one MTZ per dataset carrying the VAE
estimate alongside the observations: `recons` (the estimate) and `diff`
(F_obs − F_recons), the difference signal the rest of the pipeline searches.

```yaml
recons_path: "/path/to/vae/recons/recons.npy"
intersection_path: "/path/to/vae/intersection.pkl"
union_path: "/path/to/vae/union.pkl"
input_files: "/path/to/scaled/*.mtz"   # same datasets, same order, as preprocess
info_folder: "/path/to/vae/"           # the preprocess output_folder
output_folder: "/path/to/vae/recons/"
amplitude_col: "F-obs-scaled"
ncpu: 1
```

---

### Refining your datasets

The last two stages need an apo-refined structure for every dataset: a PDB, and an
MTZ containing phases. Produce them with `valdo.refine`, which drives PHENIX in
batch and needs a `.eff` refinement config (`notebooks/refine_drug.eff` is a
starting point):

```bash
valdo.refine --pdbpath /path/to/apo_model.pdb \
             --mtzpath "/path/to/reindexed/*.mtz" \
             --output  /path/to/refined/ \
             --eff     /path/to/refine_drug.eff
```

Refine against whatever `scale` consumed — the reindexed MTZs if `reindex` ran, the
standardized ones if it did not. The output directory is what `phasing_path` and
`model_folder` point to below, and `filter` reads its R-free values from these logs.

This is also the slowest part of a screen, so start it early: it only depends on the
scaled data, and can run while `preprocess` through `rescale` are working.

---

### `add_phases_and_blobs`

Four steps in sequence, ending in the blob list that is the point of the whole
pipeline:

1. copy phases across from the apo-refined MTZs,
2. compute weights, adding `WT` and weighted-difference `WDF` columns,
3. compute extrapolated structure factors, adding `ESF_N` columns,
4. blur the weighted difference map and flood-fill it to find blobs.

```yaml
file_list: "/path/to/vae/recons/*.mtz"
phasing_path: "/path/to/refined/"      # apo-refined MTZs carrying phases
output_folder: "/path/to/vae/recons_phased/"
blob_output_folder: "/path/to/vae/blobs/"
model_folder: "/path/to/refined/"      # refined PDBs, used for blob detection
# Phase columns (PHENIX default names)
phase_2FOFC_col_in: "PH2FOFCWT"
phase_FOFC_col_in: "PHFOFCWT"
phase_2FOFC_col_out: "PH2FOFCWT"
phase_FOFC_col_out: "PHFOFCWT"
rfree_label_in: null
# Weighting
sigF_col: "SIGF-obs-scaled"
diff_col: "diff"
sigdF_pct: 95.0
absdF_pct: 99.99
# Extrapolation
F_col: "F-obs-scaled"
recons_col: "recons"
extrapolate_factors: [2, 4, 6, 8, 16]
# Blob detection
blob_diff_col: "WDF"
phase_col: "PH2FOFCWT"
cutoff: 3.5          # blob significance threshold, in sigma
radius_in_A: 4.0     # Gaussian blur radius, in Angstroms
prefix: ""
ncpu: 1
```

`cutoff` is the main knob on sensitivity: lower finds more blobs and more noise.

---

### `tag_blobs`

Annotates each blob with what it sits next to, then filters the list down to
candidate binding events. Blobs near a known false-positive site are tagged and
removed, and symmetry-equivalent copies of the same blob are collapsed to one.

```yaml
blob_stats_path: "/path/to/vae/blobs/blob_stats.pkl"
model_folder: "/path/to/refined/"
mtz_folder: "/path/to/vae/recons_phased/"
output_folder: "/path/to/vae/blobs/"
# Focal residue: blobs within focal_radius Å are tagged and dropped from the
# filtered output (Cys215 in PTP1B is a known false-positive source)
focal_seqid: 215
focal_tag_name: "cys215"
focal_radius: 5.0
# Optional: flat folder of known bound-state PDBs, for validation runs
# bound_models_folder: "/path/to/bound_models/"
ncpu: 1
```

`bound_models_folder` is only for validating the method against a screen whose hits
you already know: given those structures, blobs are marked with whether their dataset
is a true binder (`bound`) and whether the blob overlaps the ligand (`ligand`). Omit
it for a real screen and both columns are set to 0.

---

## What you end up with

`tag_blobs` writes two DataFrames:

- **`blob_stats_tagged.pkl`** — every blob found, with its peak height, score,
  centroid, volume, radius, and the tag columns described above.
- **`filtered_blob_stats_tagged.pkl`** — the same table restricted to blobs that are
  not near the focal residue and not duplicates. This is the candidate hit list.

Rank it by peak height or score and inspect the top blobs in your favourite map
viewer. If you ran with `bound_models_folder`, `PTP1B_pipeline/plot_auc.py` shows how
to turn the `ligand` column into an ROC curve and AUC to measure how well the ranking
recovers the known binders.
