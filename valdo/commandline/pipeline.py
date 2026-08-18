"""
Run a single stage of the valdo ML pipeline from a YAML/JSON config file.

Usage
-----
    valdo.pipeline <stage_name> <config.yaml>
    valdo.pipeline init <stage_name>     # print a template config to stdout

Available stages
----------------
    standardize          Rename and copy raw MTZ files to a standard naming scheme
    reindex              Correct indexing ambiguity if present (optional)
    scale                Anisotropically scale all datasets to a reference
    filter               Filter scaled datasets by R-free and post-scaling CC; write file list
    preprocess           Build VAE input/output arrays (intersection, union, Z-score)
    train                Train the VAE model
    reconstruct          Run VAE reconstruction on all samples
    rescale              Reverse standardisation and compute difference columns
    add_phases_and_blobs Add phases, weights, extrapolation and detect blobs
    tag_blobs            Tag blobs by residue proximity and ligand overlap

Examples
--------
    valdo.pipeline init train > config_train.yaml
    valdo.pipeline train config_train.yaml
"""

import argparse
import glob
import os
import sys

from valdo.commandline.config import load_config, validate_config, expand_glob_field


# ---------------------------------------------------------------------------
# Stage runners
# ---------------------------------------------------------------------------

def run_standardize(cfg):
    import valdo.helper as helper
    os.makedirs(cfg["destination_path"], exist_ok=True)
    helper.standardize_input_mtzs(
        source_path=cfg["source_path"],
        destination_path=cfg["destination_path"],
        mtz_file_pattern=cfg["mtz_file_pattern"],
        expcolumns=cfg["expcolumns"],
        ncpu=cfg["ncpu"],
    )


def run_reindex(cfg):
    import pandas as pd
    import matplotlib.pyplot as plt
    import valdo.reindex as reindex

    os.makedirs(cfg["output_folder"], exist_ok=True)
    record_path = os.path.join(cfg["output_folder"], "reindex_record.pkl")

    if os.path.isfile(record_path) and not cfg.get("_force"):
        print(f"Found existing reindex_record.pkl — skipping reindexing (use --force to rerun).")
        df_record = pd.read_pickle(record_path)
    else:
        file_list = expand_glob_field(cfg["input_files"])
        if cfg["ncpu"] > 1:
            df_record = reindex.reindex_files_pool(
                input_files=file_list,
                reference_file=cfg["reference_file"],
                output_folder=cfg["output_folder"],
                columns=cfg["columns"],
                wcorr=cfg["wcorr"],
                cc_min_dif=cfg["cc_min_dif"],
                ncpu=cfg["ncpu"],
            )
        else:
            df_record = reindex.reindex_files(
                input_files=file_list,
                reference_file=cfg["reference_file"],
                output_folder=cfg["output_folder"],
                columns=cfg["columns"],
                wcorr=cfg["wcorr"],
                cc_min_dif=cfg["cc_min_dif"],
            )

    if df_record is None:
        print("No indexing ambiguity detected — skipping validation plots.")
        return

    # Identify CC columns (CC_symop0, CC_symop1, ...)
    cc_cols = sorted([c for c in df_record.columns if c.startswith("CC_symop")])
    if len(cc_cols) < 2:
        print("Only one symop found — skipping validation plots.")
        return

    cc_dif  = df_record[cc_cols[1]] - df_record[cc_cols[0]]
    cc_max  = df_record[cc_cols].max(axis=1)

    # Plot 1: histogram of CC difference
    fig, ax = plt.subplots()
    ax.hist(cc_dif, bins=100)
    ax.set_xlabel(f"CC ({cc_cols[1]}) - CC ({cc_cols[0]})")
    ax.set_ylabel("Count per bin")
    ax.grid(True)
    hist_path = os.path.join(cfg["output_folder"], "reindex_cc_diff_histogram.png")
    fig.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {hist_path}")

    # Plot 2: scatter of max CC vs CC difference
    fig, ax = plt.subplots()
    ax.plot(cc_max, cc_dif, ".", markersize=3, alpha=0.6)
    ax.set_xlabel("Max CC over symops")
    ax.set_ylabel(f"CC ({cc_cols[1]}) - CC ({cc_cols[0]})")
    ax.grid(True)
    scatter_path = os.path.join(cfg["output_folder"], "reindex_cc_diff_scatter.png")
    fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {scatter_path}")

    # Summary of ambiguous datasets
    ambiguous = df_record[df_record["num_duplicates"] > 1]
    print(f"\nDatasets with unresolved ambiguity (num_duplicates > 1): {len(ambiguous)}")
    if len(ambiguous) > 0:
        print(ambiguous[["file_idx", "best_symop", "num_duplicates"] + cc_cols].to_string(index=False))


def run_scale(cfg):
    import gemmi
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    from valdo.scaling import Scaler, Scaler_pool

    os.makedirs(cfg["output_folder"], exist_ok=True)
    metrics_path = os.path.join(cfg["output_folder"], cfg["prefix"] + "scaling_metrics.pkl")

    if os.path.isfile(metrics_path) and not cfg.get("_force"):
        print(f"Found existing scaling_metrics.pkl — skipping scaling (use --force to rerun).")
        metrics_df = pd.read_pickle(metrics_path)
    else:
        file_list = expand_glob_field(cfg["file_list"])
        # Scaler_pool does not support numerical optimisation (when_opt).
        # Fall back to single-process Scaler whenever when_opt is not "never".
        use_pool = cfg["ncpu"] > 1 and cfg["when_opt"] == "never"
        if use_pool:
            scaler = Scaler_pool(
                reference_mtz=cfg["reference_mtz"],
                columns=cfg["columns"],
                ncpu=cfg["ncpu"],
            )
            scaler.batch_scaling(
                mtz_path_list=file_list,
                outputmtz_path=cfg["output_folder"],
                prefix=cfg["prefix"],
            )
        else:
            if cfg["ncpu"] > 1 and cfg["when_opt"] != "never":
                print(f"Note: Scaler_pool does not support when_opt; "
                      f"using single-process Scaler with when_opt={cfg['when_opt']!r}")
            scaler = Scaler(
                reference_mtz=cfg["reference_mtz"],
                columns=cfg["columns"],
            )
            scaler.batch_scaling(
                mtz_path_list=file_list,
                outputmtz_path=cfg["output_folder"],
                prefix=cfg["prefix"],
                when_opt=cfg["when_opt"],
            )
        metrics_df = pd.read_pickle(metrics_path)

    # Warn about scaled files where the amplitude column is entirely NaN (scaling diverged)
    amplitude_col = cfg["columns"][0] + "-scaled"
    all_nan_files = []
    for sf in sorted(glob.glob(os.path.join(cfg["output_folder"], "*.mtz"))):
        try:
            col = gemmi.read_mtz_file(sf).column_with_label(amplitude_col)
            if col is not None and not np.isfinite(np.array(col)).any():
                all_nan_files.append(sf)
        except Exception:
            pass
    if all_nan_files:
        print(f"\nWarning: {len(all_nan_files)} scaled file(s) have all-NaN '{amplitude_col}' "
              f"(scaling diverged) — exclude these before preprocess:")
        for f in all_nan_files:
            print(f"  {f}")

    # Plot 1: histogram of end_corr
    fig, ax = plt.subplots()
    ax.hist(metrics_df["end_corr"].to_numpy(), bins=20)
    ax.set_xlabel("Correlation after scaling")
    ax.set_ylabel("Count per bin")
    ax.grid(True)
    hist_path = os.path.join(cfg["output_folder"], cfg["prefix"] + "scaling_end_corr_histogram.png")
    fig.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {hist_path}")

    # Plot 2: start_LS vs end_LS with diagonal
    fig, ax = plt.subplots()
    ax.plot(metrics_df["start_LS"].to_numpy(), metrics_df["end_LS"].to_numpy(), ".", alpha=0.25)
    xlim = ax.get_xlim()
    ax.plot(xlim, xlim, "r-")
    ax.set_xlim(xlim)
    ax.set_xlabel("Starting LS")
    ax.set_ylabel("Final LS")
    ax.grid(True)
    scatter_path = os.path.join(cfg["output_folder"], cfg["prefix"] + "scaling_LS_scatter.png")
    fig.savefig(scatter_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {scatter_path}")


def run_filter(cfg):
    import gemmi
    import numpy as np
    import pandas as pd

    out_path = cfg["output"]
    if os.path.isfile(out_path) and not cfg.get("_force"):
        print(f"Found existing {out_path} — skipping filter (use --force to rerun).")
        return

    # --- Load refinement summary ---
    df = pd.read_csv(cfg["refine_summary"])
    df["file_idx"] = df["file_idx"].astype(str).str.zfill(4)
    df["symop"]    = df["symop"].astype(str)

    drop = set()

    # --- Resolve reindexing ambiguity: drop the worse symop for ambiguous datasets ---
    if cfg["reindex_record"] and os.path.isfile(cfg["reindex_record"]):
        import pickle
        with open(cfg["reindex_record"], "rb") as f:
            reindex_record = pickle.load(f)
        if reindex_record is not None:
            ambiguous = reindex_record[reindex_record["num_duplicates"] > 1]["file_idx"].tolist()
            ambiguous = [str(x).zfill(4) for x in ambiguous]
            df_ambig  = df[df["file_idx"].isin(ambiguous)]
            worse     = df_ambig.loc[df_ambig.groupby("file_idx")["Rf_final"].idxmax()]
            drop1     = set((worse["file_idx"] + "_" + worse["symop"] + ".mtz").tolist())
            print(f"Dropping {len(drop1)} worse-symop duplicates from {len(ambiguous)} ambiguous datasets")
            drop |= drop1
    else:
        print("No reindex_record found; skipping ambiguity resolution")

    # --- Filter by R-free ---
    bad_R = df[df["Rf_final"] > cfg["max_rfree"]]
    drop2 = set((bad_R["file_idx"] + "_" + bad_R["symop"] + ".mtz").tolist())
    print(f"Dropping {len(drop2)} datasets with Rf_final > {cfg['max_rfree']}")
    drop |= drop2

    # --- Collect scaled files ---
    all_scaled = sorted(glob.glob(os.path.join(cfg["scaled_dir"], "*.mtz")))
    if not all_scaled:
        print(f"Error: no MTZ files found in {cfg['scaled_dir']}", file=sys.stderr)
        sys.exit(1)
    file_list = [f for f in all_scaled if os.path.basename(f) not in drop]

    # --- Drop files where F-obs-scaled is entirely NaN (scaling diverged) ---
    nan_files = []
    for mtz_path in file_list:
        try:
            col = gemmi.read_mtz_file(mtz_path).column_with_label("F-obs-scaled")
            if col is not None and not np.isfinite(np.array(col)).any():
                nan_files.append(mtz_path)
        except Exception:
            pass
    if nan_files:
        print(f"Dropping {len(nan_files)} file(s) with all-NaN F-obs-scaled: "
              + ", ".join(os.path.basename(f) for f in nan_files))
        file_list = [f for f in file_list if f not in nan_files]

    # --- Filter by post-scaling CC ---
    if cfg["min_cc"] > 0.0:
        metrics_path = cfg["metrics"]
        if not metrics_path:
            candidates = glob.glob(os.path.join(cfg["scaled_dir"], "*scaling_metrics.pkl"))
            metrics_path = candidates[0] if candidates else None
        if metrics_path and os.path.isfile(metrics_path):
            metrics = pd.read_pickle(metrics_path)
            low_cc  = set(metrics[(metrics["end_corr"] < cfg["min_cc"]) |
                                   metrics["end_corr"].isnull()]["file"].tolist())
            before  = len(file_list)
            file_list = [f for f in file_list
                         if os.path.basename(f).replace(".mtz", "") not in low_cc]
            print(f"Dropping {before - len(file_list)} datasets with post-scaling CC < {cfg['min_cc']}")
        else:
            print("Warning: min_cc set but no scaling_metrics.pkl found; skipping CC filter",
                  file=sys.stderr)

    # --- Write output ---
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write("\n".join(file_list) + "\n")
    print(f"\n{len(file_list)} datasets written to {out_path}")


def run_preprocess(cfg):
    import valdo.preprocessing as preprocessing
    sentinel = os.path.join(cfg["output_folder"], cfg["prefix"] + "vae_input.npy")
    if os.path.isfile(sentinel) and not cfg.get("_force"):
        print(f"Found existing vae_input.npy — skipping preprocess (use --force to rerun).")
        return
    file_list = expand_glob_field(cfg["file_list"])
    os.makedirs(cfg["output_folder"], exist_ok=True)
    preprocessing.find_intersection(
        input_files=file_list,
        output_path=cfg["intersection_path"],
        amplitude_col=cfg["amplitude_col"],
    )
    preprocessing.find_union(
        input_files=file_list,
        output_path=cfg["union_path"],
        sigF_path=cfg["sigF_path"],
        amplitude_col=cfg["amplitude_col"],
        error_col=cfg["error_col"],
        include_errors=cfg["include_errors"],
    )
    preprocessing.generate_vae_io(
        intersection_path=cfg["intersection_path"],
        union_path=cfg["union_path"],
        sigF_path=cfg["sigF_path"],
        io_folder=cfg["output_folder"],
        prefix=cfg["prefix"],
        include_errors=cfg["include_errors"],
    )


def run_train(cfg):
    import numpy as np
    import torch
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import valdo
    from valdo.helper import try_gpu

    if os.path.isfile(cfg["output_path"]) and not cfg.get("_force"):
        print(f"Found existing {cfg['output_path']} — skipping train (use --force to rerun).")
        return

    x = np.load(cfg["vae_input_path"])
    y = np.load(cfg["vae_output_path"])
    if cfg["include_errors"] and cfg["vae_sigF_path"]:
        e = np.load(cfg["vae_sigF_path"])
    else:
        e = np.zeros_like(y)

    # Drop samples whose entire input row is NaN (failed datasets that slipped through filtering)
    valid = ~np.isnan(x).all(axis=1)
    if not valid.all():
        n_dropped = int((~valid).sum())
        bad_idx = np.where(~valid)[0].tolist()
        print(f"Warning: dropping {n_dropped} sample(s) with all-NaN input: indices {bad_idx}")
        x, y, e = x[valid], y[valid], e[valid]

    torch.manual_seed(cfg["random_seed"])
    rng = np.random.default_rng(cfg["random_seed"])
    idx = rng.permutation(x.shape[0])
    n_train = int(x.shape[0] * cfg["train_fraction"])
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    device = try_gpu()

    activation_map = {"relu": torch.relu, "tanh": torch.tanh, "sigmoid": torch.sigmoid}
    activation = activation_map.get(cfg["activation"], torch.relu)

    def to_tensor(arr):
        return torch.tensor(arr, dtype=torch.float32).to(device)

    x_train, x_val = to_tensor(x[train_idx]), to_tensor(x[val_idx])
    y_train, y_val = to_tensor(y[train_idx]), to_tensor(y[val_idx])
    e_train, e_val = to_tensor(e[train_idx]), to_tensor(e[val_idx])

    vae_model = valdo.VAE(
        n_dim_i=x.shape[1],
        n_dim_o=y.shape[1],
        n_dim_latent=cfg["latent_dim"],
        n_hidden_layers=cfg["n_hidden_layers"],
        n_hidden_size=cfg["n_hidden_size"],
        activation=activation,
        device=device,
    )
    # Clamp logvar in the encoder output to prevent exp(logvar) overflow in the KL term.
    # The encoder outputs [z_mean | z_log_var]; we clamp only the logvar half.
    _dim_z = vae_model.dim_z
    def _clamp_logvar_hook(module, input, output):
        return torch.cat([output[:, :_dim_z], output[:, _dim_z:].clamp(-10, 10)], dim=1)
    vae_model.encoder.register_forward_hook(_clamp_logvar_hook)

    optim = torch.optim.Adam(vae_model.parameters(), lr=cfg["learning_rate"])
    _params = list(vae_model.parameters())
    _orig_step = optim.step
    def _clipped_step(*args, **kwargs):
        torch.nn.utils.clip_grad_norm_(_params, max_norm=1.0)
        return _orig_step(*args, **kwargs)
    optim.step = _clipped_step

    vae_model.train(
        x_train, y_train, e_train, optim,
        x_val=x_val, y_val=y_val, e_val=e_val,
        epochs=cfg["epochs"],
        batch_size=cfg["batch_size"],
        w_kl=cfg["w_kl"],
        eps=cfg["eps"],
        include_errors=cfg["include_errors"],
        stdof=cfg["stdof"],
    )
    output_dir = os.path.dirname(cfg["output_path"])
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    vae_model.save(cfg["output_path"])

    loss_array = np.array(vae_model.loss_train)
    plot_dir = output_dir or "."
    fig, axs = plt.subplots(3, 1, figsize=(6, 8))
    labels = [("Total Loss", 0, 3), ("NLL", 1, 4), ("KL Divergence", 2, 5)]
    for ax, (name, ti, vi) in zip(axs, labels):
        if loss_array.shape[1] > vi:
            ax.plot(loss_array[:, ti], label=f"{name}, Training")
            ax.plot(loss_array[:, vi], label=f"{name}, Validation")
        else:
            ax.plot(loss_array[:, ti], label=f"{name}, Training")
        ax.set_xlabel("Steps")
        ax.legend()
        ax.grid()
    plt.tight_layout()
    plot_path = os.path.join(plot_dir, "vae_loss_curves.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Loss curves saved to {plot_path}")


def run_reconstruct(cfg):
    import numpy as np
    import torch
    import valdo

    # output_path is saved as .npy; np.save appends .npy if not already present
    out_path = cfg["output_path"]
    if not out_path.endswith(".npy"):
        out_path += ".npy"
    if os.path.isfile(out_path) and not cfg.get("_force"):
        print(f"Found existing {out_path} — skipping reconstruct (use --force to rerun).")
        return

    vae_model = valdo.VAE.load(cfg["vae_path"])
    x = np.load(cfg["vae_input_path"])
    input_tensor = torch.tensor(x, dtype=torch.float32).to(vae_model.device)

    repeats = cfg["repeats"]
    recons = vae_model.reconstruct(input_tensor, ml_recon=cfg["ml_recon"], repeats=repeats)

    if repeats > 1:
        stacked = torch.stack(recons, dim=0)
        out = np.stack([
            stacked.mean(0).detach().cpu().numpy(),
            stacked.std(0).detach().cpu().numpy(),
        ])
    else:
        out = recons.detach().cpu().numpy()

    output_dir = os.path.dirname(cfg["output_path"])
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    np.save(cfg["output_path"], out)


def run_rescale(cfg):
    import valdo.preprocessing as preprocessing
    existing = glob.glob(os.path.join(cfg["output_folder"], "*.mtz"))
    if existing and not cfg.get("_force"):
        print(f"Found {len(existing)} existing MTZ(s) in output_folder — skipping rescale (use --force to rerun).")
        return
    file_list = expand_glob_field(cfg["input_files"])
    os.makedirs(cfg["output_folder"], exist_ok=True)
    kwargs = dict(
        recons_path=cfg["recons_path"],
        intersection_path=cfg["intersection_path"],
        union_path=cfg["union_path"],
        input_files=file_list,
        info_folder=cfg["info_folder"],
        output_folder=cfg["output_folder"],
        amplitude_col=cfg["amplitude_col"],
    )
    if cfg["ncpu"] > 1:
        preprocessing.rescale_pool(**kwargs, ncpu=cfg["ncpu"])
    else:
        preprocessing.rescale(**kwargs)


def run_add_phases_and_blobs(cfg):
    import numpy as np
    import reciprocalspaceship as rs
    import valdo.helper as helper
    import valdo.blobs as blobs

    sentinel = os.path.join(cfg["blob_output_folder"], cfg["prefix"] + "blob_stats.pkl")
    if os.path.isfile(sentinel) and not cfg.get("_force"):
        print(f"Found existing blob_stats.pkl — skipping add_phases_and_blobs (use --force to rerun).")
        return

    file_list = expand_glob_field(cfg["file_list"])
    os.makedirs(cfg["output_folder"], exist_ok=True)
    os.makedirs(cfg["blob_output_folder"], exist_ok=True)

    # Step 1: add phases
    phase_kwargs = dict(
        apo_mtzs_path=cfg["phasing_path"],
        vae_reconstructed_with_phases_path=cfg["output_folder"],
        phase_2FOFC_col_out=cfg["phase_2FOFC_col_out"],
        phase_FOFC_col_out=cfg["phase_FOFC_col_out"],
        phase_2FOFC_col_in=cfg["phase_2FOFC_col_in"],
        phase_FOFC_col_in=cfg["phase_FOFC_col_in"],
        rfree_label_in=cfg["rfree_label_in"],
    )
    if cfg["ncpu"] > 1:
        helper.add_phases_pool(file_list=file_list, ncpu=cfg["ncpu"], **phase_kwargs)
    else:
        helper.add_phases(file_list=file_list, **phase_kwargs)

    # Re-glob to get the files that had phases successfully added
    phased_files = sorted(glob.glob(os.path.join(cfg["output_folder"], "*.mtz")))
    if not phased_files:
        print("Warning: no MTZ files found in output_folder after add_phases.", file=sys.stderr)
        return

    # Step 2: add weights (in-place, adds WT and WDF columns)
    helper.add_weights(
        file_list=phased_files,
        sigF_col=cfg["sigF_col"],
        diff_col=cfg["diff_col"],
        sigdF_pct=cfg["sigdF_pct"],
        absdF_pct=cfg["absdF_pct"],
        ncpu=cfg["ncpu"],
    )

    # Step 3: extrapolate (in-place, adds ESF_N columns)
    helper.extrapolate(
        file_list=phased_files,
        F_col=cfg["F_col"],
        sigF_col=cfg["sigF_col"],
        recons_col=cfg["recons_col"],
        extrapolate_factors=cfg["extrapolate_factors"],
        ncpu=cfg["ncpu"],
    )

    # Step 3b: drop NaN rows produced by extrapolation (mirrors pipeline_doeke.py)
    for f in phased_files:
        try:
            ds = rs.read_mtz(f)
            if np.count_nonzero(ds.isnull()) > 0:
                ds.dropna().write_mtz(f)
        except Exception as e:
            print(f"Warning: NaN cleanup failed for {f}: {e}", file=sys.stderr)

    # Step 4: detect blobs
    blob_kwargs = dict(
        input_files=phased_files,
        model_folder=cfg["model_folder"],
        diff_col=cfg["blob_diff_col"],
        phase_col=cfg["phase_col"],
        output_folder=cfg["blob_output_folder"],
        prefix=cfg["prefix"],
        cutoff=cfg["cutoff"],
        radius_in_A=cfg["radius_in_A"],
    )
    if cfg["ncpu"] > 1:
        blobs.generate_blobs_pool(**blob_kwargs, ncpu=cfg["ncpu"])
    else:
        blobs.generate_blobs(**blob_kwargs)


def run_tag_blobs(cfg):
    import re
    import shutil
    import tempfile
    import pandas as pd
    import valdo.tag as tag

    sentinel = os.path.join(cfg["output_folder"], "filtered_blob_stats_tagged.pkl")
    if os.path.isfile(sentinel) and not cfg.get("_force"):
        print(f"Found existing filtered_blob_stats_tagged.pkl — skipping tag_blobs (use --force to rerun).")
        return

    blob_df = pd.read_pickle(cfg["blob_stats_path"])

    # Tag blobs near the focal residue (e.g. Cys215) using per-dataset refined PDBs
    blob_df = tag.tag_blobs_around_seqid(
        blob_df,
        cfg["model_folder"],
        radius=cfg["focal_radius"],
        tag=cfg["focal_tag_name"],
        focal_seqid=cfg["focal_seqid"],
        ncpu=cfg["ncpu"],
    )

    # Optionally tag ligand-containing blobs using known bound structures.
    # If bound_models_folder is not provided, skip this step (bound=0, ligand=0 for all).
    if cfg.get("bound_models_folder"):
        # bound_models_folder contains flat PDB files named like y0049_og_superposed.pdb.
        # Extract the numeric ID (strip leading alpha prefix).
        bound_models_folder = cfg["bound_models_folder"].rstrip("/") + "/"
        bound_samples = set()
        pdb_map = {}   # numeric_id -> absolute pdb path
        for pdb_path in glob.glob(os.path.join(bound_models_folder, "*.pdb")):
            stem = os.path.basename(pdb_path)
            m = re.match(r"[a-zA-Z]*(\d+)", stem)
            if m:
                num = m.group(1)
                bound_samples.add(num)
                pdb_map[num] = pdb_path
        blob_df["bound"] = blob_df["sample"].apply(
            lambda x: 1 if x.split("_")[0] in bound_samples else 0
        )
        print(f"Marked {blob_df['bound'].sum()} blobs from {len(bound_samples)} bound datasets")

        # tag_lig_blobs expects files named {number}.pdb — create a temp dir with symlinks
        tmp_dir = tempfile.mkdtemp(prefix="valdo_bound_pdbs_")
        try:
            for num, src in pdb_map.items():
                os.symlink(src, os.path.join(tmp_dir, f"{num}.pdb"))
            blob_df = tag.tag_lig_blobs(blob_df, tmp_dir + "/", ncpu=cfg["ncpu"])
        finally:
            shutil.rmtree(tmp_dir, ignore_errors=True)
    else:
        blob_df["bound"] = 0
        blob_df["ligand"] = 0
        print("No bound_models_folder provided; skipping ligand tagging (bound=0, ligand=0)")

    blob_df = tag.determine_locations(blob_df, cfg["mtz_folder"], ncpu=cfg["ncpu"])
    blob_df = tag.mark_duplicates(blob_df)

    tag_name = cfg["focal_tag_name"]
    filtered = blob_df[(blob_df[tag_name] == 0) & (blob_df["duplicate"] == 0)]

    os.makedirs(cfg["output_folder"], exist_ok=True)
    blob_df.to_pickle(os.path.join(cfg["output_folder"], "blob_stats_tagged.pkl"))
    filtered.to_pickle(os.path.join(cfg["output_folder"], "filtered_blob_stats_tagged.pkl"))
    print(f"Saved {len(blob_df)} tagged blobs and {len(filtered)} filtered blobs to {cfg['output_folder']}")


# ---------------------------------------------------------------------------
# Stage registry
# ---------------------------------------------------------------------------

STAGE_REGISTRY = {
    "standardize":          run_standardize,
    "reindex":              run_reindex,
    "scale":                run_scale,
    "filter":               run_filter,
    "preprocess":           run_preprocess,
    "train":                run_train,
    "reconstruct":          run_reconstruct,
    "rescale":              run_rescale,
    "add_phases_and_blobs": run_add_phases_and_blobs,
    "tag_blobs":            run_tag_blobs,
}


# ---------------------------------------------------------------------------
# Template YAML strings for `valdo.pipeline init <stage>`
# ---------------------------------------------------------------------------

_TEMPLATES = {
    "standardize": """\
# valdo.pipeline standardize config
# Copies and renames raw MTZ files to a standard ####.mtz naming scheme.
source_path: "/path/to/original/mtzs/"        # directory containing raw MTZ files
destination_path: "/path/to/pipeline/input_mtzs/"
mtz_file_pattern: ".*([0-9]{4}).*.mtz"         # regex with one capture group for the 4-digit ID
expcolumns:                                     # columns that must be present (NaN rows are dropped)
  - "F-obs"
  - "SIGF-obs"
ncpu: 1                                         # optional, default 1
""",
    "reindex": """\
# valdo.pipeline reindex config
# Corrects indexing ambiguity if present.
# This stage can be SKIPPED if no ambiguity is detected; the function
# will print "No ambiguity for this spacegroup!" and write no output files.
# When skipping, pass the standardize output directly to scale.
input_files: "/path/to/input_mtzs/*.mtz"       # glob or explicit list
reference_file: "/path/to/input_mtzs/0001.mtz" # high-quality reference dataset
output_folder: "/path/to/reindexed/"
columns:
  - "F-obs"
  - "SIGF-obs"
wcorr: true                                     # use weighted Pearson correlation (recommended)
cc_min_dif: 0.2                                 # min CC gap to resolve ambiguity
ncpu: 1
""",
    "scale": """\
# valdo.pipeline scale config
# Anisotropically scales all datasets to a reference using Debye-Waller factors.
# Note: filter file_list manually before this stage if needed (R-factor, CC, etc.)
# If reindex was skipped: set file_list to standardize output and reference_mtz
#   to the standardized reference (e.g. 0001.mtz).
# If reindex was run: set file_list to reindexed/*.mtz and reference_mtz to
#   the reindexed reference (e.g. 0001_0.mtz, using best_symop from reindex_record.pkl).
file_list: "/path/to/reindexed/*.mtz"          # glob or explicit list
reference_mtz: "/path/to/reindexed/0001_0.mtz"
columns:
  - "F-obs"
  - "SIGF-obs"
output_folder: "/path/to/scaled/"
prefix: ""                                      # prefixes scaling_metrics.pkl and the two plots, not the MTZs
when_opt: "all"                                 # "all", "never", or float threshold [0.0, 1.0]
ncpu: 1
""",
    "filter": """\
# valdo.pipeline filter config
# Filters scaled MTZ files by R-free and post-scaling CC; writes a file list for preprocess.
# Run after `scale` and before `preprocess`.
refine_summary: "/path/to/refine_1nwl/refine_summary.csv"   # from parse_refine_logs.py
reindex_record: "/path/to/reindexed/reindex_record.pkl"     # null if reindex was skipped
scaled_dir: "/path/to/scaled/"
metrics: null                  # path to scaling_metrics.pkl; auto-detected in scaled_dir if null
max_rfree: 0.45                # discard datasets with Rf_final above this
min_cc: 0.55                   # discard datasets with post-scaling CC below this (0.0 = no filter)
output: "/path/to/configs/scaled_filtered_files.txt"
""",
    "preprocess": """\
# valdo.pipeline preprocess config
# Builds VAE input/output numpy arrays via Z-score normalisation.
# Outputs: {output_folder}{prefix}vae_input.npy, vae_output.npy, vae_sigF.npy
# Also saves union_mean.pkl and union_sd.pkl in output_folder (needed by rescale).
file_list: "/path/to/scaled/*.mtz"
output_folder: "/path/to/vae/"
intersection_path: "/path/to/vae/intersection.pkl"
union_path: "/path/to/vae/union.pkl"
sigF_path: "/path/to/vae/sigF.pkl"
amplitude_col: "F-obs-scaled"
error_col: "SIGF-obs-scaled"
include_errors: true
prefix: ""                                      # prefix for vae_input.npy etc.
""",
    "train": """\
# valdo.pipeline train config
# Trains a VAE model on the preprocessed structure factor amplitudes.
vae_input_path: "/path/to/vae/vae_input.npy"
vae_output_path: "/path/to/vae/vae_output.npy"
vae_sigF_path: "/path/to/vae/vae_sigF.npy"    # required when include_errors: true
output_path: "/path/to/vae/trained_vae.pkl"
# Architecture
latent_dim: 7
n_hidden_layers: [3, 6]                         # [encoder_layers, decoder_layers]
n_hidden_size: 100
# Training
epochs: 500
train_fraction: 0.8
batch_size: 100
learning_rate: 0.001
w_kl: 1.0
eps: 0.02
stdof: 128                                      # null = Gaussian ELBO; integer = Student-t df (128 recommended)
include_errors: true
random_seed: 11231
activation: "relu"
""",
    "reconstruct": """\
# valdo.pipeline reconstruct config
# Runs VAE reconstruction to obtain structure factor estimates for all samples.
vae_path: "/path/to/vae/trained_vae.pkl"
vae_input_path: "/path/to/vae/vae_input.npy"
output_path: "/path/to/vae/recons/recons.npy"
ml_recon: true                                  # true = MAP (deterministic); false = sample
repeats: 1                                      # >1 saves mean+std array of shape [2, N, M]
""",
    "rescale": """\
# valdo.pipeline rescale config
# Reverses Z-score normalisation to recover original-scale amplitudes and
# computes difference columns: recons and diff, plus SIG_recons when the
# reconstruct stage was run with repeats > 1.
# info_folder must be the same as the preprocess output_folder
# (it contains union_mean.pkl and union_sd.pkl).
recons_path: "/path/to/vae/recons/recons.npy"
intersection_path: "/path/to/vae/intersection.pkl"
union_path: "/path/to/vae/union.pkl"
input_files: "/path/to/scaled/*.mtz"           # must match the order used in preprocess
info_folder: "/path/to/vae/"
output_folder: "/path/to/vae/recons/"
amplitude_col: "F-obs-scaled"
ncpu: 1
""",
    "add_phases_and_blobs": """\
# valdo.pipeline add_phases_and_blobs config
# Sequential steps: add phases from refinement → add weights → extrapolate → detect blobs.
file_list: "/path/to/vae/recons/*.mtz"
phasing_path: "/path/to/refined/"              # directory with apo-refined MTZ files containing phases
output_folder: "/path/to/vae/recons_phased/"
blob_output_folder: "/path/to/vae/blobs/"
model_folder: "/path/to/refined/"              # directory with refined PDB models for blob detection
# Phase column names (from PHENIX refinement output)
phase_2FOFC_col_in: "PH2FOFCWT"
phase_FOFC_col_in: "PHFOFCWT"
phase_2FOFC_col_out: "PH2FOFCWT"
phase_FOFC_col_out: "PHFOFCWT"
rfree_label_in: null                           # input R-free column name, or null
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
blob_diff_col: "WDF"                           # weighted difference column
phase_col: "PH2FOFCWT"
cutoff: 3.5                                    # blob significance threshold (sigma)
radius_in_A: 4.0                               # Gaussian blur radius
prefix: ""
ncpu: 1
""",
    "tag_blobs": """\
# valdo.pipeline tag_blobs config
# Tags blobs by proximity to a focal residue and known ligands, removes duplicates,
# and saves both the full tagged DataFrame and a filtered one.
blob_stats_path: "/path/to/vae/blobs/blob_stats.pkl"
model_folder: "/path/to/refined/"       # per-dataset PDB files, e.g. refine_output/
mtz_folder: "/path/to/vae/recons_phased/"
# Focal residue to exclude (e.g. Cys215 in PTP1B — a known false-positive source)
focal_seqid: 215
focal_tag_name: "cys215"
focal_radius: 5.0
output_folder: "/path/to/vae/blobs/"
# Optional: flat folder of known bound-state PDB files (e.g. y0049_og_superposed.pdb).
# If omitted, ligand tagging is skipped and bound/ligand columns are set to 0.
# bound_models_folder: "/path/to/bound_models/"
ncpu: 1
""",
}


def _print_template(stage_name):
    if stage_name not in _TEMPLATES:
        valid = ", ".join(sorted(_TEMPLATES))
        print(f"Error: no template for stage '{stage_name}'. Valid stages: {valid}", file=sys.stderr)
        sys.exit(1)
    print(_TEMPLATES[stage_name], end="")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

class _ArgumentParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__(
            formatter_class=argparse.RawTextHelpFormatter,
            description=__doc__,
        )
        self.add_argument(
            "stage",
            type=str,
            help="Pipeline stage name, or 'init' to print a template config.",
        )
        self.add_argument(
            "config",
            type=str,
            help="Path to YAML/JSON config file (or stage name when using 'init').",
        )
        self.add_argument(
            "--force",
            action="store_true",
            default=False,
            help="Force rerun even if output files already exist.",
        )


def main():
    args = _ArgumentParser().parse_args()

    if args.stage == "init":
        _print_template(args.config)
        return

    if args.stage not in STAGE_REGISTRY:
        valid = ", ".join(sorted(STAGE_REGISTRY))
        print(f"Error: unknown stage '{args.stage}'.\nValid stages: {valid}", file=sys.stderr)
        sys.exit(1)

    cfg = load_config(args.config)
    cfg = validate_config(args.stage, cfg)
    cfg["_force"] = args.force
    STAGE_REGISTRY[args.stage](cfg)
