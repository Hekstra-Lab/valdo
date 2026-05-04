#!/usr/bin/env python
"""Filter scaled MTZ files by R-free and scaling CC, write a file list for downstream steps.

Usage:
    python filter_datasets.py [options]

This script reads refine_summary.csv and scaling_metrics.pkl, removes poor-quality
datasets, resolves reindexing ambiguity, and writes the surviving paths to a .txt file.

Options:
    --refine-summary    Path to refine_summary.csv  (default: refine_1nwl/refine_summary.csv)
    --reindex-record    Path to reindex_record.pkl  (default: reindexed/reindex_record.pkl)
    --scaled-dir        Directory of scaled MTZ files (default: scaled/)
    --metrics           Path to scaling_metrics.pkl (default: auto-detected in scaled-dir)
    --max-rfree         Discard datasets with Rf_final above this value (default: 0.45)
    --min-cc            Discard datasets with post-scaling CC below this value (default: 0.0,
                        i.e. no CC filter; set e.g. 0.55 to enable)
    --output            Output .txt file path (default: configs/scaled_filtered_files.txt)
"""

import sys
import os
import glob
import pickle
import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser(description="Filter scaled datasets and write file list")
    parser.add_argument("--refine-summary", default="refine_1nwl/refine_summary.csv")
    parser.add_argument("--reindex-record", default="reindexed/reindex_record.pkl")
    parser.add_argument("--scaled-dir",     default="scaled/")
    parser.add_argument("--metrics",        default=None,
                        help="Path to scaling_metrics.pkl; auto-detected if omitted")
    parser.add_argument("--max-rfree",      type=float, default=0.45)
    parser.add_argument("--min-cc",         type=float, default=0.0)
    parser.add_argument("--output",         default="configs/scaled_filtered_files.txt")
    args = parser.parse_args()

    # --- Load refinement summary ---
    if not os.path.isfile(args.refine_summary):
        print(f"Error: refine_summary not found: {args.refine_summary}", file=sys.stderr)
        sys.exit(1)
    df = pd.read_csv(args.refine_summary)
    df["file_idx"] = df["file_idx"].astype(str).str.zfill(4)
    df["symop"]    = df["symop"].astype(str)

    drop = set()

    # --- Resolve reindexing ambiguity ---
    if os.path.isfile(args.reindex_record):
        with open(args.reindex_record, "rb") as f:
            reindex_record = pickle.load(f)
        if reindex_record is not None:
            ambiguous = reindex_record[reindex_record["num_duplicates"] > 1]["file_idx"].tolist()
            ambiguous = [str(x).zfill(4) for x in ambiguous]
            df_ambig  = df[df["file_idx"].isin(ambiguous)]
            worse = df_ambig.loc[df_ambig.groupby("file_idx")["Rf_final"].idxmax()]
            drop1 = set((worse["file_idx"] + "_" + worse["symop"] + ".mtz").tolist())
            print(f"Dropping {len(drop1)} worse-symop duplicates from {len(ambiguous)} ambiguous datasets")
            drop |= drop1
    else:
        print(f"No reindex_record found at {args.reindex_record}; skipping ambiguity resolution")

    # --- Filter by R-free ---
    bad_R = df[df["Rf_final"] > args.max_rfree]
    drop2 = set((bad_R["file_idx"] + "_" + bad_R["symop"] + ".mtz").tolist())
    print(f"Dropping {len(drop2)} datasets with Rf_final > {args.max_rfree}")
    drop |= drop2

    # --- Collect scaled files ---
    all_scaled = sorted(glob.glob(os.path.join(args.scaled_dir, "*.mtz")))
    if not all_scaled:
        print(f"Error: no MTZ files found in {args.scaled_dir}", file=sys.stderr)
        sys.exit(1)
    file_list = [f for f in all_scaled if os.path.basename(f) not in drop]

    # --- Drop files where F-obs-scaled is entirely NaN (silent scaling divergence) ---
    import gemmi, numpy as np
    nan_files = []
    for mtz_path in file_list:
        try:
            mtz = gemmi.read_mtz_file(mtz_path)
            col = mtz.column_with_label("F-obs-scaled")
            if col is not None and not np.isfinite(np.array(col)).any():
                nan_files.append(mtz_path)
        except Exception:
            pass
    if nan_files:
        print(f"Dropping {len(nan_files)} file(s) with all-NaN F-obs-scaled: "
              + ", ".join(os.path.basename(f) for f in nan_files))
        file_list = [f for f in file_list if f not in nan_files]

    # --- Filter by post-scaling CC ---
    if args.min_cc > 0.0:
        metrics_path = args.metrics
        if metrics_path is None:
            candidates = glob.glob(os.path.join(args.scaled_dir, "*scaling_metrics.pkl"))
            if candidates:
                metrics_path = candidates[0]
        if metrics_path and os.path.isfile(metrics_path):
            metrics = pd.read_pickle(metrics_path)
            low_cc  = set(metrics[(metrics["end_corr"] < args.min_cc) |
                                   metrics["end_corr"].isnull()]["file"].tolist())
            before  = len(file_list)
            file_list = [f for f in file_list
                         if os.path.basename(f).replace(".mtz", "") not in low_cc]
            print(f"Dropping {before - len(file_list)} datasets with post-scaling CC < {args.min_cc}")
        else:
            print("Warning: --min-cc set but no scaling_metrics.pkl found; skipping CC filter")

    # --- Write output ---
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(file_list) + "\n")
    print(f"\n{len(file_list)} datasets written to {args.output}")


if __name__ == "__main__":
    main()
