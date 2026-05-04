#!/usr/bin/env python
"""Plot ROC/AUC curve from a tagged blob stats pickle file.

Usage:
    python plot_auc.py [filtered_blob_stats.pkl] [output.png]

Defaults to vae/blobs/filtered_blob_stats_tagged.pkl in the current directory.
"""

import sys
import os
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics


def plot_roc(blob_df, name="VALDO", ax=None):
    fpr, tpr, thresholds = metrics.roc_curve(blob_df["ligand"], blob_df["score"], pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        fig = ax.get_figure()

    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=name)
    display.plot(ax=ax)
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(alpha=0.3)

    return fig, roc_auc


def main():
    parser = argparse.ArgumentParser(description="Plot ROC/AUC from tagged blob stats")
    parser.add_argument(
        "blob_stats",
        nargs="?",
        default="vae/blobs/filtered_blob_stats_tagged.pkl",
        help="Path to filtered_blob_stats_tagged.pkl",
    )
    parser.add_argument(
        "output",
        nargs="?",
        default=None,
        help="Output image path (default: roc_curve.png next to blob_stats)",
    )
    parser.add_argument("--name", default="VALDO", help="Label for the ROC curve")
    args = parser.parse_args()

    if not os.path.isfile(args.blob_stats):
        print(f"Error: file not found: {args.blob_stats}", file=sys.stderr)
        sys.exit(1)

    blob_df = pd.read_pickle(args.blob_stats)

    if "ligand" not in blob_df.columns:
        print(
            "Error: 'ligand' column not found — run tag_blobs with bound_models_folder first.",
            file=sys.stderr,
        )
        sys.exit(1)

    n_total = len(blob_df)
    n_samples = blob_df["sample"].nunique()
    n_pos = int(blob_df["ligand"].sum())
    n_neg = n_total - n_pos

    print(f"Total blobs:        {n_total}")
    print(f"Unique datasets:    {n_samples}")
    print(f"Positive (ligand):  {n_pos}")
    print(f"Negative:           {n_neg}")

    fig, roc_auc = plot_roc(blob_df, name=args.name)
    print(f"AUC:                {roc_auc:.4f}")

    out_path = args.output or os.path.join(os.path.dirname(args.blob_stats), "roc_curve.png")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
