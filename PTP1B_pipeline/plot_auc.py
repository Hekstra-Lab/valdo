#!/usr/bin/env python
"""Plot ROC/AUC curve and AUC-vs-N-blobs curve from a tagged blob stats pickle file.

Usage:
    python plot_auc.py [filtered_blob_stats.pkl] [output.png]

Defaults to vae/blobs/filtered_blob_stats_tagged.pkl in the current directory.
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics


def plot_roc(blob_df, name="VALDO", ax=None):
    fpr, tpr, thresholds = metrics.roc_curve(blob_df["ligand"], blob_df["score"], pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 3))
    else:
        fig = ax.get_figure()

    display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name=name)
    display.plot(ax=ax)
    ax.set_aspect("auto")
    ax.plot([0, 1], [0, 1], "k--", lw=1.5)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.grid(alpha=0.3)

    return fig, roc_auc


def plot_auc_vs_n(blob_df, name="VALDO", ax=None):
    blob_sorted = blob_df.sort_values("peakz", ascending=False).reset_index(drop=True)
    n_total = len(blob_sorted)

    checkpoints = list(np.linspace(500, n_total, 20, dtype=int))
    if n_total not in checkpoints:
        checkpoints.append(n_total)

    ns, aucs = [], []
    for n in checkpoints:
        sub = blob_sorted.iloc[:n]
        if sub["ligand"].sum() == 0 or sub["ligand"].sum() == len(sub):
            continue
        fpr, tpr, _ = metrics.roc_curve(sub["ligand"], sub["score"], pos_label=1)
        ns.append(n)
        aucs.append(metrics.auc(fpr, tpr))

    ns = np.array(ns)
    aucs = np.array(aucs)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.get_figure()

    ax.plot(ns, aucs, "o-", lw=2, ms=4, label=name)
    ax.set_xlabel("Number of blobs (top-N by peakz)")
    ax.set_ylabel("AUC")
    ax.set_title("AUC vs number of blobs")
    ax.legend()
    ax.grid(alpha=0.3)

    return fig, ns, aucs


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

    out_dir = os.path.dirname(args.blob_stats)

    # ROC curve (all blobs)
    fig_roc, roc_auc = plot_roc(blob_df, name=args.name)
    print(f"AUC (all blobs):    {roc_auc:.4f}")
    roc_path = args.output or os.path.join(out_dir, "roc_curve.png")
    fig_roc.tight_layout()
    fig_roc.savefig(roc_path, dpi=150)
    roc_svg_path = roc_path.replace(".png", ".svg")
    fig_roc.savefig(roc_svg_path, transparent=True)
    print(f"ROC curve saved to {roc_path}, {roc_svg_path}")

    fpr, tpr, thresholds = metrics.roc_curve(blob_df["ligand"], blob_df["score"], pos_label=1)
    roc_csv_path = roc_path.replace(".png", "_data.csv")
    pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": thresholds}).to_csv(roc_csv_path, index=False)
    print(f"ROC data saved to  {roc_csv_path}")

    # AUC vs N blobs
    fig_n, ns, aucs = plot_auc_vs_n(blob_df, name=args.name)
    best_idx = int(np.argmax(aucs))
    print(f"Best AUC:           {aucs[best_idx]:.4f} at N={ns[best_idx]}")
    auc_n_path = os.path.join(out_dir, "auc_vs_nblobs.png")
    fig_n.tight_layout()
    fig_n.savefig(auc_n_path, dpi=150)
    auc_n_svg_path = auc_n_path.replace(".png", ".svg")
    fig_n.savefig(auc_n_svg_path, transparent=True)
    print(f"AUC-vs-N saved to  {auc_n_path}, {auc_n_svg_path}")

    auc_n_csv_path = auc_n_path.replace(".png", "_data.csv")
    pd.DataFrame({"n_blobs": ns, "auc": aucs}).to_csv(auc_n_csv_path, index=False)
    print(f"AUC-vs-N data saved to {auc_n_csv_path}")


if __name__ == "__main__":
    main()
