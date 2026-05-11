#!/usr/bin/env python
"""
Collect ablation metrics across all hyperparameter settings and print a comparison table.

Run from PTP1B_pipeline/ after ablation jobs have completed:
    python collect_ablation_metrics.py
"""

import os
import sys
import csv
import glob as _glob

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from compute_valdo_metrics import compute_metrics

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
ABLATION_DIR = os.path.join(SCRIPT_DIR, "ablation")
OUTPUT_CSV   = os.path.join(ABLATION_DIR, "ablation_metrics.csv")

SETTINGS = [
    ("baseline (latent=7, relu, wkl=1, [3,6]/100)",
     os.path.join(SCRIPT_DIR, "vae", "recons_phased")),
    ("latent_dim_3",
     os.path.join(ABLATION_DIR, "latent_dim_3",    "vae", "recons_phased")),
    ("latent_dim_5",
     os.path.join(ABLATION_DIR, "latent_dim_5",    "vae", "recons_phased")),
    ("latent_dim_9",
     os.path.join(ABLATION_DIR, "latent_dim_9",    "vae", "recons_phased")),
    ("activation_tanh",
     os.path.join(ABLATION_DIR, "activation_tanh", "vae", "recons_phased")),
    ("w_kl_0.1",
     os.path.join(ABLATION_DIR, "w_kl_0.1",        "vae", "recons_phased")),
    ("w_kl_10",
     os.path.join(ABLATION_DIR, "w_kl_10",         "vae", "recons_phased")),
    ("hidden_small ([2,4]/100)",
     os.path.join(ABLATION_DIR, "hidden_small",    "vae", "recons_phased")),
    ("hidden_large ([4,8]/100)",
     os.path.join(ABLATION_DIR, "hidden_large",    "vae", "recons_phased")),
    ("hidden_wide ([3,6]/200)",
     os.path.join(ABLATION_DIR, "hidden_wide",     "vae", "recons_phased")),
]

FIELDS = ["setting", "apo_mean", "apo_std", "n_apo",
          "keedy_mean", "n_keedy", "ginn_mean", "n_ginn", "all_mean", "n_all"]

COL_W = 38
rows = []

print(f"{'Setting':<{COL_W}} {'Apo mean':>10} {'Keedy HA':>10} {'Ginn HA':>10} {'All HA':>10}  {'N_apo':>6}")
print("-" * (COL_W + 50))

for name, phased_dir in SETTINGS:
    n_mtz = len(_glob.glob(os.path.join(phased_dir, "*.mtz"))) if os.path.isdir(phased_dir) else 0

    if n_mtz == 0:
        print(f"{name:<{COL_W}} {'NOT READY':>10}")
        rows.append({"setting": name, **{f: "N/A" for f in FIELDS[1:]}})
        continue

    m = compute_metrics(phased_dir)
    print(
        f"{name:<{COL_W}} "
        f"{m['apo_mean']:>10.3f} "
        f"{m['keedy_mean']:>10.3f} "
        f"{m['ginn_mean']:>10.3f} "
        f"{m['all_mean']:>10.3f}  "
        f"{m['n_apo']:>6}"
    )
    rows.append({
        "setting":    name,
        "apo_mean":   f"{m['apo_mean']:.4f}",
        "apo_std":    f"{m['apo_std']:.4f}",
        "n_apo":      m["n_apo"],
        "keedy_mean": f"{m['keedy_mean']:.4f}",
        "n_keedy":    m["n_keedy"],
        "ginn_mean":  f"{m['ginn_mean']:.4f}",
        "n_ginn":     m["n_ginn"],
        "all_mean":   f"{m['all_mean']:.4f}",
        "n_all":      m["n_all"],
    })

with open(OUTPUT_CSV, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(rows)

print(f"\nCSV written to: {OUTPUT_CSV}")
