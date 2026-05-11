#!/usr/bin/env python
"""
Compute VAE map quality metrics for the PTP1B VALDO pipeline:
  1. Apo peak metric  — mean highest WDF peak across true-apo datasets
  2. Heavy atom metric — WDF map value at ligand heavy atom (Cl/Br/S/I) positions

Run from PTP1B_pipeline/:
    python compute_valdo_metrics.py
"""

import os
import re
import glob
import numpy as np
import gemmi
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RECONS_PHASED           = os.path.join(SCRIPT_DIR, "vae", "recons_phased")
BOUND_MODELS_STD        = os.path.join(SCRIPT_DIR, "bound_models_standardized")
ALL_SUPERPOSED_V2       = os.path.join(SCRIPT_DIR, "all_superposed_v2")
MAPPING_TXT             = os.path.join(SCRIPT_DIR, "..", "notebooks", "ligand_cif_to_dataset_mapping.txt")

DIFF_COL  = "WDF"
PHASE_COL = "PH2FOFCWT"


def classify_bound_models(superposed_dir):
    """
    Return (keedy_ids, ginn_ids) sets of 4-digit dataset IDs by parsing
    all_superposed_v2 filenames:
      Keedy: PTP1B_yXXXX_*
      Ginn:  yXXXX_cluster4x_*
    IDs in both (e.g. 0205) appear in both sets.
    """
    keedy_ids, ginn_ids = set(), set()
    for fname in os.listdir(superposed_dir):
        if fname == "README.txt":
            continue
        m_keedy = re.match(r"PTP1B_y(\d{4})_", fname)
        m_ginn  = re.match(r"y(\d{4})_cluster4x_", fname)
        if m_keedy:
            keedy_ids.add(m_keedy.group(1))
        if m_ginn:
            ginn_ids.add(m_ginn.group(1))
    return keedy_ids, ginn_ids


def load_apo_ids(mapping_txt):
    """Return set of 4-digit dataset IDs where no ligand was soaked (line has no .cif)."""
    apo_ids = set()
    with open(mapping_txt) as f:
        for line in f:
            stripped = line.strip()
            if stripped and not stripped.endswith(".cif"):
                # format: PTP1B-yXXXX:  → [-5:-1] = XXXX
                apo_ids.add(stripped[-5:-1])
    return apo_ids


def find_mtz(dataset_id, phased_dir):
    """Find phased MTZ for a 4-digit dataset ID; tries _0 then _1 suffix."""
    for suffix in ("_0", "_1"):
        path = os.path.join(phased_dir, f"{dataset_id}{suffix}.mtz")
        if os.path.exists(path):
            return path
    matches = glob.glob(os.path.join(phased_dir, f"{dataset_id}_*.mtz"))
    return matches[0] if matches else None


def extract_id_from_mtz(mtz_path):
    """Extract 4-digit ID from filename like 0049_0.mtz."""
    m = re.match(r"(\d{4})_", os.path.basename(mtz_path))
    return m.group(1) if m else None


def extract_id_from_pdb(pdb_path):
    """Extract 4-digit ID from bound_models_standardized filename XXXX.pdb."""
    m = re.match(r"(\d{4})\.pdb$", os.path.basename(pdb_path))
    return m.group(1) if m else None


def wdf_grid(mtz_path):
    mtz = gemmi.read_mtz_file(mtz_path)
    grid = mtz.transform_f_phi_to_map(DIFF_COL, PHASE_COL, sample_rate=3.0)
    grid.normalize()
    return grid


def heavy_atom_peak(grid, pdb_path):
    """
    Return max normalized WDF value across all Cl/Br/S/I atoms of LIG residue,
    expanded over crystallographic symmetry. Returns None if no heavy atoms found.
    """
    st = gemmi.read_pdb(pdb_path)
    sel = gemmi.Selection("[CL,Br,S,I]")
    sel_model = sel.copy_model_selection(st[0])
    lig_atoms = [cra for cra in sel_model.all() if cra.residue.name == "LIG"]

    if not lig_atoms:
        return None

    ops = grid.spacegroup.operations()
    peak_vals = []

    for cra in lig_atoms:
        frac = st.cell.fractionalize(cra.atom.pos)
        for op in ops:
            mapped = op.apply_to_xyz(frac.tolist())
            wx = mapped[0] - np.floor(mapped[0])
            wy = mapped[1] - np.floor(mapped[1])
            wz = mapped[2] - np.floor(mapped[2])
            a = round(wx * grid.nu) % grid.nu
            b = round(wy * grid.nv) % grid.nv
            c = round(wz * grid.nw) % grid.nw
            peak_vals.append(grid.get_value(a, b, c))

    return float(np.max(peak_vals))


def report_ha(results):
    if not results:
        print("  No results.")
        return
    print(f"  Datasets with heavy atoms : {len(results)}")
    for did, val in results:
        print(f"    {did}: {val:.4f}")
    vals = [v for _, v in results]
    print(f"  Mean WDF peak : {np.mean(vals):.4f}  Std : {np.std(vals):.4f}")


def main():
    apo_ids = load_apo_ids(MAPPING_TXT)
    keedy_ids, ginn_ids = classify_bound_models(ALL_SUPERPOSED_V2)
    all_mtz = sorted(glob.glob(os.path.join(RECONS_PHASED, "*.mtz")))

    # ── Metric 1: Apo peak ──────────────────────────────────────────────────
    print("=== APO PEAK METRIC ===")
    apo_peaks = []
    apo_failed = []

    for mtz_path in tqdm(all_mtz, desc="Apo peak"):
        did = extract_id_from_mtz(mtz_path)
        if did is None or did not in apo_ids:
            continue
        try:
            grid = wdf_grid(mtz_path)
            apo_peaks.append(float(np.max(grid.array)))
        except Exception as e:
            apo_failed.append((did, str(e)))

    print(f"Apo datasets processed : {len(apo_peaks)}")
    if apo_failed:
        print(f"Failed                 : {len(apo_failed)}")
        for did, err in apo_failed:
            print(f"  {did}: {err}")
    print(f"Mean highest WDF peak  : {np.mean(apo_peaks):.4f}")
    print(f"Std                    : {np.std(apo_peaks):.4f}")

    # ── Metric 2: Heavy atom peak ────────────────────────────────────────────
    print("\n=== HEAVY ATOM PEAK METRIC ===")
    all_pdb = sorted(glob.glob(os.path.join(BOUND_MODELS_STD, "*.pdb")))
    ha_all     = []   # (did, peak) — all bound models
    ha_no_heavy = []
    ha_missing  = []

    for pdb_path in tqdm(all_pdb, desc="Heavy atom peak"):
        did = extract_id_from_pdb(pdb_path)
        if did is None:
            continue
        mtz_path = find_mtz(did, RECONS_PHASED)
        if mtz_path is None:
            ha_missing.append(did)
            continue
        try:
            grid = wdf_grid(mtz_path)
            val = heavy_atom_peak(grid, pdb_path)
            if val is None:
                ha_no_heavy.append(did)
            else:
                ha_all.append((did, val))
        except Exception as e:
            ha_missing.append((did, str(e)))

    if ha_no_heavy:
        print(f"No Cl/Br/S/I in LIG   : {len(ha_no_heavy)} — {ha_no_heavy}")
    if ha_missing:
        print(f"MTZ missing / failed   : {len(ha_missing)}")

    ha_keedy = [(did, v) for did, v in ha_all if did in keedy_ids]
    ha_ginn  = [(did, v) for did, v in ha_all if did in ginn_ids]

    print(f"\n-- Keedy bound models --")
    report_ha(ha_keedy)
    print(f"\n-- Ginn bound models --")
    report_ha(ha_ginn)
    print(f"\n-- All bound models --")
    report_ha(ha_all)


if __name__ == "__main__":
    main()
