#!/usr/bin/env python3
"""
Create a multi-block merged SF-mmCIF (sf.cif) from:
  1) a PHENIX refinement MTZ
  2) a VALDO output MTZ

This script mirrors the workflow in create_VALDO_sf_cifs.ipynb:
- Read both MTZs with reciprocalspaceship
- reset_index() so H/K/L are real columns
- Create several "views" of the VALDO dataset as separate gemmi.Mtz objects (including an ESF block for a chosen extrapolation factor)
- Convert each gemmi.Mtz to SF-mmCIF blocks via gemmi.MtzToCif + spec_lines
- Carry each MTZ's dataset_name into _diffrn.details; ESF block dataset_name and block_name include the extrapolation factor
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import reciprocalspaceship as rs
import gemmi
from gemmi import cif


def build_default_spec_lines() -> List[str]:
    # One specification per output tag.
    return [
        "H H index_h",
        "K H index_k",
        "L H index_l",

        # Optional FreeR/flags (skip if missing)
        "? FREE|RFREE|FREER|FreeR_flag|R-free-flags I status S",

        # Typical observed amplitudes
        "? FP|F|FOBS|F-obs-filtered|F-obs-scaled|recons|ESF F F_meas_au",
        "& SIGFP|SIGF|SIGFOBS|SIGF-obs-filtered|SIGF-obs-scaled|SIG-recons|SIGESF Q F_meas_sigma_au",

        # Example map coefficients
        "? FWT|2FOFCWT|2FOFCWT_no_fill F pdbx_FWT",
        "& PHWT|refine_PH2FOFCWT|PH2FOFCWT|PH2FOFCWT_no_fill P pdbx_PHWT",

        # Difference map treatment (Fo-Fc)
        "? FOFCWT|DF|WDF F pdbx_DELFWT",
        "& PHFOFCWT|PHDF P pdbx_DELPHWT",

        # F-model
        "? F-model F F_calc",
        "& PHIF-model P phase_calc",
    ]


def _pick_dataset_name(mtz: gemmi.Mtz) -> str:
    # Prefer the first non-base dataset_name if present.
    for ds in mtz.datasets:
        if ds.dataset_name and ds.dataset_name != "HKL_base":
            return ds.dataset_name
    for ds in mtz.datasets:
        if ds.dataset_name:
            return ds.dataset_name
    return ""


def _upsert_diffrn(block: cif.Block, diffrn_fields: Dict[str, str], move_before_loops: bool = True) -> None:
    """
    Ensure the diffrn category exists as a loop, set/update row 0, and optionally
    move that loop before the first non-diffrn loop (usually _refln).
    """
    tags = list(diffrn_fields.keys())
    t = block.find_or_add("_diffrn.", tags)
    t.ensure_loop()

    if len(t) == 0:
        t.append_row([diffrn_fields[k] for k in tags])
    else:
        row0 = t[0]
        for k, v in diffrn_fields.items():
            row0[f"_diffrn.{k}"] = v

    if not move_before_loops:
        return

    # Find first loop that is not _diffrn.* (typically _refln.*)
    insert_pos = None
    for idx, item in enumerate(block):
        lp = item.loop
        if lp is not None:
            if not lp.tags or not lp.tags[0].startswith("_diffrn."):
                insert_pos = idx
                break

    if insert_pos is None:
        return

    # Move the diffrn loop item before the first non-diffrn loop
    try:
        diffrn_item_pos = block.get_index("_diffrn.id")
        block.move_item(diffrn_item_pos, insert_pos)
    except Exception:
        # If get_index/move_item isn't available in some gemmi builds, just leave order alone.
        pass


def mtz_objects_to_multiblock_sf_cif(
    mtz_list: List[gemmi.Mtz],
    out_path: str,
    block_names: List[str] | None = None,
    spec_lines: List[str] | None = None,
    with_history: bool = True,
    diffrn_id: str = "1",
) -> cif.Document:
    if block_names is not None and len(block_names) != len(mtz_list):
        raise ValueError("block_names must have the same length as mtz_list")

    conv = gemmi.MtzToCif()
    conv.with_history = bool(with_history)
    if spec_lines is not None:
        conv.spec_lines = list(spec_lines)

    out_doc = cif.Document()

    for i, mtz in enumerate(mtz_list, start=1):
        ds_name = _pick_dataset_name(mtz)
        tmp_doc = cif.read_string(conv.write_cif_to_string(mtz))

        for j, blk in enumerate(tmp_doc, start=1):
            new_blk = out_doc.add_copied_block(blk)

            base = (block_names[i - 1] if block_names else f"mtz{i}")
            new_blk.name = base if j == 1 else f"{base}_{j}"

            if ds_name:
                _upsert_diffrn(
                    new_blk,
                    diffrn_fields={"id": diffrn_id, "details": ds_name},
                    move_before_loops=True,
                )

    out_doc.write_file(out_path, style=cif.Style.Pdbx)
    return out_doc


def _require_columns(ds: rs.DataSet, cols: List[str], where: str) -> None:
    missing = [c for c in cols if c not in ds.columns]
    if missing:
        raise ValueError(f"{where}: missing columns: {missing}")


def prepare_phenix_blocks(
    ds_phenix: rs.DataSet,
    *,
    project_name: str = "1",
    crystal_name: str = "1",
    dataset_name: str = "PHENIX refinement against extrapolated structure factor amplitudes",
    block_name: str = "refinement",
) -> Tuple[gemmi.Mtz, str]:
    mtz = rs.io.to_gemmi(
        ds_phenix,
        skip_problem_mtztypes=True,
        project_name=project_name,
        crystal_name=crystal_name,
        dataset_name=dataset_name,
    )
    return mtz, block_name


def prepare_valdo_blocks(
    ds_valdo: rs.DataSet,
    *,
    project_name: str = "1",
    crystal_name: str = "1",
    extrap_factor: int = 8,
    # VALDO column labels (defaults match your notebook)
    fobs: str = "F-obs",
    sigfobs: str = "SIGF-obs",
    fobs_scaled: str = "F-obs-scaled",
    sigfobs_scaled: str = "SIGF-obs-scaled",
    recon: str = "recons",
    sig_recon: str = "SIG_recons",
    phase: str = "refine_PH2FOFCWT",
    diff: str = "diff",
    wdiff: str = "WDF",
    include_scaled: bool = True,
) -> Tuple[List[gemmi.Mtz], List[str]]:
    mtz_list: List[gemmi.Mtz] = []
    block_names: List[str] = []

    if extrap_factor not in (2, 4, 6, 8, 16):
        raise ValueError("extrap_factor must be one of 2, 4, 6, 8, 16")

    # VALDO commonly labels extrapolated SF columns as ESF_<factor> / SIGESF_<factor>.
    esf_col = f"ESF_{extrap_factor}"
    sig_esf_col = f"SIGESF_{extrap_factor}"

    # Always required for at least "raw_data" and "reconstructed_sf"
    _require_columns(ds_valdo, ["H", "K", "L", fobs, sigfobs], "VALDO (raw_data)")
    _require_columns(ds_valdo, ["H", "K", "L", recon, sig_recon], "VALDO (reconstructed_sf)")

    ds_orig = ds_valdo[["H", "K", "L", fobs, sigfobs]].copy()
    ds_orig.rename(columns={fobs: "FP", sigfobs: "SIGFP"}, inplace=True)

    mtz_list.append(
        rs.io.to_gemmi(
            ds_orig,
            skip_problem_mtztypes=True,
            project_name=project_name,
            crystal_name=crystal_name,
            dataset_name="Observed structure factor amplitudes (unextrapolated)",
        )
    )
    block_names.append("raw_data")

    # Optional: scaled raw data
    if include_scaled and (fobs_scaled in ds_valdo.columns) and (sigfobs_scaled in ds_valdo.columns):
        ds_orig_sc = ds_valdo[["H", "K", "L", fobs_scaled, sigfobs_scaled]].copy()
        ds_orig_sc.rename(columns={fobs_scaled: "FP", sigfobs_scaled: "SIGFP"}, inplace=True)
        mtz_list.append(
            rs.io.to_gemmi(
                ds_orig_sc,
                skip_problem_mtztypes=True,
                project_name=project_name,
                crystal_name=crystal_name,
                dataset_name="Observed structure factor amplitudes (unextrapolated) after scaling to a reference",
            )
        )
        block_names.append("raw_data_scaled_to_ref")

    # Reconstructed SFs
    ds_recon = ds_valdo[["H", "K", "L", recon, sig_recon]].copy()
    ds_recon.rename(columns={recon: "FP", sig_recon: "SIGFP"}, inplace=True)
    mtz_list.append(
        rs.io.to_gemmi(
            ds_recon,
            skip_problem_mtztypes=True,
            project_name=project_name,
            crystal_name=crystal_name,
            dataset_name="Structure factor amplitude reconstructions produced by VALDO (apo state surrogate)",
        )
    )
    block_names.append("reconstructed_sf")

    # Optional: ESF (event) block (needs ESF, SIGESF, and phase)
    if esf_col in ds_valdo.columns and sig_esf_col in ds_valdo.columns and phase in ds_valdo.columns:
        ds_event = ds_valdo[["H", "K", "L", esf_col, sig_esf_col, phase]].copy()
        ds_event.rename(columns={esf_col: "ESF", sig_esf_col: "SIGESF"}, inplace=True)
        # rename ESF_<factor>/SIGESF_<factor> to ESF/SIGESF so spec_lines are factor-agnostic
        mtz_list.append(
            rs.io.to_gemmi(
                ds_event,
                skip_problem_mtztypes=True,
                project_name=project_name,
                crystal_name=crystal_name,
                dataset_name=f"{extrap_factor}x extrapolated structure factors (ESF) based on obs-reconstructed differences (analog of PanDDA event map; apo phases).",
            )
        )
        block_names.append(f"{extrap_factor}x_extrapol_sf")

    # Optional: unweighted diff map (diff + phase)
    if diff in ds_valdo.columns and phase in ds_valdo.columns:
        ds_z = ds_valdo[["H", "K", "L", diff, phase]].copy()
        # rename into the labels expected by spec_lines for Fo-Fc terms
        ds_z.rename(columns={diff: "DF", phase: "PHDF"}, inplace=True)
        mtz_list.append(
            rs.io.to_gemmi(
                ds_z,
                skip_problem_mtztypes=True,
                project_name=project_name,
                crystal_name=crystal_name,
                dataset_name="Unweighted difference map between observed and reconstructed sf factors; apo phases.",
            )
        )
        block_names.append("unweighted_diff_map")

    # Optional: weighted diff map (WDF + phase)
    if wdiff in ds_valdo.columns and phase in ds_valdo.columns:
        ds_z_wt = ds_valdo[["H", "K", "L", wdiff, phase]].copy()
        ds_z_wt.rename(columns={wdiff: "DF", phase: "PHDF"}, inplace=True)
        mtz_list.append(
            rs.io.to_gemmi(
                ds_z_wt,
                skip_problem_mtztypes=True,
                project_name=project_name,
                crystal_name=crystal_name,
                dataset_name="Weighted difference map between observed and reconstructed sf factors (analog of PanDDA Z maps); apo phases.",
            )
        )
        block_names.append("weighted_diff_map")

    return mtz_list, block_names


def create_valdo_sf_cif(
    phenix_mtz_path: str,
    valdo_mtz_path: str,
    out_cif_path: str,
    *,
    extrap_factor: int = 8,
    spec_lines: List[str] | None = None,
    with_history: bool = True,
    include_scaled: bool = True,
    phenix_block_name: str = "refinement",
    diffrn_id: str = "1",
) -> cif.Document:
    # Read MTZs
    ds_valdo = rs.read_mtz(valdo_mtz_path)
    ds_phenix = rs.read_mtz(phenix_mtz_path)

    # Your working fix: ensure H/K/L are columns, not only an index
    ds_valdo.reset_index(inplace=True)
    ds_phenix.reset_index(inplace=True)

    # Build blocks
    phenix_mtz, phenix_name = prepare_phenix_blocks(ds_phenix, block_name=phenix_block_name)
    valdo_mtz_list, valdo_names = prepare_valdo_blocks(ds_valdo, include_scaled=include_scaled, extrap_factor=extrap_factor)

    mtz_list = [phenix_mtz] + valdo_mtz_list
    block_names = [phenix_name] + valdo_names

    # Convert to CIF
    if spec_lines is None:
        spec_lines = build_default_spec_lines()

    return mtz_objects_to_multiblock_sf_cif(
        mtz_list=mtz_list,
        out_path=out_cif_path,
        block_names=block_names,
        spec_lines=spec_lines,
        with_history=with_history,
        diffrn_id=diffrn_id,
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Build a multi-block merged SF-mmCIF (sf.cif) from PHENIX + VALDO MTZs."
    )
    p.add_argument("phenix_mtz", help="PHENIX output MTZ (refinement, map coeffs, etc.)")
    p.add_argument("valdo_mtz", help="VALDO output MTZ (F-obs, recon, ESF, diff maps, etc.)")
    p.add_argument("-o", "--out", default="merged-sf.cif", help="Output CIF path (default: merged-sf.cif)")
    p.add_argument("--no-history", action="store_true", help="Do not include MTZ history in output")
    p.add_argument("--omit-scaled", action="store_true", help="Do not emit the scaled raw-data block (if present)")
    p.add_argument("--phenix-block-name", default="refinement", help="CIF block name for the PHENIX MTZ block")
    p.add_argument("--diffrn-id", default="1", help="Value for _diffrn.id in each block (default: 1)")
    p.add_argument("--extrap-factor", type=int, default=8, choices=[2, 4, 6, 8, 16],
                   help="Extrapolation factor for ESF columns (expects ESF_<factor>/SIGESF_<factor>; default: 8)")

    # Optional: allow overriding VALDO column labels if your pipeline changes
    p.add_argument("--valdo-phase", default="refine_PH2FOFCWT", help="VALDO phase column label (default: refine_PH2FOFCWT)")
    p.add_argument("--valdo-diff", default="diff", help="VALDO unweighted diff column label (default: diff)")
    p.add_argument("--valdo-wdiff", default="WDF", help="VALDO weighted diff column label (default: WDF)")
    p.add_argument("--valdo-fobs", default="F-obs", help="VALDO observed F column label (default: F-obs)")
    p.add_argument("--valdo-sigfobs", default="SIGF-obs", help="VALDO sigma observed F label (default: SIGF-obs)")
    p.add_argument("--valdo-fobs-scaled", default="F-obs-scaled", help="VALDO scaled observed F label (default: F-obs-scaled)")
    p.add_argument("--valdo-sigfobs-scaled", default="SIGF-obs-scaled", help="VALDO sigma scaled observed F label (default: SIGF-obs-scaled)")
    p.add_argument("--valdo-recon", default="recons", help="VALDO reconstructed F label (default: recons)")
    p.add_argument("--valdo-sig-recon", default="SIG_recons", help="VALDO sigma reconstructed F label (default: SIG_recons)")

    return p


def main() -> None:
    args = _build_arg_parser().parse_args()

    # Read, reset_index, and prepare blocks using possibly overridden labels
    ds_valdo = rs.read_mtz(args.valdo_mtz)
    ds_phenix = rs.read_mtz(args.phenix_mtz)
    ds_valdo.reset_index(inplace=True)
    ds_phenix.reset_index(inplace=True)

    phenix_mtz, phenix_name = prepare_phenix_blocks(ds_phenix, block_name=args.phenix_block_name)

    valdo_mtz_list, valdo_names = prepare_valdo_blocks(
        ds_valdo,
        include_scaled=(not args.omit_scaled),
        extrap_factor=args.extrap_factor,
        phase=args.valdo_phase,
        diff=args.valdo_diff,
        wdiff=args.valdo_wdiff,
        fobs=args.valdo_fobs,
        sigfobs=args.valdo_sigfobs,
        fobs_scaled=args.valdo_fobs_scaled,
        sigfobs_scaled=args.valdo_sigfobs_scaled,
        recon=args.valdo_recon,
        sig_recon=args.valdo_sig_recon,
    )

    mtz_list = [phenix_mtz] + valdo_mtz_list
    block_names = [phenix_name] + valdo_names

    mtz_objects_to_multiblock_sf_cif(
        mtz_list=mtz_list,
        out_path=args.out,
        block_names=block_names,
        spec_lines=build_default_spec_lines(),
        with_history=(not args.no_history),
        diffrn_id=args.diffrn_id,
    )


if __name__ == "__main__":
    main()
