#!/usr/bin/env python
"""Parse PHENIX refinement logs from refine_output/ into a summary CSV.

Usage:
    python parse_refine_logs.py [refine_output_dir] [output_csv]

Defaults:
    refine_output_dir : refine_1nwl/refine_output/
    output_csv        : refine_1nwl/refine_summary.csv

Output CSV columns: file_idx, symop, Rw_start, Rf_start, Rw_final, Rf_final, time(s)
"""

import sys
import os
import glob
import argparse
import pandas as pd


def parse_log(log_path):
    with open(log_path) as f:
        content = f.read()

    # PHENIX prints a summary table at the end with lines like:
    # "r_work = 0.xxxx r_free = 0.xxxx" for start and final
    lines = [l.strip() for l in content.splitlines()
             if "r_work" in l.lower() and "r_free" in l.lower()]
    if len(lines) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")

    def _parse_line(line):
        parts = line.replace(",", " ").replace("=", " ").split()
        rw = float(parts[parts.index("r_work") + 1])
        rf = float(parts[parts.index("r_free") + 1])
        return rw, rf

    try:
        rw_start, rf_start = _parse_line(lines[0])
        rw_final, rf_final = _parse_line(lines[-1])
    except Exception:
        return float("nan"), float("nan"), float("nan"), float("nan")

    return rw_start, rf_start, rw_final, rf_final


def main():
    parser = argparse.ArgumentParser(description="Parse PHENIX log files into a summary CSV")
    parser.add_argument("refine_dir", nargs="?", default="refine_1nwl/refine_output/")
    parser.add_argument("output_csv", nargs="?", default="refine_1nwl/refine_summary.csv")
    args = parser.parse_args()

    logs = sorted(glob.glob(os.path.join(args.refine_dir, "*.log")))
    if not logs:
        print(f"No .log files found in {args.refine_dir}", file=sys.stderr)
        sys.exit(1)

    records = []
    for log in logs:
        stem = os.path.basename(log)
        # Expected naming: refine_####_{symop}_001.log
        stem = stem.replace("_001.log", "").lstrip("refine_")
        parts = stem.rsplit("_", 1)
        if len(parts) != 2:
            continue
        file_idx, symop = parts

        rw_start, rf_start, rw_final, rf_final = parse_log(log)

        # Parse elapsed time from last "Time:" line
        elapsed = float("nan")
        try:
            with open(log) as f:
                for line in f:
                    if line.strip().lower().startswith("time:"):
                        elapsed = float(line.strip().split()[-1])
        except Exception:
            pass

        records.append(dict(file_idx=file_idx, symop=symop,
                            Rw_start=rw_start, Rf_start=rf_start,
                            Rw_final=rw_final, Rf_final=rf_final,
                            **{"time(s)": elapsed}))

    df = pd.DataFrame(records)
    os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
    df.to_csv(args.output_csv, index=False)
    print(f"Wrote {len(df)} records to {args.output_csv}")
    print(f"  R-free range: {df['Rf_final'].min():.3f} – {df['Rf_final'].max():.3f}")
    print(f"  NaN rows:     {df['Rf_final'].isna().sum()}")


if __name__ == "__main__":
    main()
