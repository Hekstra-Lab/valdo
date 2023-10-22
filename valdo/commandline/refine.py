"""
Prepare phase with phenix automatic refinement

Examples
--------
Refine all reindexed mtz files starting from a single apo model 
    > valdo.refine --pdbpath "xxx/xxx_apo.pdb" --mtzpath "xxx/*.mtz" --output "yyy/" --eff "xxx/refine_drug.eff"
"""

import argparse
import subprocess
import glob, os

class ArgumentParser(argparse.ArgumentParser):
    def __init__(self):
        super().__init__(
            formatter_class=argparse.RawTextHelpFormatter, description=__doc__
        )

        self.add_argument(
            "--pdbpath", type=str, help="Path to the initial PDB model(s)"
        )

        self.add_argument(
            "--mtzpath", type=str, help="Path to the mtz data to be refined"
        )

        self.add_argument(
            "-o", "--output", type=str, help="Output path prefix"
        )

        self.add_argument(
            "-e", "--eff", type=str, help="Path to the phenix refine eff file"
        )

def main():
    args = ArgumentParser().parse_args()
    pdb_files = glob.glob(args.pdbpath)
    mtz_files = glob.glob(args.mtzpath)
    
    if len(pdb_files) > 1:
        assert len(pdb_files) == len(mtz_files), "Ensure that the number of provided pdb files matches the number of mtz files!"
        pdb_files.sort()
        mtz_files.sort()
        for pdbfile, mtzfile in zip(pdb_files, mtz_files):
            mtz_name = os.path.splitext(os.path.basename(mtzfile))[0]
            output_prefix = os.path.join(args.output, f'refine_{mtz_name}')
            subprocess.run(["phenix.refine", 
                            pdbfile, 
                            mtzfile, 
                            f"output.prefix={output_prefix})",
                            f"{args.eff}", "--overwrite"])
    else:
        print("Only one PDB file is provided, will be used for all refinement initialization!", flush=True)
        pdbfile = pdb_files[0]
        for mtzfile in mtz_files:
            mtz_name = os.path.splitext(os.path.basename(mtzfile))[0]
            output_prefix = os.path.join(args.output, f'refine_{mtz_name}')
            subprocess.run(["phenix.refine", 
                            pdbfile, 
                            mtzfile, 
                            f"output.prefix={output_prefix}",
                            f"{args.eff}", "--overwrite"])