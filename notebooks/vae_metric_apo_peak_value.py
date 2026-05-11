import pandas as pd
import numpy as np
import gemmi
import reciprocalspaceship as rs
from tqdm import tqdm
import glob, os

file_path = './ligand_cif_to_dataset_mapping.txt'

apo_samples = []
with open(file_path, 'r') as file:
    for line in file:
        if not line.strip().endswith('.cif'):
            apo_samples.append(line.strip()[-5:-1])

len(apo_samples)

my_dir                             = "/n/holyscratch01/hekstra_lab/dhekstra/valdo-tests/"
basepath                           = my_dir + 'pipeline/'

# VAE METRIC APO PEAK VALUE

vae_reconstructed_with_phases_path = basepath + 'vae/reconstructed_w_phases/'

peak_values = []
column='WDF'
for pdbid in tqdm(apo_samples):
    print(glob.glob(os.path.join(vae_reconstructed_with_phases_path, f"{pdbid}*.mtz")))
    try:
        mtz_file = gemmi.read_mtz_file(glob.glob(os.path.join(vae_reconstructed_with_phases_path, f"{pdbid}*.mtz"))[0])
        real_grid = mtz_file.transform_f_phi_to_map(column, 'refine_PH2FOFCWT', sample_rate=3.0)
        real_grid.normalize()

        peak_values.append(np.max(real_grid))
    except:
        print(f"Can't handle {pdbid}")

print("Mean Highest Peak Value in Apo models: ", np.mean(peak_values))

# Zmap mean peak value as metric

z_maps_path = basepath + 'data/z_maps/'

peak_values = []

for pdbid in tqdm(apo_samples):
    zmap = gemmi.read_ccp4_map(z_maps_path + f'PTP1B-y{pdbid}-z_map.native.ccp4')

    real_grid = zmap.grid
    real_grid.normalize()

    peak_values.append(np.max(real_grid))

print("Mean Highest Peak Value in Apo models: ", np.mean(peak_values))
