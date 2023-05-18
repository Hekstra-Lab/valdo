import numpy as np
import reciprocalspaceship as rs
from tqdm import tqdm
import pandas as pd
import re
import glob
from scipy.ndimage import gaussian_filter
import os


def generate_preprocess_diff_map_blobs(input_files, model_folder, diff_col, phase_col, output_folder, cutoff=5, negate=False, sample_rate=3):
    
    def preprocess(matrix, radius_in_A=5):
        grid_spacing = np.min(matrix.spacing)
    
        matrix = np.absolute(matrix)
        radius_in_voxels = int(radius_in_A / grid_spacing)
        sigma = int(radius_in_voxels / 3)

        return gaussian_filter(matrix, sigma=sigma, radius=radius_in_voxels)
    
    peaks = []
    blob_stats = []

    for file in tqdm(input_files):

        sample = rs.read_mtz(file)[[diff_col, phase_col]].dropna()
        
        match = re.match(r".*(\d{4}).*", file)
        PTP1B_id = match.group(1)
        
        phases_file = glob.glob(f'{model_folder}/*{PTP1B_id}*.pdb')[0]        
        structure = gemmi.read_pdb(phases_file)

        sample_gemmi=sample.to_gemmi()
        grid = sample_gemmi.transform_f_phi_to_map(diff_col, phase_col, sample_rate=sample_rate)
        grid.normalize()
        
        blurred_grid = preprocess(grid)
        grid.set_subarray(blurred_grid, [0, 0, 0])
        grid.normalize()
        
        mean, sigma = np.mean(np.array(grid)), np.std(np.array(grid))
        
        blobs = gemmi.find_blobs_by_flood_fill(grid, cutoff=cutoff, negate=negate)

        use_long_names = False
        sort_by_key='peakz'

        ns = gemmi.NeighborSearch(structure[0], structure.cell, 5).populate()
        count = 0

        for blob in blobs:

            blob_stat = {
                "sample"  :    PTP1B_id,
                "peakz"   :    (blob.peak_value-mean)/sigma,
                "peak"    :    blob.peak_value,
                "score"   :    blob.score,
                "cenx"    :    blob.centroid.x,
                "ceny"    :    blob.centroid.y,
                "cenz"    :    blob.centroid.z,
                "volume"  :    blob.volume,
                "radius"  :    (blob.volume / (4/3 * np.pi)) ** (1/3)
            }
            
            if negate:
                negative_keys = ['peak', 'peakz', 'score', 'scorez']
                for k in negative_keys:
                    blob_stat[k] = -blob_stat[k]
                
            blob_stats.append(blob_stat)

            #This is a list of weird pointer objects. It is safest to convert them `gemmi.CRA` objects (see below)
            marks = ns.find_atoms(blob.centroid)
            if len(marks) == 0:
                continue

            cra = dist = None
            for mark in marks:
                image_idx = mark.image_idx
                cra = mark.to_cra(structure[0])
                dist = structure.cell.find_nearest_pbc_image(blob.centroid, cra.atom.pos, mark.image_idx).dist()

                record = {
                    "sample"  :    PTP1B_id,
                    "chain"   :    cra.chain.name,
                    "seqid"   :    cra.residue.seqid.num,
                    "residue" :    cra.residue.name,
                    "atom"    :    cra.atom.name,
                    "element" :    cra.atom.element.name,
                    "peakz"   :    (blob.peak_value-mean)/sigma,
                    "scorez"  :    (blob.score-mean)/sigma,
                    "peak"    :    blob.peak_value,
                    "score"   :    blob.score,
                    "cenx"    :    blob.centroid.x,
                    "ceny"    :    blob.centroid.y,
                    "cenz"    :    blob.centroid.z,
                    "coordx"  :    cra.atom.pos.x,
                    "coordy"  :    cra.atom.pos.y,
                    "coordz"  :    cra.atom.pos.z,
                }

                if negate:
                    negative_keys = ['peak', 'peakz', 'score', 'scorez']
                    for k in negative_keys:
                        record[k] = -record[k]
                peaks.append(record)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)   

    peaks_df = pd.DataFrame(peaks)
    blob_stats_df = pd.DataFrame(blob_stats)
    peaks_df.to_pickle(os.path.join(output_folder, 'peaks.pkl'))
    blob_stats_df.to_pickle(os.path.join(output_folder, 'blob_stats.pkl'))

    return peaks_df, blob_stats_df

def find_nearby_atoms(cen, structure_path, sample_no, radius=3):
    
    peaks = []
    
    structure = gemmi.read_pdb(structure_path)
    ns = gemmi.NeighborSearch(structure[0], structure.cell, radius).populate()
    centroid = gemmi.Position(cen["x"], cen["y"], cen["z"])
    marks = ns.find_atoms(centroid)
    
    for mark in marks:
        image_idx = mark.image_idx
        cra = mark.to_cra(structure[0])
        dist = structure.cell.find_nearest_pbc_image(centroid, cra.atom.pos, mark.image_idx).dist()

        record = {
            "sample"  :    sample_no,
            "chain"   :    cra.chain.name,
            "seqid"   :    cra.residue.seqid.num,
            "residue" :    cra.residue.name,
            "atom"    :    cra.atom.name,
            "element" :    cra.atom.element.name,
            "coordx"  :    cra.atom.pos.x,
            "coordy"  :    cra.atom.pos.y,
            "coordz"  :    cra.atom.pos.z,
            "dist"    :    dist
        }

        peaks.append(record)
        
    return pd.DataFrame(peaks)

def tag_cys_215_blobs(df, structure_path, radius=3):
    
    def check_blob_for_cys(row):
        
        sample = row["sample"]
        
        structure_file = glob.glob(f'{structure_path}/*{sample}*.pdb')[0]        
        cenx, ceny, cenz = row['cenx'], row['ceny'], row['cenz']
        atoms_df = find_nearby_atoms({"x": cenx, "y": ceny, "z": cenz}, structure_file, sample, radius)
        
        if len(atoms_df) < 1:
            return 0
        
        if 215 in set(atoms_df['seqid']):
            return 1
        return 0
    
    df['cys215'] = [1 if check_blob_for_cys(row) else 0 for i, row in tqdm(df.iterrows())]
    return df

def tag_lig_blobs(df, structure_path, radius=3):
    
    def check_blob_for_lig(row):
        
        if row["bound"] == 0:
            return 0
        
        sample = row["sample"]
        
        structure_file = glob.glob(f'{structure_path}/*{sample}*.pdb')[0]        
        cenx, ceny, cenz = row['cenx'], row['ceny'], row['cenz']
        atoms_df = find_nearby_atoms({"x": cenx, "y": ceny, "z": cenz}, structure_file, sample, row['radius'])
        
        if len(atoms_df) < 1:
            return 0
        
        if 'LIG' in set(atoms_df['residue']):
            return 1
        return 0
    
    df['ligand'] = [1 if check_blob_for_lig(row) else 0 for i, row in tqdm(df.iterrows())]
    return df


# function to find mtz file for sample number in folder
def find_mtz_file(sample_no, folder):
    for file_name in os.listdir(folder):
        if f"{sample_no}" in file_name and file_name.endswith(".mtz"):
            return os.path.join(folder, file_name)
    return None

# function to check if fractional coordinates are valid
def valid_fractional_coords(coords):
    
    valid_coords = np.array(coords)
    for i in range(3):
        while valid_coords[i] > 1:
            valid_coords[i] -= 1
        while valid_coords[i] < 0:
            valid_coords[i] += 1
    return valid_coords

# function to fractionalize coordinates and find smallest x value
def determine_locations(row, folder):
    # find mtz file for sample number
    mtz_file = find_mtz_file(row['sample'], folder)
    if mtz_file is None:
        return pd.Series({'fractional': np.nan, 'smallest_x_frac': np.nan, 'smallest_x_cart': np.nan})
    
    # read in mtz file
    sample_file = rs.read_mtz(mtz_file)
    
    # fractionalize coordinates using move2cell
    frac_coords = move2cell([row['cenx'], row['ceny'], row['cenz']], sample_file.cell)
    
    # apply symmetry operations and find smallest x value
    all_ops = list(sample_file.spacegroup.operations().sym_ops)

    all_possible = []
    for op in all_ops:
        result = op.apply_to_xyz(frac_coords)
        result = valid_fractional_coords(result)
        all_possible.append(result)
        
    smallest_x_frac = sorted(all_possible, key=lambda x: x[0])[0]
                
    # orthogonalize smallest x value fractional coordinates
    smallest_x_cart = sample_file.cell.orthogonalize(gemmi.Fractional(*smallest_x_frac))
    
    smallest_x_cart = np.array([smallest_x_cart.x, smallest_x_cart.y, smallest_x_cart.z])
    
    return pd.Series({'fractional': frac_coords, 'smallest_x_frac': smallest_x_frac, 'smallest_x_cart': smallest_x_cart})

def move2cell(cartesian_coordinates, unit_cell, fractionalize=True):
    '''
    Move your points into a unitcell with translational vectors
    
    Parameters
    ----------
    cartesian_coordinates: array-like
        [N_points, 3], cartesian positions of points you want to move
        
    unit_cell, gemmi.UnitCell
        A gemmi unitcell instance
    
    fractionalize: boolean, default True
        If True, output coordinates will be fractional; Or will be cartesians
    
    Returns
    -------
    array-like, coordinates inside the unitcell
    '''
    o2f_matrix = np.array(unit_cell.fractionalization_matrix)
    frac_pos = np.dot(cartesian_coordinates, o2f_matrix.T) 
    frac_pos_incell = frac_pos % 1
    for i in range(len(frac_pos_incell)):
        if frac_pos_incell[i] < 0:
            frac_pos_incell[i] += 1
    if fractionalize:
        return frac_pos_incell
    else:
        f2o_matrix = np.array(unit_cell.orthogonalization_matrix)
        return np.dot(frac_pos_incell, f2o_matrix.T)