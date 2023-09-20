import numpy as np
import reciprocalspaceship as rs
from tqdm import tqdm
from multiprocessing import Pool
from itertools import repeat
import pandas as pd
import math
import re
import glob
import os
import gemmi


def valid_fractional_coords(fractional_coords):
    """
    Converts fractional coordinates to valid fractional coordinates within the range [0, 1).

    Args:
        fractional_coords (list or numpy.ndarray): The input fractional coordinates.

    Returns:
        numpy.ndarray: The converted valid fractional coordinates within the range [0, 1).

    """
    
    valid_coords = np.array(fractional_coords)
    for i in range(3):
        while valid_coords[i] > 1:
            valid_coords[i] -= 1
        while valid_coords[i] < 0:
            valid_coords[i] += 1
    return valid_coords


def find_nearby_atoms(centroid_dict, structure_path, sample_no=None, radius=3):
    
    """
    Finds nearby atoms within a specified radius around a given centroid position in a structure file and returns the atom details as a DataFrame.

    The function reads the structure file in PDB format using gemmi, performs a neighbor search within the specified radius around the centroid position, and retrieves information about the nearby atoms. The atom details include the sample number, chain name, residue sequence ID, residue name, atom name, element name, coordinates (x, y, z), and distance from the centroid.

    Args:
        centroid_dict (dict): Dictionary containing the centroid position with keys 'x', 'y', and 'z'.
        structure_path (str): Path to the structure file in PDB format.
        sample_no (str): Sample number or identifier. Optional.
        radius (float, optional): Radius in angstroms to search for nearby atoms. Default is 3.

    Returns:
        pandas.DataFrame: DataFrame containing the details of the nearby atoms.

    Example:
        centroid = {'x': 10.0, 'y': 20.0, 'z': 30.0}
        structure_file = './data/structure.pdb'
        sample_number = 'S001'
        nearby_atoms = find_nearby_atoms(centroid, structure_file, sample_number, radius=5)
    """
    
    peaks = []
    
    structure = gemmi.read_pdb(structure_path)
    ns = gemmi.NeighborSearch(structure[0], structure.cell, radius).populate()
    centroid = gemmi.Position(centroid_dict["x"], centroid_dict["y"], centroid_dict["z"])
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


def check_blob_for_lig(row, additional_args):
    """
    Helper function for tag_lig_blobs() (below)
    Args:
        row (pandas.Series): A row of the DataFrame representing a blob.
        additional_args (list) : Additional arguments passed on from tag_lig_blobs().
        
    Returns:
        row (pandas.Series): with field 'ligand' added with value 1 if the blob contains ligands, otherwise 0.
    """
    structure_path=additional_args[0]
    sample = row["sample"]

    row['ligand'] = 0
    if row["bound"] == 0:
        return row
    
    cenx, ceny, cenz = row['cenx'], row['ceny'], row['cenz']
    atoms_df = find_nearby_atoms({"x": cenx, "y": ceny, "z": cenz}, glob.glob(os.path.join(structure_path, f'{sample.split("_")[0]}.pdb'))[0], sample, row['radius'])

    if len(atoms_df) < 1:
        return row
    
    if 'LIG' in set(atoms_df['residue']):
        # print(row)
        row['ligand']=1

    return row

    
def tag_lig_blobs(df, structure_path, ncpu=1):
    """
    Tags the blobs in the DataFrame 'df' that contain ligands based on the nearby atoms found in PDB files.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the blobs information.
        structure_path (str): The path to the folder containing the PDB files used for identifying nearby atoms.

    Returns:
        pandas.DataFrame: The modified DataFrame with an additional 'ligand' column indicating the presence (1) or absence (0) of ligands in the blobs.

    """

    # df['ligand']=0
    additional_args=[structure_path]
    input_args = zip(df.to_dict('records'),repeat(additional_args))
    if ncpu>1:
        with Pool(ncpu) as pool:
            result = pool.starmap(check_blob_for_lig, tqdm(input_args, total=len(df)))
        
        # merge results back together
        df = pd.DataFrame.from_records(result)
    else:
        # tqdm.pandas()
        df = df.apply(check_blob_for_lig,args=(additional_args,), axis=1)
        
    return df


def apply_determine_locations(row, additional_args): 
    """
    Converts coordinates to fractional form and determines all symmetry-related cartesian points for the given row.
    Called by determine_locations()

    Args:
        row (pandas.Series): A row of the DataFrame representing a blob.
        additional_args (list): Additional args passed on from determine_locations()

    Returns:
        pandas.Series: A pandas Series containing the fractional coordinates, all possible fractional coordinates, and all possible cartesian coordinates.

    """
    folder=additional_args[0]
    
    # find mtz file for sample number
    mtz_file = folder + row['sample'] + '.mtz'
    if mtz_file is None:
        return pd.Series({'fractional': np.nan, 'all_possible_frac': np.nan, 'all_possible_cart': np.nan})
    
    # read in mtz file
    sample_file = rs.read_mtz(mtz_file)
    
    # fractionalize coordinates using move2cell
    frac_coords = move2cell([row['cenx'], row['ceny'], row['cenz']], sample_file.cell)
    
    # identify all symmetry operations
    all_ops = list(sample_file.spacegroup.operations().sym_ops)

    all_possible_frac = []
    for op in all_ops:
        result = op.apply_to_xyz(frac_coords)
        result = valid_fractional_coords(result)
        all_possible_frac.append(result)
        
    all_possible_frac = sorted(all_possible_frac, key=lambda x: x[0])
                
    # orthogonalize fractional coordinates
    all_possible_cart = [sample_file.cell.orthogonalize(gemmi.Fractional(*elt)) for elt in all_possible_frac]
    
    all_possible_cart = [np.array([elt.x, elt.y, elt.z]) for elt in all_possible_cart]
    
    return pd.Series({'fractional': frac_coords, 'all_possible_frac': all_possible_frac, 'all_possible_cart': all_possible_cart})


def determine_locations(df, folder, ncpu=1):
    """
    Converts coordinates to fractional form and determines all symmetry-related cartesian points for the given row.
    Wraps around apply_determine_locations(), which does the actual work

    Args:
        row (pandas.Series): A row of the DataFrame representing a blob.
        folder (str): The path to the folder containing the mtz files.

    Returns:
        pandas.Series: A pandas Series containing the fractional coordinates, all possible fractional coordinates, and all possible cartesian coordinates.

    """

    additional_args=[folder]
    input_args = zip(df.to_dict('records'),repeat(additional_args))
    if ncpu>1:
        with Pool(ncpu) as pool:
            result = pool.starmap(apply_determine_locations, tqdm(input_args, total=len(df)))
            result_df=pd.concat(result,axis=1).transpose()

        if not df.index.equals(result_df.index):
            # Should not happen, but let's be sure.
            print("Warning: apply_determine_locations returns a dataframe with different index than the input df.")
            
        df = pd.concat([df, result_df],axis=1,join='outer')
    else:
        # tqdm.pandas()
        df[['fractional', 'all_possible_frac', 'all_possible_cart']] = df.apply(apply_determine_locations, args=(additional_args,), axis=1)
    
    return df


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


def mark_duplicates(blobs_df, distance_cutoff=1):
    
    """
    Marks duplicate blobs in the DataFrame based on proximity in cartesian coordinates. 
    Checks on a per-sample basis, checking blobs with adjacent peak values.

    Args:
        blobs_df (pandas.DataFrame): The input DataFrame containing the blob information.
        distance_cutoff (float or int): distance between mapped positions below which blobs are considered duplicate (default: 1.0 A)

    Returns:
        pandas.DataFrame: The modified DataFrame with an additional 'duplicate' column indicating duplicate blobs.

    """
    
    blobs_df = blobs_df.sort_values(by='peak', ascending=False)
    blobs_df['duplicate'] = 0  # Initialize 'duplicate' column with 0
    
    def check_euclidean_distance(list1, list2, distance_cutoff=1):
        
        """
        Checks if the distance between any pairwise points in two lists is less than 1 A.

        Args:
            list1 (list): The first list of cartesian coordinates.
            list2 (list): The second list of cartesian coordinates.

        Returns:
            bool: Returns True if any distance is less than 1 A, otherwise returns False.

        """
        
        for point1 in list1:
            for point2 in list2:
                distance = math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2 + (point2[2] - point1[2])**2)
                if distance < distance_cutoff:
                    return True
        return False
    
    grouped = blobs_df.groupby('sample')
    
    for _, group in grouped:
        if len(group) > 1:
            # print(group['all_possible_cart'])
            # all_possible_cart_lists = group['all_possible_cart'].tolist() # <<
            all_possible_cart_lists = group['all_possible_cart'].values.tolist() # works for ncpu=1 output
            for i in range(1, len(all_possible_cart_lists)):
                if check_euclidean_distance(all_possible_cart_lists[i-1], all_possible_cart_lists[i], distance_cutoff):
                    blobs_df.at[group.index[i], 'duplicate'] = 1
    
    return blobs_df


def check_blob_for_nearby_seqid(row, additional_args):
    """
    Does the actual work for tag_blobs_around_seqid() (below).
    
    Args:
        row (pandas.Series): A row of the DataFrame representing a blob.
        additional_args (list) : additional arguments passed on by tag_blobs_around_seqid().

    Returns:
        int: Tags a column (tag) as 1 if the blob is near the specified residue (seqid), otherwise 0.

    Example: for PTP1B, we want to label hits near seqid 215 (Cys 215) with a tag 'cys215'.
    """
    
    structure_path=additional_args[0]
    radius        =additional_args[1]
    focal_seqid   =additional_args[2]
    tag           =additional_args[3]
    sample = row["sample"]
    
    cenx, ceny, cenz = row['cenx'], row['ceny'], row['cenz']
    atoms_df = find_nearby_atoms({"x": cenx, "y": ceny, "z": cenz}, glob.glob(os.path.join(structure_path, f'*{sample}*.pdb'))[0], sample, radius)

    row[tag]=0
    if len(atoms_df) < 1:
        return row
    
    # if 215 in set(atoms_df['seqid']):
    #     return 1
    # return 0
    try:
        if focal_seqid in set(atoms_df['seqid']):
            row[tag]=1
    except Exception as e:
        print(e)
        print(atoms_df.info())
        raise
    
    return row

def tag_blobs_around_seqid(df, structure_path, radius=3,tag='cys215', focal_seqid=215, ncpu=1):
    
    """
    Tags the blobs in the DataFrame 'df' near a residue of interest (e.g. Cys 215 for PTP1B), based on the nearby atoms found in PDB files.

    Args:
        df (pandas.DataFrame): The input DataFrame containing the blobs information.
        structure_path (str): The path to the folder containing the PDB files used for identifying nearby atoms.
        radius (int, optional): The radius in Angstroms for finding nearby atoms. Default is 3.
        tag (str): key for df column indicating a blob is near (1) or not near (0) the focal residue (default: 'cys215').
        seqid (int): value of the residue number around which blobs are tagged (default: 215)
        ncpu (int): number of available CPU, to decide whether to use multiprocessing.

    Returns:
        pandas.DataFrame: The modified DataFrame with an additional 'cys215' column indicating the presence (1) or absence (0) of CYS 215 in the blobs.
    """
    
    
    df[tag]=0
    additional_args=[structure_path,radius,focal_seqid,tag]
    input_args = zip(df.to_dict('records'),repeat(additional_args))
    if ncpu>1:
        with Pool(ncpu) as pool:
            result = pool.starmap(check_blob_for_nearby_seqid, tqdm(input_args, total=len(df)))

        # merge results back together
        # print(result)
        df_result = pd.DataFrame.from_records(result)
        df[tag] = df_result[tag]
    else:
        # tqdm.pandas()
        try:
            df = df.apply(check_blob_for_nearby_seqid, axis=1, args=(additional_args,))
        except Exception as e:
            print(e)
    
    return df