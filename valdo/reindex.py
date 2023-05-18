import pandas as pd
import reciprocalspaceship as rs
import numpy as np
import os
from tqdm import tqdm
import pickle

def reindex_files(input_files, reference_file, output_folder):
    """
    Reindexes a list of input MTZ files to a reference MTZ file using gemmi.

    Parameters:
    input_files (list of str): List of paths to input MTZ files.
    reference_file (str): Path to reference MTZ file.
    output_folder (str): Path to folder where reindexed MTZ files will be saved.

    Returns:
    list of str: List of paths to reindexed MTZ files.
    """

    # Read the reference MTZ file
    reference = rs.read_mtz(reference_file)[['F-obs', 'SIGF-obs']]
    reference_asu = reference.hkl_to_asu()

    # Reindex each input MTZ file
    reindexed_files = []
    for input_file in tqdm(input_files):
        try:
            # Read the input MTZ file
            input_df = rs.read_mtz(input_file)[['F-obs', 'SIGF-obs']]
            symop1_asu = input_df.apply_symop(gemmi.Op("x,y,z")).hkl_to_asu()
            symop2_asu = input_df.apply_symop(gemmi.Op("-x,-y,z")).hkl_to_asu()

            # Merge with reference MTZ file
            merged1 = reference_asu.merge(symop1_asu, left_index=True, right_index=True, suffixes=('_ref', '_input'), check_isomorphous=False)
            merged2 = reference_asu.merge(symop2_asu, left_index=True, right_index=True, suffixes=('_ref', '_input'), check_isomorphous=False)
            
            # Compute correlations
            corr_ref_input1 = np.corrcoef(merged1['F-obs_ref'], merged1['F-obs_input'])[0][1]
            corr_ref_input2 = np.corrcoef(merged2['F-obs_ref'], merged2['F-obs_input'])[0][1]
            # Choose the better correlation and write the reindexed file
            if corr_ref_input1 > corr_ref_input2:
                output_file = os.path.join(output_folder, os.path.basename(input_file))
                symop1_asu.write_mtz(output_file)
            else:
                output_file = os.path.join(output_folder, os.path.basename(input_file))
                symop2_asu.write_mtz(output_file)
                reindexed_files.append(output_file)
        except:
            continue
    with open(os.path.join(output_folder, 'reindexed_files.pkl'), "wb") as f:
        # Use the pickle module to dump the list to the file
        pickle.dump(reindexed_files, f)    
        
    return reindexed_files