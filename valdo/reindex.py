import pandas as pd
import reciprocalspaceship as rs
import gemmi
import numpy as np
import os
from tqdm import tqdm
import pickle

def reindex_files(input_files, reference_file, output_folder, columns=['F-obs', 'SIGF-obs']):
    """
    Reindexes a list of input MTZ files to a reference MTZ file using gemmi.

    Parameters:
    input_files (list of str) : List of paths to input MTZ files.
    reference_file (str) : Path to reference MTZ file.
    output_folder (str) : Path to folder where reindexed MTZ files will be saved.

    Returns:
    List[List[str]]: [[i, path_to_file],...]
    if i == 0, no reindex
    """

    # Read the reference MTZ file
    reference = rs.read_mtz(reference_file)[columns]
    reference_asu = reference.hkl_to_asu()

    # Find the possible reindex ambiguity ops
    unit_ops = [gemmi.Op("x,y,z")]
    alt_ops = gemmi.find_twin_laws(reference.cell, reference_asu.spacegroup, max_obliq=3, all_ops=False)
    if len(alt_ops) == 0:
        print("No ambiguity for this spacegroup! No need to reindex!")
        return None
    else:
        try_ops = unit_ops + alt_ops

    # Reindex each input MTZ file with all possible ops
    reindexed_record = []
    for input_file in tqdm(input_files):
        try:
            # Read the input MTZ file
            input_df = rs.read_mtz(input_file)[columns]
            corr_ref = []
            for op in try_ops:
                symopi_asu = input_df.apply_symop(op).hkl_to_asu()
                mergedi = reference_asu.merge(symopi_asu, left_index=True, right_index=True, suffixes=('_ref', '_input'), check_isomorphous=False)
                corr_ref.append(np.corrcoef(mergedi[columns[0]+'_ref'], mergedi[columns[0]+'_input'])[0][1])
            i = np.argmax(corr_ref)
            output_file = os.path.join(output_folder, os.path.basename(input_file))
            symopi_asu = input_df.apply_symop(try_ops[i]).hkl_to_asu()
            symopi_asu.write_mtz(output_file)
            reindexed_record.append([i,output_file]) # if i == 0, no reindex
        except Exception as e:
            print(input_file + e)
            continue
    
    with open(os.path.join(output_folder, 'reindex_record.pkl'), "wb") as f:
        # Use the pickle module to dump the list to the file
        pickle.dump(reindexed_record, f)    
        
    return reindexed_record