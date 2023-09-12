import pandas as pd
import reciprocalspaceship as rs
import gemmi
import numpy as np
import os
from tqdm import tqdm
from multiprocessing import Pool
from itertools import repeat
import pickle


def weighted_pearsonr(ds1,ds2,data_col="F-obs"):
    def wcorr(group):
        x = group[data_col + '_1'].to_numpy().flatten()
        y = group[data_col + '_2'].to_numpy().flatten()
        w = group['W'].to_numpy().flatten()
        return rs.utils.weighted_pearsonr(x,y,w)

    mergedi = ds1.merge(ds2, left_index=True, right_index=True, suffixes=('_1', '_2'), check_isomorphous=False)
    mergedi.assign_resolution_bins(bins=20, inplace=True)
    
    quad_var=mergedi["SIGF-obs_1"].to_numpy()**2 + mergedi["SIGF-obs_2"].to_numpy()**2
    w=1/quad_var
    mergedi["W"]=w
    
    grouped=mergedi.groupby("bin")
    result=grouped.apply(wcorr).mean()
    return result

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
            try:
                input_df = rs.read_mtz(input_file)[columns]
            except PermissionError:
                print("read_mtz error: " + input_file + e)
                continue
            corr_ref = []
            for op in try_ops:
                symopi_asu = input_df.apply_symop(op).hkl_to_asu()
                mergedi = reference_asu.merge(symopi_asu, left_index=True, right_index=True, suffixes=('_ref', '_input'), check_isomorphous=False)
                mergedi.assign_resolution_bins(bins=20, inplace=True)
                quad_var=mergedi[columns[1]+"_ref"].to_numpy()**2 + mergedi[columns[1]+"_input"].to_numpy()**2
                w=1/quad_var
                
                corr_ref.append(np.corrcoef(mergedi[columns[0]+'_ref'], mergedi[columns[0]+'_input'])[0][1])
            i = np.argmax(corr_ref)
            output_file = os.path.join(output_folder, os.path.splitext(os.path.basename(input_file))[0] + f"_{i}" + ".mtz")
            symopi_asu = input_df.apply_symop(try_ops[i]).hkl_to_asu()
            symopi_asu.write_mtz(output_file)
            reindexed_record.append([i,output_file, corr_ref]) # if i == 0, no reindex
        except Exception as e:
            print(input_file + e)
            continue
    
    with open(os.path.join(output_folder, 'reindex_record.pkl'), "wb") as f:
        # Use the pickle module to dump the list to the file
        pickle.dump(reindexed_record, f)    
        
    return reindexed_record

# &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
# Parallelized Version

# def chunks(lst, n):
#     """Yield successive n-sized chunks from lst."""
#     list_of_list_chunks=[]
#     for i in range(0, len(lst), n):
#         list_of_list_chunks.append(lst[i:i + n])
#     return list_of_list_chunks
        
def reindex_from_pool_map(input_file, additional_args):
    """
    Reindexes a single input MTZ files to a reference MTZ file using gemmi.
    Specialized variant of reindex_files for use with multiprocessing. 

    Parameters:
    input_file (str) : List of paths to input MTZ files.
    additional_args (list) : List of additional input arguments, containing:
        try_ops_list (list of str) : list of string-based designations of alternative indexing operations
        reference_asu (rs dataSet) : Reference ASU for comparison
        output_folder (str) : Path to folder where reindexed MTZ files will be saved.
        columns (list of two str) : column labels for amplitudes and errors in amplitudes

    Returns:
    List[str]] [i, path_to_file] with: if i == 0, no reindex
    """

    try_ops_list  = additional_args[0]
    reference_asu = additional_args[1]
    output_folder = additional_args[2]
    columns       = additional_args[3]

    # Reindex each input MTZ file with all possible ops
    try:
        # print("input file: " + input_file)
        input_df = rs.read_mtz(input_file)[columns]
        corr_ref = []
        try_ops=[gemmi.Op(op) for op in try_ops_list]
        for op in try_ops:
            symopi_asu = input_df.apply_symop(op).hkl_to_asu()
            corr_ref.append(weighted_pearsonr(reference_asu,symopi_asu,data_col="F-obs"))
            # mergedi = reference_asu.merge(symopi_asu, left_index=True, right_index=True, suffixes=('_ref', '_input'), check_isomorphous=False)
            # corr_ref.append(np.corrcoef(mergedi[columns[0]+'_ref'], mergedi[columns[0]+'_input'])[0][1])
        i = np.argmax(corr_ref)
        output_file = os.path.join(output_folder, os.path.splitext(os.path.basename(input_file))[0] + f"_{i}" + ".mtz")
        symopi_asu = input_df.apply_symop(try_ops[i]).hkl_to_asu()
        symopi_asu.write_mtz(output_file)
        reindexed_record = [i,output_file, corr_ref] # if i == 0, no reindex
    except Exception as e:
        print("For " + input_file + " the following occurred: " )
        print(e)
    return reindexed_record


def find_alt_ops(reference_mtz, columns):
    """
    Find the alternative indexing settings based on a reference MTZ file.
    Helper function for reindex_files_pool()

    Parameters:
    reference_mtz (str) : Reference MTZ file.
    columns (list of two str) : column labels to use
    
    Returns:
    try_ops_list (list of gemmi.Op) : list of string-based designations of alternative indexing operations
    reference_asu (rs dataSet) : Reference ASU for comparison
    """

    reference = rs.read_mtz(reference_mtz)[columns]
    reference_asu = reference.hkl_to_asu()

    # Find the possible reindex ambiguity ops
    unit_ops = [gemmi.Op("x,y,z")]
    alt_ops = gemmi.find_twin_laws(reference.cell, reference_asu.spacegroup, max_obliq=3, all_ops=False)
    if len(alt_ops) == 0:
        # print("No ambiguity for this spacegroup! No need to reindex!")
        return unit_ops, reference_asu
    else:
        try_ops = unit_ops + alt_ops
    return try_ops, reference_asu

def reindex_files_pool(input_files, reference_file, output_folder, columns=['F-obs', 'SIGF-obs'],ncpu=None):
    """
    Reindexes a list of input MTZ files to a reference MTZ file using gemmi.
    For use with multiprocessing

    Parameters:
    input_files (list of str) : List of paths to input MTZ files.
    reference_file (str) : Path to reference MTZ file.
    output_folder (str) : Path to folder where reindexed MTZ files will be saved.
    columns (list of str) : dataSet keys for the F and sigF to be used.
    ncpu (int) : number of logical CPUs to use (default None: use all available)

    Returns:
    List[List[str]]: [[i, path_to_file],...]
    if i == 0, no reindex
    """
    try_ops, reference_asu = find_alt_ops(reference_file, columns)
    # file_lists = chunks(input_files, ncpu)
    
    # since we can't pickle gemmi symops
    try_ops_triplets = [op.triplet() for op in try_ops] 
    additional_args=[try_ops_triplets, reference_asu, output_folder, columns]
    # print(repeat(additional_args))
    input_args = zip(input_files, repeat(additional_args))
    if len(try_ops)>1:
        if ncpu is None:
            with Pool() as pool:
                result = pool.starmap(reindex_from_pool_map, tqdm(input_args, total=len(input_files)))
        else:
            with Pool(ncpu) as pool:
                result = pool.starmap(reindex_from_pool_map, tqdm(input_args, total=len(input_files)))
    else:
        print("No reindexing required!")
        result = None
        
    with open(os.path.join(output_folder, 'reindex_record.pkl'), "wb") as f:
        # Use the pickle module to dump the list to the file
        pickle.dump(result, f)    
        
    return result #reindexed_record