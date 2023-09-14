import pandas as pd
import reciprocalspaceship as rs
import gemmi
import numpy as np
import os
from tqdm import tqdm
from multiprocessing import Pool
from itertools import repeat
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors       import KNeighborsRegressor
import pickle
# from valdo import knn_tools as knn


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

def reindex_files(input_files, reference_file, output_folder, columns=['F-obs', 'SIGF-obs'], wcorr=False, check_isomorphous=False, cc_min_dif=0.2):
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
                if wcorr:
                    corr_ref.append(weighted_pearsonr(reference_asu,symopi_asu,data_col="F-obs"))
                else:
                    mergedi = reference_asu.merge(symopi_asu, left_index=True, right_index=True, suffixes=('_ref', '_input'), check_isomorphous=check_isomorphous)
                    corr_ref.append(np.corrcoef(mergedi[columns[0]+'_ref'], mergedi[columns[0]+'_input'])[0][1])

            argorder = np.argsort(corr_ref)[::-1]
            max_cc = corr_ref[argorder[0]]
            n_duplicates = 0
            for i in argorder:
                if max_cc - corr_ref[i] < cc_min_dif:
                    output_file = os.path.join(output_folder, os.path.splitext(os.path.basename(input_file))[0] + f"_{i}" + ".mtz")
                    symopi_asu = input_df.apply_symop(try_ops[i]).hkl_to_asu()
                    symopi_asu.write_mtz(output_file)
                    n_duplicates += 1
            reindexed_record.append([os.path.splitext(os.path.basename(input_file))[0], argorder[0], n_duplicates, output_file, *corr_ref]) # if i == 0, no reindex
        except Exception as e:
            print(input_file + e)
            continue
    df_record = pd.DataFrame(reindexed_record)
    df_record.columns=['file_idx', 'best_symop', 'num_duplicates', 'reindexed_file', *[f"CC_symop{i}" for i in range(len(try_ops))]]
    df_record.to_pickle(os.path.join(output_folder, 'reindex_record.pkl'))
    print("Reindex statistics record has been saved at:", flush=True)
    print(os.path.join(output_folder, 'reindex_record.pkl'), flush=True) 
    return df_record


def ds_add_rs(ds, force_rs=False, inplace=True):
    """
    Adds three columns to an rs dataframe with the reciprocal space coordinates (in A^-1) for each Miller index.
    Inplace by default!
    """
    if force_rs or (not "rs_a" in ds.keys()):
        orthomat_list = ds.cell.orthogonalization_matrix.tolist()
        orthomat = np.asarray(orthomat_list)

        hkl_array = np.asarray(list(ds.index.to_numpy()))

        orthomat_inv_t = np.linalg.inv(orthomat).transpose()
        S = np.matmul(orthomat_inv_t, hkl_array.transpose())

        if inplace == True:
            ds["rs_a"] = S.transpose()[:, 0]
            ds["rs_b"] = S.transpose()[:, 1]
            ds["rs_c"] = S.transpose()[:, 2]
        else:
            ds_out = ds.copy()
            ds_out["rs_a"] = S.transpose()[:, 0]
            ds_out["rs_b"] = S.transpose()[:, 1]
            ds_out["rs_c"] = S.transpose()[:, 2]
    else:
        if inplace == True:
            pass
        else:
            ds_out = ds.copy()
            ds_out["rs_a"] = ds["rs_a"]
            ds_out["rs_b"] = ds["rs_b"]
            ds_out["rs_c"] = ds["rs_c"]
    if inplace == True:
        return  # already done
    else:
        return ds_out


def apply_local_scale(ds_input, ds_ref, columns=['F-obs', 'SIGF-obs'],corrected_input_only=True, gridsearch=False,ncpu=1):
    """
    Scale ds_input to ds_ref.

    Parameters:
    ds_ref(rs.DataSet) : DataSet to locally scale to. Should be mapped to ASU already.
    ds_input(rs.DataSet) : DataSet to locally scale.  Should be mapped to ASU already.
    columns (list of str): names of Fobs and SigFobs columns
    gridsearch (bool): whether to choose optimal regression kernels by crossvalidation (False will default to hard-coded values)

    Returns:
    scaled_ds (rs.DataSet): scaled version of ds_ref
    """
    if not ("rs_a" in ds_ref.columns):
        ds_add_rs(ds_ref,inplace=True)
    if not ("CENTRIC" in ds_ref.columns):
        ds_ref.label_centrics(inplace=True)
    if not ("EPSILON" in ds_ref.columns):
        ds_ref.compute_multiplicity(inplace=True)        
    
    
    ds12=ds_input.merge(ds_ref, left_index=True, right_index=True, suffixes=("_input", "_ref"))
    ds12["RATIO"]=ds12[columns[0]+"_input"]/ds12[columns[0]+"_ref"]
    ds12["SIGRATIO"]=np.abs(ds12["RATIO"])*np.sqrt((ds12[columns[1]+"_input"]/ds12[columns[0]+"_input"])**2 +\
                                                   (ds12[columns[1]+"_ref"]/  ds12[columns[0]+"_ref"]  )**2)
    ds12_filtered=ds12.loc[ds12["SIGRATIO"]<np.percentile(ds12["SIGRATIO"],95),["RATIO","rs_a","rs_b","rs_c"]]

    param_grid={"n_neighbors":[50,100,200,400,800, 1200,1600],'weights':['uniform',knn.knn_weight_exp_p05, knn.knn_weight_norm_p05]}
    if gridsearch: 
        knn_1 = GridSearchCV(KNeighborsRegressor(n_jobs=ncpu),param_grid=param_grid)
        knn_1.fit(ds12_filtered[["rs_a", "rs_b", "rs_c"]].to_numpy(), (ds12_filtered["RATIO"].to_numpy())) # these should be corrected for eps already
    else:
        knn_1 = KNeighborsRegressor(800, weights=knn.knn_weight_exp_p05,n_jobs=ncpu)
        knn_1.fit(ds12_filtered[["rs_a", "rs_b", "rs_c"]], (ds12_filtered[["RATIO"]].to_numpy()))

    inferred_ratio = knn_1.predict(ds12[["rs_a", "rs_b", "rs_c"]]).reshape(-1,1).flatten()
    ds12[columns[0]+"_input_corr"] = ds12[columns[0]+"_input"].to_numpy()/inferred_ratio
    ds12[columns[1]+"_input_corr"] = ds12[columns[1]+"_input"].to_numpy()/inferred_ratio

    if corrected_input_only:
        mapper={columns[0]+"_input_corr":columns[0],columns[1]+"_input_corr":columns[1]}
        ds_out=ds12[[columns[0]+"_input_corr",columns[1]+"_input_corr"]].rename(mapper=mapper)
    else:
        ds_out=ds12
    # print('-'*50)
    print(np.mean(ds12[columns[0]+"_input"]/ds12[columns[0]+"_ref"]))
    print(np.mean(ds12[columns[0]+"_input_corr"]/ds12[columns[0]+"_ref"]))
    return ds_out



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
    wcorr         = additional_args[4]
    check_isomorphous = additional_args[5]
    cc_min_dif = additional_args[6]

    # Reindex each input MTZ file with all possible ops
    try:
        # print("input file: " + input_file)
        input_df = rs.read_mtz(input_file)[columns]
        corr_ref = []
        try_ops=[gemmi.Op(op) for op in try_ops_list]
        for op in try_ops:
            symopi_asu = input_df.apply_symop(op).hkl_to_asu()
            # symopi_asu = apply_local_scale(symopi_asu, reference_asu,corrected_input_only=True)
            if wcorr:
                corr_ref.append(weighted_pearsonr(reference_asu,symopi_asu,data_col="F-obs"))
            else:
                mergedi = reference_asu.merge(symopi_asu, left_index=True, right_index=True, suffixes=('_ref', '_input'), check_isomorphous=check_isomorphous)
                corr_ref.append(np.corrcoef(mergedi[columns[0]+'_ref'], mergedi[columns[0]+'_input'])[0][1])
        argorder = np.argsort(corr_ref)[::-1]
        max_cc = corr_ref[argorder[0]]
        n_duplicates = 0
        for i in argorder:
            if max_cc - corr_ref[i] < cc_min_dif:
                output_file = os.path.join(output_folder, os.path.splitext(os.path.basename(input_file))[0] + f"_{i}" + ".mtz")
                symopi_asu = input_df.apply_symop(try_ops[i]).hkl_to_asu()
                symopi_asu.write_mtz(output_file)
                n_duplicates += 1
        reindexed_record = [os.path.splitext(os.path.basename(input_file))[0], argorder[0], n_duplicates, output_file, *corr_ref] # if i == 0, no reindex
    except Exception as e:
        print("For " + input_file + " the following occurred: \n" )
        print(e, flush=True)
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

def reindex_files_pool(input_files, reference_file, output_folder, columns=['F-obs', 'SIGF-obs'], check_isomorphous=False, wcorr=True, ncpu=None, cc_min_dif=0.2):
    """
    Reindexes a list of input MTZ files to a reference MTZ file using gemmi.
    For use with multiprocessing

    Parameters:
    input_files (list of str) : List of paths to input MTZ files.
    reference_file (str) : Path to reference MTZ file.
    output_folder (str) : Path to folder where reindexed MTZ files will be saved.
    columns (list of str) : dataSet keys for the F and sigF to be used.
    check_isomorphous (bool): whether to check isomorphism with the reference dataset (default: False)
    wcorr (bool): whether to use a weighted correlation coefficient (default: True)
    ncpu (int) : number of logical CPUs to use (default None: use all available)

    Returns:
    List[List[str]]: [[i, path_to_file],...]
    if i == 0, no reindex
    """
    try_ops, reference_asu = find_alt_ops(reference_file, columns)
    if not ("rs_a" in reference_asu.columns):
        ds_add_rs(reference_asu,inplace=True)
    if not ("CENTRIC" in reference_asu.columns):
        reference_asu.label_centrics(inplace=True)
    if not ("EPSILON" in reference_asu.columns):
        reference_asu.compute_multiplicity(inplace=True)        
    
    # since we can't pickle gemmi symops
    try_ops_triplets = [op.triplet() for op in try_ops] 
    additional_args=[try_ops_triplets, reference_asu, output_folder, columns, wcorr, check_isomorphous, cc_min_dif]
    # print(repeat(additional_args))
    input_args = zip(input_files, repeat(additional_args))
    if len(try_ops)>1:
        if ncpu is None:
            with Pool() as pool:
                result = pool.starmap(reindex_from_pool_map, tqdm(input_args, total=len(input_files)))
        else:
            with Pool(ncpu) as pool:
                result = pool.starmap(reindex_from_pool_map, tqdm(input_args, total=len(input_files)))

        df_record = pd.DataFrame(result)
        df_record.columns=['file_idx', 'best_symop', 'num_duplicates', 'reindexed_file', *[f"CC_symop{i}" for i in range(len(try_ops))]]
        df_record.to_pickle(os.path.join(output_folder, 'reindex_record.pkl'))
        print("Reindex statistics record has been saved at:", flush=True)
        print(os.path.join(output_folder, 'reindex_record.pkl'), flush=True)
        return df_record
    else:
        print("No reindexing required!")
        return None