import pandas as pd
import reciprocalspaceship as rs
import numpy as np
import os
from tqdm import tqdm
from functools import reduce
from multiprocessing import Pool
from itertools import repeat


def find_intersection(input_files, output_path, amplitude_col='F-obs-scaled'):
    
    """
    Finds the intersection of `amplitude_col` from multiple input MTZ files.

    Args:
        input_files (list): List of input MTZ file paths.
        output_path (str): Path to save the output pickle file containing the intersection data.
    """
        
    df_list = []
    for file in tqdm(input_files):
        try:
            df = rs.read_mtz(file)[[amplitude_col]]
            df = df.rename(columns={amplitude_col: os.path.basename(file)})
            df_list.append(df)
        except:
            continue
    result = pd.concat(df_list, axis=1, join='inner')
    print(f"The intersection has shape {result.shape}. Pickling begins now.")
    result.to_pickle(output_path)
    
def find_union(input_files, output_path, sigF_path, amplitude_col='F-obs-scaled', error_col='SIGF-obs-scaled',include_errors=True):
    
    """
    Finds the union of `amplitude_col` from multiple input MTZ files.

    Args:
        input_files (list): List of input MTZ file paths.
        output_path (str): Path to save the output pickle file containing the union data.
    """
        
    df_list = []
    for file in input_files:
        try:
            if include_errors:
                df = rs.read_mtz(file)[[amplitude_col, error_col]]
                df = df.rename(columns={amplitude_col: "F_" + os.path.basename(file), error_col: "SIGF_" + os.path.basename(file)})
            else:
                df = rs.read_mtz(file)[[amplitude_col]]
                df = df.rename(columns={amplitude_col: os.path.basename(file)})
            df_list.append(df)
        except Exception as e:
            print(e)
            print("Failed to retrieve " + file)
            continue
            
    print("Done reading in " + str(len(input_files)) + " files (disregarding errors); now starting concatenation. Please be patient.")
    result = pd.concat(df_list, axis=1, join='outer')
    if include_errors:
        result_F    = result.loc[:,[key for key in result.columns if key.startswith("F_")]]
        result_SIGF = result.loc[:,[key for key in result.columns if key.startswith("SIGF_")]]
        stripped_keys={}
        for key in result_F.columns:
            stripped_keys[key]=key[2:]
        result_F.rename(columns=stripped_keys)
        result_SIGF.rename(columns=stripped_keys)
        # print(result_F.info())
        # print(result_SIGF.info())
        result_F.to_pickle(output_path)
        result_SIGF.to_pickle(sigF_path)
    else:
        stripped_keys={}
        for key in result.columns:
            stripped_keys[key]=key[2:]
        result.rename(columns=stripped_keys)
        print(result.info())
        result.to_pickle(output_path)
    print(f"The union has shape {result.shape}. Pickling complete.")
    

# ABOUT EQUALLY FAST:
# def find_union(input_files, output_path, amplitude_col='F-obs-scaled'):
    
#     """
#     Finds the union of `amplitude_col` from multiple input MTZ files.

#     Args:
#         input_files (list): List of input MTZ file paths.
#         output_path (str): Path to save the output pickle file containing the union data.
#     """
        
#     df_list = []
#     index_list = []
#     for file in tqdm(input_files[0:20]):
#         try:
#             df = rs.read_mtz(file)[[amplitude_col]]
#             df = df.rename(columns={amplitude_col: os.path.basename(file)})
#             df_list.append(df)
#         except:
#             continue
            
#     df_merged = reduce(lambda left,right: pd.merge(left,right,left_index=True,right_index=True,
#         how='outer', sort=True), df_list)   
#     print(df_merged.info())
#     df_merged.to_pickle(output_path)

def standardize(input_, output_folder):
    
    """
    Used by `generate_vae_io`, this helper function standardizes the input data and saves the standardized data, mean, and standard deviation to the specified output folder.

    Args:
        input_ (DataFrame): The input data to be standardized.
        output_folder (str): The path to the output folder where the standardized data, mean, and standard deviation will be saved as pickle files.

    Returns:
        tuple: A tuple containing the standardized data (numpy.ndarray), mean (float), and standard deviation (float).
    """

    # mean = np.mean(input_,axis=0)
    # sd = np.std(input_,axis=0)
    mean = input_.mean(axis=0)
    sd = input_.std(axis=0,ddof=0)
    standard = (input_ - mean)/sd
    
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    standard.to_pickle(os.path.join(output_folder, 'union_standardized.pkl'))
    mean.to_pickle(os.path.join(output_folder, 'union_mean.pkl'))
    sd.to_pickle(os.path.join(output_folder, 'union_sd.pkl'))
    # except:
    #     print('saving as numpy array (npy) instead')
    #     np.save(os.path.join(output_folder, 'union_mean.pkl'), mean)
    #     np.save(os.path.join(output_folder, 'union_sd.pkl'),   sd)
    
    return standard, mean, sd

    
def generate_vae_io(intersection_path, union_path, sigF_path, io_folder, prefix=None, include_errors=False):
    
    """
    Generates VAE input and output data from the intersection and union datasets and saves them to the specified folder. Mean and SD data, calculated from union data, to re-scale are saved in io_folder. Standardized union becomes the VAE output. Intersection is standardized with the aforementioned mean and SD and becomes the VAE input.

    Args:
        intersection_path (str): The path to the intersection dataset pickle file.
        union_path (str): The path to the union dataset pickle file.
        io_folder (str): The path to the output folder where the VAE input and output will be saved.
        prefix (str, optional): prefix to assign to the output (useful when comparing different runs).
    """
        
    # Read in the intersection and union data
    intersection = pd.read_pickle(intersection_path)
    union = pd.read_pickle(union_path)
    # union = union.to_numpy()

    # Generate VAE output (targets for reconstruction)
    vae_output, vae_output_mean, vae_output_std = standardize(union.T, io_folder)
    print("Size of vae_output (training data): " + str(vae_output.shape))
    print("Size of mean across datasets: " + str(vae_output_mean.shape))
    print("Size of stdev across datasets: " + str(vae_output_std.shape))
    
    # Generate VAE input
    vae_input = intersection.T
    vae_input = (vae_input - vae_output_mean[vae_input.columns])/vae_output_std[vae_input.columns]
    print("Size of vae_input (training data): " + str(vae_input.shape))
    
    if include_errors:
        sigF = pd.read_pickle(sigF_path).T
        vae_sigF = sigF/vae_output_std
        vae_sigF = vae_sigF.values.astype(np.float32)

    print("Size of vae_sigF (like training data): " + str(vae_sigF.shape))
    
    # keep this below the if statement:
    vae_output = vae_output.values.astype(np.float32)
    vae_input = vae_input.values.astype(np.float32)

    
    # Save VAE input and output to specified folder path
    if not os.path.exists(io_folder):
        os.makedirs(io_folder)
        
    np.save(os.path.join(io_folder, prefix + "vae_input.npy"), vae_input)
    np.save(os.path.join(io_folder, prefix + "vae_output.npy"), vae_output)
    if include_errors:
        np.save(os.path.join(io_folder, prefix + "vae_sigF.npy"), vae_sigF)
    print("Created starting files for VAE in " + io_folder + " with prefix = " + prefix)
    
    
def rescale(recons_path, intersection_path, union_path, input_files, info_folder, output_folder, amplitude_col='F-obs-scaled', return_full_recon=False):
    
    """
    Re-scales datasets accordingly to recover the outputs in the original scale in column 'recons' and calculates the difference in amplitudes in column 'diff'.
    Input files should be in the same order as intersection & union rows.

    Args:
        recons_path (str): Path to the reconstructed output of the VAE in NumPy format.
        intersection_path (str): Path to the pickle file containing the intersection of all scaled datasets.
        union_path (str): Path to the pickle file containing the union data of all scaled datasets.
        
        input_files (list): List of input file paths.
        info_folder (str): Path to the folder containing files with the mean and SD used to standardize previously.
        output_folder (str): Path to the folder where the reconstructed data will be saved.
        
        amplitude_col (str): Column in MTZ file that contains structure factor amplitudes to calculate the difference column.
        return_full_recon (bool): Whether or not to also output an MTZ with all of the reconstructed amplitudes, 
        not just for the Miller indices for which the dataset of interest has observations (False by default).

    Returns:
        None

    """
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)   
    if return_full_recon:
        full_recon_output_folder=output_folder+"full_recon/"
        if not os.path.exists(full_recon_output_folder):
            os.makedirs(full_recon_output_folder)   

    recons = np.load(recons_path)
    intersection = pd.read_pickle(intersection_path)
    union = pd.read_pickle(union_path)
    # print(intersection.info()) # columns are the dataset IDs; rows the Miller indices of the intersection
    
    print(recons.shape)
    if recons.shape[0]==2: # mean and std in array
        recons_df = pd.DataFrame(np.squeeze(recons[0,:,:]).T, \
                                      index=union.index, \
                                      columns=intersection.columns)
        recons_df_std  = pd.DataFrame(np.squeeze(recons[1,:,:]).T, \
                                      index=union.index, \
                                      columns=intersection.columns)
        include_std=True
    else:
        recons_df = pd.DataFrame(recons.T, index=union.index, columns=intersection.columns)
        include_std=False

    mean = pd.read_pickle(os.path.join(info_folder, 'union_mean.pkl'))
    sd   = pd.read_pickle(os.path.join(info_folder, 'union_sd.pkl'))
        
    for file in tqdm(input_files):
        
        col = recons_df[os.path.basename(file)]

        ds = rs.read_mtz(file)
        idx = ds.index

        recons_col = col[idx] * sd[idx] + mean[idx]
        recons_col = rs.DataSeries(recons_col, dtype="SFAmplitude")

        ds['recons'] = recons_col
        ds['diff'] = ds[amplitude_col] - ds['recons']
    
        # if a second list element is present, it is the stdev across VAE sampling
        if include_std:
            std_col = recons_df_std[os.path.basename(file)]
            recons_std_col = std_col[idx] * sd[idx]
            recons_std_col = rs.DataSeries(recons_std_col, dtype="Stddev")
            ds['SIG_recons'] = recons_std_col
        
        ds.write_mtz(os.path.join(output_folder, os.path.basename(file)))

        if return_full_recon:
            base_name=os.path.basename(file)[:-4]+"_full_recon.mtz"
            recons_col_full = col * sd + mean
    
            tmp=pd.DataFrame(index=union.index)
            tmp["recons"]=recons_col_full
            tmp["recons"]=tmp["recons"].astype("SFAmplitude")
    
            if include_std:
                recons_col_full_std = std_col * sd
                tmp["SIG_recons"]=recons_col_full_std
                tmp["SIG_recons"]=tmp["SIG_recons"].astype("Stddev")
            
            full_ds = rs.DataSet(tmp, spacegroup=ds.spacegroup, cell=ds.cell, merged=ds.merged)
            full_ds.infer_mtz_dtypes()
            full_ds.write_mtz(os.path.join(full_recon_output_folder, base_name))
            

# $$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$ #

def rescale_from_pool_map(input_file, additional_args):
    recons_df_list= additional_args[0]
    mean          = additional_args[1]
    sd            = additional_args[2]
    amplitude_col = additional_args[3]
    output_folder = additional_args[4]
    full_recon    = additional_args[5]
    full_recon_output_folder = additional_args[6]
    union_index   = additional_args[7]

    recons_df = recons_df_list[0] 
    col = recons_df[os.path.basename(input_file)]
        
    ds = rs.read_mtz(input_file)
    idx = ds.index

    recons_col = col[idx] * sd[idx] + mean[idx]
    recons_col = rs.DataSeries(recons_col, dtype="SFAmplitude")
    
    ds['recons'] = recons_col
    ds['diff']   = ds[amplitude_col] - ds['recons']

    # if a second list element is present, it is the stdev across VAE sampling
    if len(recons_df_list)>1:
        include_std=True
        recons_df_std=recons_df_list[1] 
        std_col = recons_df_std[os.path.basename(input_file)]
        recons_std_col = std_col[idx] * sd[idx]
        recons_std_col = rs.DataSeries(recons_std_col, dtype="Stddev")
        ds['SIG_recons'] = recons_std_col
    
    ds.write_mtz(os.path.join(output_folder, os.path.basename(input_file)))

    if full_recon:
        base_name=os.path.basename(input_file)[:-4]+"_full_recon.mtz"
        recons_col_full = col * sd + mean
        
        tmp=pd.DataFrame(index=union_index)
        tmp["recons"]=recons_col_full
        tmp["recons"]=tmp["recons"].astype("SFAmplitude")

        if include_std:
            recons_col_full_std = std_col * sd
            tmp["SIG_recons"]=recons_col_full_std
            tmp["SIG_recons"]=tmp["SIG_recons"].astype("Stddev")
        
        full_ds = rs.DataSet(tmp, spacegroup=ds.spacegroup, cell=ds.cell, merged=ds.merged)
        full_ds.infer_mtz_dtypes()
        full_ds.write_mtz(os.path.join(full_recon_output_folder, base_name))

    return 1

def rescale_pool(recons_path, intersection_path, union_path, input_files, info_folder, output_folder, amplitude_col='F-obs-scaled', return_full_recon=False, ncpu=None):
    
    """
    Re-scales datasets accordingly to recover the outputs in the original scale in column 'recons' and calculates the difference in amplitudes in column 'diff'.
    Input files should be in the same order as intersection & union rows.

    Args:
        recons_path (str): Path to the reconstructed output of the VAE in NumPy format.
        intersection_path (str): Path to the pickle file containing the intersection of all scaled datasets.
        union_path (str): Path to the pickle file containing the union data of all scaled datasets.
        
        input_files (list): List of input file paths.
        info_folder (str): Path to the folder containing files with the mean and SD used to standardize previously.
        output_folder (str): Path to the folder where the reconstructed data will be saved.
        
        amplitude_col (str): Column in MTZ file that contains structure factor amplitudes to calculate the difference column.
        return_full_recon (bool): Whether or not to also output an MTZ with all of the reconstructed amplitudes, 
        not just for the Miller indices for which the dataset of interest has observations (False by default).

        ncpu (int): number of CPU to allocate for multiprocessing (default: None: whatever is available).
        
    Returns:
        None

    """
    
    full_recon_output_folder=output_folder+"full_recon/" # only used if return_full_recon == True
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)   
    if return_full_recon:
        if not os.path.exists(full_recon_output_folder):
            os.makedirs(full_recon_output_folder)   

    recons = np.load(recons_path)
    intersection = pd.read_pickle(intersection_path)
    union = pd.read_pickle(union_path)

    if recons.shape[0]==2: # mean and std in array
        recons_df_mean = pd.DataFrame(np.squeeze(recons[0,:,:]).T, \
                                      index=union.index, \
                                      columns=intersection.columns)
        recons_df_std  = pd.DataFrame(np.squeeze(recons[1,:,:]).T, \
                                      index=union.index, \
                                      columns=intersection.columns)
        recons_df_list = [recons_df_mean, recons_df_std]
    else:
        recons_df = pd.DataFrame(recons.T, index=union.index, columns=intersection.columns)
        recons_df_list = [recons_df]
    
    mean = pd.read_pickle(os.path.join(info_folder, 'union_mean.pkl'))
    sd = pd.read_pickle(os.path.join(info_folder, 'union_sd.pkl'))

    additional_args=[recons_df_list, mean, sd, amplitude_col, output_folder, \
                     return_full_recon, full_recon_output_folder, union.index]
    input_args = zip(input_files, repeat(additional_args))
    if ncpu is not None:
        with Pool(ncpu) as pool:
            results = pool.starmap(rescale_from_pool_map, tqdm(input_args, total=len(input_files)))
    else:
        with Pool() as pool:
            results = pool.starmap(rescale_from_pool_map, tqdm(input_args, total=len(input_files)))

    print("Done rescaling.")