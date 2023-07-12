import pandas as pd
import reciprocalspaceship as rs
import numpy as np
import os
from tqdm import tqdm

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
    result.to_pickle(output_path)
    
def find_union(input_files, output_path, amplitude_col='F-obs-scaled'):
    
    """
    Finds the union of `amplitude_col` from multiple input MTZ files.

    Args:
        input_files (list): List of input MTZ file paths.
        output_path (str): Path to save the output pickle file containing the union data.
    """
        
    df_list = []
    for file in tqdm(input_files):
        try:
            df = rs.read_mtz(file)[[amplitude_col]]
            df = df.rename(columns={amplitude_col: os.path.basename(file)})
            df_list.append(df)
        except:
            continue
    result = pd.concat(df_list, axis=1, join='outer')
    result.to_pickle(output_path)
    
def standardize(input_, output_folder):
    
    """
    Used by `generate_vae_io`, this helper function standardizes the input data and saves the standardized data, mean, and standard deviation to the specified output folder.

    Args:
        input_ (numpy.ndarray): The input data to be standardized.
        output_folder (str): The path to the output folder where the standardized data, mean, and standard deviation will be saved as pickle files.

    Returns:
        tuple: A tuple containing the standardized data (numpy.ndarray), mean (float), and standard deviation (float).
    """

    mean = np.mean(input_)
    sd = np.std(input_)
    standard = (input_ - mean)/sd
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    standard.to_pickle(os.path.join(output_folder, 'union_standardized.pkl'))
    mean.to_pickle(os.path.join(output_folder, 'union_mean.pkl'))
    sd.to_pickle(os.path.join(output_folder, 'union_sd.pkl'))
    
    return standard, mean, sd

    
def generate_vae_io(intersection_path, union_path, io_folder):
    
    """
    Generates VAE input and output data from the intersection and union datasets and saves them to the specified folder. Mean and SD data, calculated from union data, to re-scale are saved in io_folder. Standardized union becomes the VAE output. Intersection is standardized with the aforementioned mean and SD and becomes the VAE input.

    Args:
        intersection_path (str): The path to the intersection dataset pickle file.
        union_path (str): The path to the union dataset pickle file.
        io_folder (str): The path to the output folder where the VAE input and output will be saved.
    """
        
    # Read in the intersection and union data
    intersection = pd.read_pickle(intersection_path)
    union = pd.read_pickle(union_path)

    # Generate VAE output
    vae_output, vae_output_mean, vae_output_std = standardize(union.T, io_folder)
    vae_output = vae_output.values.astype(np.float32)
    
    # Generate VAE input
    vae_input = intersection.T
    vae_input = (vae_input - vae_output_mean[vae_input.columns])/vae_output_std[vae_input.columns]
    vae_input = vae_input.values.astype(np.float32)

    # Save VAE input and output to specified folder path
    if not os.path.exists(io_folder):
        os.makedirs(io_folder)
        
    np.save(os.path.join(io_folder, "vae_input.npy"), vae_input)
    np.save(os.path.join(io_folder, "vae_output.npy"), vae_output)
    
def rescale(recons_path, intersection_path, union_path, input_files, info_folder, output_folder, amplitude_col='F-obs-scaled'):
    
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

    Returns:
        None

    """
    
    recons = np.load(recons_path)
    intersection = pd.read_pickle(intersection_path)
    union = pd.read_pickle(union_path)
    
    recons_df = pd.DataFrame(recons.T, index=union.index, columns=intersection.columns)
    mean = pd.read_pickle(os.path.join(info_folder, 'union_mean.pkl'))
    sd = pd.read_pickle(os.path.join(info_folder, 'union_sd.pkl'))
    
    for file in tqdm(input_files):
        
        col = recons_df[os.path.basename(file)]

        ds = rs.read_mtz(file)
        idx = ds.index

        recons_col = col[idx] * sd[idx] + mean[idx]
        recons_col = rs.DataSeries(recons_col, dtype="SFAmplitude")

        ds['recons'] = recons_col

        ds['diff'] = ds[amplitude_col] - ds['recons']
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)   

        ds.write_mtz(os.path.join(output_folder, os.path.basename(file)))

    