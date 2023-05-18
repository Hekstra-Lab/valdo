import pandas as pd
import reciprocalspaceship as rs
import numpy as np
import os
from tqdm import tqdm

def find_intersection(input_files, output_path):
    df_list = []
    for file in tqdm(input_files):
        try:
            df = rs.read_mtz(file)[['F-obs-scaled']]
            df = df.rename(columns={'F-obs-scaled': os.path.basename(file)})
            df_list.append(df)
        except:
            continue
    result = pd.concat(df_list, axis=1, join='inner')
    result.to_pickle(output_path)
    
def find_union(input_files, output_path):
    df_list = []
    for file in tqdm(input_files):
        try:
            df = rs.read_mtz(file)[['F-obs-scaled']]
            df = df.rename(columns={'F-obs-scaled': os.path.basename(file)})
            df_list.append(df)
        except:
            continue
    result = pd.concat(df_list, axis=1, join='outer')
    result.to_pickle(output_path)
    
def standardize(input_, output_folder):

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
    
    return vae_input, vae_output

def reconstruct(recons_path, intersection_path, union_path, input_files, info_folder, output_folder):
    
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

        ds['F-obs-recons'] = recons_col

        ds['F-obs-diff'] = ds['F-obs-scaled'] - ds['F-obs-recons']
        
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)   

        ds.write_mtz(os.path.join(output_folder, os.path.basename(file)))

    