import numpy as np
import reciprocalspaceship as rs
from tqdm import tqdm
from multiprocessing import Pool
from itertools import repeat
import pandas as pd
import re
import glob
from scipy.ndimage import gaussian_filter
import os
import gemmi


def preprocess(matrix, radius_in_A=5):
    
    """
    Preprocesses the input matrix by applying Gaussian filtering.

    Args:
        matrix (numpy.ndarray): Input matrix to be preprocessed.
        radius_in_A (int, optional): Radius in Angstroms for Gaussian filtering. Default is 5.

    Returns:
        numpy.ndarray: Preprocessed matrix.

    """
    grid_spacing = np.min(matrix.spacing)

    matrix = np.absolute(matrix)
    radius_in_voxels = int(radius_in_A / grid_spacing)
    sigma = int(radius_in_voxels / 3)

    try:
        result=gaussian_filter(matrix, sigma=sigma, radius=radius_in_voxels)
    except:
        # earlier versions of scipy don't support 'radius'
        result=gaussian_filter(matrix, sigma=sigma, truncate=3.0)

    return result


def generate_blobs(input_files, model_folder, diff_col, phase_col, output_folder, prefix=None, cutoff=5, radius_in_A=5, negate=False, sample_rate=3):
    
    """
    Generates blobs from electron density maps that have been passed through a pre-processing function using the specified parameters and saves the blob statistics to a DataFrame.
    The pre-processing function in this case takes the absolute value of the difference map and applies a Gaussian blur with radius 5 Angstroms.

    The function identifies blobs above a certain contour level and volume threshold using gemmi's find_blobs_by_floodfill method. The blobs are characterized by metrics such as volume (proportional to the number of voxels in the region), score (sum of values at every voxel in the region), peak value (highest sigma value in the region), and more.

    Args:
        input_files (list): List of input file paths.
        model_folder (str): Path to the folder containing the refined models for each dataset (pdb format).
        diff_col (str): Name of the column representing diffraction values.
        phase_col (str): Name of the column representing phase values.
        output_folder (str): Path to the output folder where the blob statistics DataFrame will be saved.
        prefix (str, optional) : Prefix for the output pickle--useful for keeping track of tests.
        cutoff (int, optional): Blob cutoff value. Blobs with values below this cutoff will be ignored. Default is 5.
        negate (bool, optional): Whether to negate the blob statistics. Default is False. Use True if there is interest in both positive and negative peaks.
        sample_rate (int, optional): Sample rate for generating the grid in the FFT process. Default is 3.

    Returns:
        None

    Example:
        input_files = ['./data/file1.mtz', './data/file2.mtz']
        model_folder = './models'
        diff_col = 'diff'
        phase_col = 'refine_PH2FOFCWT'
        output_folder = './output'
        
        generate_blobs(input_files, model_folder, diff_col, phase_col, output_folder)
    """
        
    peaks = []
    blob_stats = []
    
    # print(input_files)
    for file in tqdm(input_files):
        blob_stats_per_file = blob_helper(file, model_folder, diff_col, phase_col, output_folder, cutoff, radius_in_A, negate, sample_rate)            
        blob_stats.append(blob_stats_per_file)

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)   

    blob_stats_df = pd.concat(blob_stats)
    blob_stats_df.to_pickle(os.path.join(output_folder, prefix + 'blob_stats.pkl'))
    print("Done generating blobs and wrote " + os.path.join(output_folder, prefix + 'blob_stats.pkl'))


def blob_helper(file, model_folder, diff_col, phase_col, output_folder, cutoff=4, radius_in_A=5, negate=False, sample_rate=3):
    sample = rs.read_mtz(file)
    
    sample = rs.read_mtz(file)[[diff_col, phase_col]].dropna()
    sample_id = os.path.splitext(os.path.basename(file))[0]
    # print(sample_id)
    # print("1: " + glob.glob(os.path.join(model_folder, f"*{sample_id}*.pdb"))[0])
    # print("2: " + os.path.join(model_folder, f"{sample_id}.pdb"))
    error_file = os.path.join(output_folder, 'error_log.txt')  # Path to the error log file
    blob_stats=[]
    
    try:
        try:
            structure = gemmi.read_pdb(os.path.join(model_folder, f"{sample_id}.pdb"))
        except: 
            structure = gemmi.read_pdb(glob.glob(os.path.join(model_folder, f"*{sample_id}*.pdb"))[0])
    except Exception as e:        
        error_message = f'Could not identify the model file for sample {sample_id}: {str(e)}.\n'
        print(error_message)
        with open(error_file, 'a') as f:
            f.write(error_message)
        return blob_stats
        
    sample_gemmi=sample.to_gemmi()
    grid = sample_gemmi.transform_f_phi_to_map(diff_col, phase_col, sample_rate=sample_rate)
    grid.normalize()
    
    blurred_grid = preprocess(grid, radius_in_A)
    grid.set_subarray(blurred_grid, [0, 0, 0])
    grid.normalize()
    
    mean, sigma = np.mean(np.array(grid)), np.std(np.array(grid))
    
    blobs = gemmi.find_blobs_by_flood_fill(grid, cutoff=cutoff, negate=negate)

    use_long_names = False
    sort_by_key='peakz'

    ns = gemmi.NeighborSearch(structure[0], structure.cell, 5).populate()
    count = 0
    # print(str(len(blobs)))
    for blob in blobs:

        blob_stat = {
            "sample"  :    sample_id,
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
        
    blob_stats_df = pd.DataFrame(blob_stats)
    
    return blob_stats_df

def blob_helper_wrapper(input_file, additional_args):
    model_folder = additional_args[0]
    diff_col     = additional_args[1]
    phase_col    = additional_args[2]
    output_folder= additional_args[3]
    cutoff       = additional_args[4]
    radius_in_A  = additional_args[5]
    negate       = additional_args[6]
    sample_rate  = additional_args[7]
    
    blob_stats_df=  blob_helper(input_file, model_folder, diff_col, phase_col, output_folder, cutoff=cutoff, radius_in_A=radius_in_A, negate=negate, sample_rate=sample_rate)
    return blob_stats_df
    
def generate_blobs_pool(input_files, model_folder, diff_col, phase_col, output_folder, prefix=None, cutoff=4, radius_in_A=5, negate=False, sample_rate=3,ncpu=None):
    """
    See generate_blobs()
    """
        
    if not os.path.exists(output_folder):
        print(output_folder + " should exist already!")

    additional_args=[model_folder, diff_col, phase_col, output_folder, cutoff, radius_in_A, negate, sample_rate]
    input_args = zip(input_files, repeat(additional_args))
    with Pool(ncpu) as pool:
        blob_stats = pool.starmap(blob_helper_wrapper, tqdm(input_args, total=len(input_files)))

    blob_stats_df = pd.concat(blob_stats,ignore_index=True)
    blob_stats_df.to_pickle(os.path.join(output_folder, prefix + 'blob_stats.pkl'))
    print("Done generating blobs with starmap/pool and wrote " + os.path.join(output_folder, prefix + 'blob_stats.pkl'))

    return blob_stats_df