import numpy as np
import reciprocalspaceship as rs
from tqdm import tqdm
import pandas as pd
import re
import glob
from scipy.ndimage import gaussian_filter
import os
import gemmi


def generate_blobs(input_files, model_folder, diff_col, phase_col, output_folder, cutoff=5, negate=False, sample_rate=3):
    
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

        return gaussian_filter(matrix, sigma=sigma, radius=radius_in_voxels)
    
    peaks = []
    blob_stats = []
    
    error_file = os.path.join(output_folder, 'error_log.txt')  # Path to the error log file

    for file in tqdm(input_files):

        sample = rs.read_mtz(file)[[diff_col, phase_col]].dropna()

        sample_id = os.path.splitext(os.path.basename(file))[0]
        
        try:
            structure = gemmi.read_pdb(f'{model_folder}/{sample_id}.pdb')
            
        except Exception as e:
            
            error_message = f'Could not identify the model file for sample {sample_id}: {str(e)}.\n'
            
            with open(error_file, 'a') as f:
                f.write(error_message)
            continue

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

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)   

    blob_stats_df = pd.DataFrame(blob_stats)
    blob_stats_df.to_pickle(os.path.join(output_folder, 'blob_stats.pkl'))
