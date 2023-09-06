import os
from multiprocessing import Pool
from itertools import repeat
from tqdm import tqdm
import torch
import pandas as pd
import reciprocalspaceship as rs

def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f"cuda:{i}")
    return torch.device("cpu")


def add_phases(file_list, apo_mtzs_path, vae_reconstructed_with_phases_path, phase_2FOFC_col_out='PH2FOFCWT', phase_FOFC_col_out='PHFOFCWT',phase_2FOFC_col_in='PH2FOFCWT', phase_FOFC_col_in='PHFOFCWT'):
    """
    Add phases from apo models refined against the data (or otherwise) to the corresponding files in file_list and 
    write the resulting MTZ to vae_reconstructed_with_phases_path. Filenames in the file_list and the "apo" MTZs should match (e.g., ####.mtz)

        Parameters:
            file_list (list of str) : list of input files (complete path!)
            apo_mtzs_path (str) : path to reference datasets refined as apo
            vae_reconstructed_with_phases_path (str) : path for the output MTZs with reconstructed amplitudes and added phases
            phase_2FOFC_col_out (str) : output MTZ column name for 2Fo-Fc phases
            phase_FOFC_col_out (str)  : output MTZ column name for  Fo-Fc phases
            phase_2FOFC_col_in (str) : *input* MTZ column name for 2Fo-Fc phases
            phase_FOFC_col_in (str)  : *input* MTZ column name for  Fo-Fc phases

        Returns:
            list of input files for which no matching file with phases could be found.
    """
    
    no_phases_files = []
    # Phases here are copied from refinement 
    for file in tqdm(file_list):
        current = rs.read_mtz(file)
        try:
            phases_df = rs.read_mtz(apo_mtzs_path + os.path.basename(file))    
        except:
            no_phases_files.append(file)
            continue
        
        current[phase_2FOFC_col_out] = phases_df[phase_2FOFC_col_in]
        current[phase_FOFC_col_out]  = phases_df[phase_FOFC_col_in]
        current.write_mtz(vae_reconstructed_with_phases_path + os.path.basename(file))

    return no_phases_files


def add_phases_from_pool_map(file, additional_args):
    """ 
    Adds phases to input files. Called by valdo.helper.add_phases_pool.
    
        Parameters:
            file (str) : input MTZ to which phases will be added
            additional_args (list) : additional parameters passed on from valdo.helper.add_phases_pool

        Returns:
            list with input file name (str) and boolean indicating whether phases were added successfully.
    """
    apo_mtzs_path                      = additional_args[0]
    vae_reconstructed_with_phases_path = additional_args[1]
    phase_2FOFC_col_out                = additional_args[2]
    phase_FOFC_col_out                 = additional_args[3]
    phase_2FOFC_col_in                 = additional_args[4]
    phase_FOFC_col_in                  = additional_args[5]
    
    current = rs.read_mtz(file)
    success=False
    try:
        phases_df = rs.read_mtz(apo_mtzs_path + os.path.basename(file)) 
        # print(phases_df.columns)
        current[phase_2FOFC_col_out] = phases_df[phase_2FOFC_col_in]
        current[phase_FOFC_col_out]  = phases_df[phase_FOFC_col_in]
        current.write_mtz(vae_reconstructed_with_phases_path + os.path.basename(file))
        success=True
    except Exception as e:
        # print(e,flush=True)
        pass
        
    return [file, success]


def add_phases_pool(file_list, apo_mtzs_path, vae_reconstructed_with_phases_path, phase_2FOFC_col_out='PH2FOFCWT', phase_FOFC_col_out='PHFOFCWT',phase_2FOFC_col_in='PH2FOFCWT', phase_FOFC_col_in='PHFOFCWT',prefix=None, ncpu=None):
    """
    Add phases from apo models refined against the data (or otherwise) to the corresponding files in file_list and 
    write the resulting MTZ to vae_reconstructed_with_phases_path. Filenames in the file_list and the "apo" MTZs should match (e.g., ####.mtz)
    (multiprocessing variant of valdo.helper.add_phases)

        Parameters:
            file_list (list of str) : list of input files (complete path!)
            apo_mtzs_path (str) : path to reference datasets refined as apo
            vae_reconstructed_with_phases_path (str) : path for the output MTZs with reconstructed amplitudes and added phases
            phase_2FOFC_col_out (str) : output MTZ column name for 2Fo-Fc phases
            phase_FOFC_col_out (str)  : output MTZ column name for  Fo-Fc phases
            phase_2FOFC_col_in (str) : *input* MTZ column name for 2Fo-Fc phases
            phase_FOFC_col_in (str)  : *input* MTZ column name for  Fo-Fc phases
            prefix (str) : prefix to add to pickle file report.
            ncpu (int) : Number of CPUs available

        Returns:
            a dataframe reporting for each dataset whether phases were successfully added.
    """

    additional_args=[apo_mtzs_path, vae_reconstructed_with_phases_path, phase_2FOFC_col_out, phase_FOFC_col_out,phase_2FOFC_col_in, phase_FOFC_col_in]
    
    with Pool(ncpu) as pool:
        metrics = pool.starmap(add_phases_from_pool_map, zip(file_list, repeat(additional_args)))
            
        metrics_df = pd.DataFrame(metrics)
        metrics_df.columns=['file', 'success']
        metrics_df.to_pickle(vae_reconstructed_with_phases_path + prefix + 'add_phases_report.pkl')
    
    return metrics_df

    
        
