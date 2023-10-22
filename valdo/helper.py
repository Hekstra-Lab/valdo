import os
import sys
import glob
import re
import shutil
from multiprocessing import Pool
from itertools import repeat
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
import reciprocalspaceship as rs

def try_gpu(i=0):
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f"cuda:{i}")
    return torch.device("cpu")


def configure_session(nfree=1):
    """
    nfree : number of free cpus you want to keep in multprocessing, default 1
        Increase the number if you got cpu memory issue
    """
    bGPU=torch.cuda.is_available() 
    if bGPU:
        print("We will use GPU for torch operations (esp. VAE training).")
    
    ncpu=int(os.environ.get('SLURM_CPUS_PER_TASK', os.cpu_count()))
    print("There are " + str(ncpu) + " CPUs available.")
    if ncpu > nfree:
        ncpu=ncpu-nfree    
        print("For multiprocessing, we will use " + str(ncpu) + " CPUs.")
    else:
        ncpu=1
        print("We will not use multiprocessing.")
    return bGPU, ncpu


def report_mem_usage(top_n=5):
    """
    report_mem_usage(top_n) allows the user to see the largest top_n (int) local variables taking up memory.
    """
    def sizeof_fmt(num, suffix='B'):
        ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
        for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
            if abs(num) < 1024.0:
                return "%3.1f %s%s" % (num, unit, suffix)
            num /= 1024.0
        return "%.1f %s%s" % (num, 'Yi', suffix)

    print("Memory use of top " + str(top_n) + " local variables.")
    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in list(
                              locals().items())), key= lambda x: -x[1])[:top_n]:
        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
    
    return 0


def standardize_single_mtzs(filename, additional_args):
    """
    Helper function for standardize_input_mtzs().
    """
    source_path     =additional_args[0]
    destination_path=additional_args[1]
    pattern         =additional_args[2]
    
    # Check if the file matches the pattern
    match = re.match(pattern, filename)
    if match:
        # Extract the ID from the filename
        id = match.group(1)
        
        # Define the new filename
        new_filename = id + ".mtz"
        
        # Construct the full source and destination paths
        tmp_source_path = os.path.join(source_path, filename)
        tmp_destination_path = os.path.join(destination_path, new_filename)
        
        # Copy the file to the destination folder with the new name
        # print(source_path)
        # print(destination_path)
        try:
            shutil.copy(tmp_source_path, tmp_destination_path)
            return new_filename
        except Exception as e:
            print(e)
            return None
    else:
        print("No match for " + filename)
        return None

def standardize_input_mtzs(source_path, destination_path, mtz_file_pattern, ncpu=1):
    """
    Prepare the raw observed (inut) MTZ files by copying them to the pipeline folder and standardizing their names.

    Parameters:
        source_path (str): where to find the input MTZ files.
        destination_path (str): where to put the standardized filenames.
        mtz_file_pattern (str): regular expression for the input MTZ file names.

    Returns:
        list of standardized file names for successfully copied files.
    """

    # Get a list of all files in the source folder
    file_list = glob.glob(source_path + "*.mtz")
    print("Copying & renaming " + str(len(file_list)) + " MTZ files from " + source_path + " to " + destination_path)
    

    additional_args=[source_path, destination_path, mtz_file_pattern]
    if ncpu>1:
        with Pool(ncpu) as pool:
            result = pool.starmap(standardize_single_mtzs, zip(file_list,repeat(additional_args)))
        
    else:
        result=[]
        for filename in tqdm(file_list):
            result.append(standardize_single_mtzs(filename, additional_args))
    result = [i for i in result if i is not None]
    return result
 

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
            print("Reading in " +   glob.glob(os.path.join(apo_mtzs_path, f"*{os.path.splitext(os.path.basename(file))[0]}*.mtz"))[0])
            phases_df = rs.read_mtz(glob.glob(os.path.join(apo_mtzs_path, f"*{os.path.splitext(os.path.basename(file))[0]}*.mtz"))[0])   
        except:
            no_phases_files.append(file)
            continue
        
        current[phase_2FOFC_col_out] = phases_df[phase_2FOFC_col_in]
        current[phase_FOFC_col_out]  = phases_df[phase_FOFC_col_in]
        current.write_mtz(vae_reconstructed_with_phases_path + os.path.basename(file))

    return no_phases_files


def add_weights_single_file(file, additional_args):
    sigF_col =additional_args[0]
    sigF_col_recons=additional_args[1]
    diff_col =additional_args[2]
    sigdF_pct=additional_args[3]
    absdF_pct=additional_args[4]
    redo     =additional_args[5]

    success=0
    try:
        current = rs.read_mtz(file)
    except Exception as e:
        print(e)
    if "WT" in current and not redo:
        print("WT already present")
        already+=1
    else:
        # print("Calculating weights for " + file)
        sigdF=current[sigF_col].to_numpy()
        # include reconstruction errors iff we have them.
        if sigF_col_recons in current:
            sigdF=np.sqrt(sigdF**2 + current[sigF_col_recons].to_numpy()**2)
        absdF=np.abs(current[diff_col].to_numpy())
        
        w = 1+(sigdF/np.percentile(sigdF,sigdF_pct))**2+(absdF/np.percentile(absdF,absdF_pct))**2
        w=1/w
        current["WT"] = w
        current["WT"]=current["WT"].astype("W")
        current["WDF"]=current["WT"]*current[diff_col]
        current["WDF"]=current["WDF"].astype("F")
        current.write_mtz(file)
        success=1

    return success

def add_weights(file_list, sigF_col="SIGF-obs", sigF_col_recons="SIG_recons", diff_col="diff",sigdF_pct=90.0, absdF_pct=99.99, redo=True, ncpu=1):
    """
    Add difference map coefficient weights to the corresponding files in file_list in vae_reconstructed_with_phases_path. 
        Parameters:
        -----------
            file_list (list of str) : list of input files (complete path!)
            sigF_col (str) : name of column containing error estimates for measured amplitudes
            sigF_col_recons (str) : name of column containing error estimates for reconstructed ampls (default: "SIG_recons, as produced by the VAE reconstruction method)
            diff_col (str): name of column for output Fobs-Frecon (default: "diff")
            sigdF_pct (float): value of sig(deltaF) at which weights substantially diminish
            absdF_pct (float): value of abs(deltaF) at which weights substantially diminish
            redo (bool): whether to override existing weights (default: True)
            ncpu (int): Number of CPUs to use for multiprocessing (default: 1)
            
        Returns:
        --------
            list of input files for which no matching file with phases could be found.
    """
    
    additional_args=[sigF_col, sigF_col_recons, diff_col, sigdF_pct, absdF_pct, redo]
    if ncpu>1:
        input_args = zip(file_list,repeat(additional_args))
        with Pool(ncpu) as pool:
            result = pool.starmap(add_weights_single_file, tqdm(input_args, total=len(file_list)))
        
    else:
        result=[]
        for filename in tqdm(file_list):
            result.append(add_weights_single_file(filename, additional_args))

    return result


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
        try:
            phases_df = rs.read_mtz(os.path.join(apo_mtzs_path, os.path.basename(file)))
        except:
            phases_df = rs.read_mtz(glob.glob(os.path.join(apo_mtzs_path, f"*{os.path.splitext(os.path.basename(file))[0]}*.mtz"))[0])

        current[phase_2FOFC_col_out] = phases_df[phase_2FOFC_col_in]
        current[phase_FOFC_col_out]  = phases_df[phase_FOFC_col_in]
        current.write_mtz(vae_reconstructed_with_phases_path + os.path.basename(file))
        success=True
    except Exception as e:
        print(e,flush=True)
        # pass
        
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
    input_args = zip(file_list, repeat(additional_args))
    with Pool(ncpu) as pool:
        metrics = pool.starmap(add_phases_from_pool_map, tqdm(input_args, total=len(file_list)))
            
        metrics_df = pd.DataFrame(metrics)
        metrics_df.columns=['file', 'success']
    
    return metrics_df[~metrics_df['success']]['file'].tolist()

    
        
