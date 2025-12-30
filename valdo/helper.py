import os
import sys
import glob
import re
import shutil
import warnings
from multiprocessing import Pool, cpu_count
from itertools import repeat
from tqdm import tqdm
import torch
import pandas as pd
import numpy as np
import reciprocalspaceship as rs
import gemmi

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
    expcolumns      =additional_args[3]
    
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
        
        # We drop nan values here to avoid issues with downstream processing
        try:
            mtz_original = rs.read_mtz(tmp_source_path)
            mtz_original = mtz_original.dropna(axis=0, subset=expcolumns)
            mtz_original.write_mtz(tmp_destination_path)
            return new_filename
        except Exception as e:
            print(e)
            return None
    else:
        print("No match for " + filename)
        return None

def standardize_input_mtzs(source_path, destination_path, mtz_file_pattern, expcolumns, ncpu=1):
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

    additional_args=[source_path, destination_path, mtz_file_pattern, expcolumns]
    if ncpu>1:
        with Pool(ncpu) as pool:
            result = pool.starmap(standardize_single_mtzs, zip(file_list,repeat(additional_args)))
        
    else:
        result=[]
        for filename in tqdm(file_list):
            result.append(standardize_single_mtzs(filename, additional_args))
    # print(result)
    result = [i for i in result if i is not None]
    return result
 

def add_phases(file_list, apo_mtzs_path, vae_reconstructed_with_phases_path, 
               phase_2FOFC_col_out='PH2FOFCWT', phase_FOFC_col_out='PHFOFCWT',phase_2FOFC_col_in='PH2FOFCWT', phase_FOFC_col_in='PHFOFCWT',
               parser=None, rfree_label_in=None):
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
            parser (function name) : parser that maps input MTZs to names of MTZ files containing apo phases (default: None)
                                     see pipeline notebook for examples.
            rfree_label_in (None or str)  : *input* MTZ column name for Rfree flags

        Returns:
            list of input files for which no matching file with phases could be found.
    """
    
    no_phases_files = []
    # Phases here are copied from refinement 
    for file in tqdm(file_list):
        current = rs.read_mtz(file)

        mtz, handler = find_phase_file(file, apo_mtzs_path, parser)
        if handler > 0:
            phases_df = rs.read_mtz(mtz)
            
            # add Rfree from phase mtz to reconstructed mtz, if not existing, generate rfree flags
            try:
                current = rs.utils.copy_rfree(current, phases_df, rfree_key=rfree_label_in)
            except:
                current = rs.utils.add_rfree(current)
            
            try:    
                current[phase_2FOFC_col_out] = phases_df[phase_2FOFC_col_in]
                current[phase_FOFC_col_out]  = phases_df[phase_FOFC_col_in]
                current.write_mtz(vae_reconstructed_with_phases_path + os.path.basename(file))
                success=True
            except Exception as e:
                print(e,flush=True)
        else:
            print(f"Handler: {handler} for input MTZ {mtz}")
            no_phases_files.append(file)
            continue
        
    return no_phases_files


def add_weights_single_file(file, additional_args):
    sigF_col =additional_args[0]
    sigF_col_recons=additional_args[1]
    diff_col =additional_args[2]
    sigdF_pct=additional_args[3]
    absdF_pct=additional_args[4]
    redo     =additional_args[5]
    cutoff   =additional_args[6]
    wt_coefs =additional_args[7]

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
        
        w = 1+wt_coefs[0]*(sigdF/np.percentile(sigdF,sigdF_pct))**2 + wt_coefs[1]*(absdF/np.percentile(absdF,absdF_pct))**2
        w=1/w
        current["WT"] = w
        current.compute_dHKL(inplace=True)
        current.loc[current.dHKL>cutoff, "WT"]=0
        current["WT"]=current["WT"].astype("W")
        current["WDF"]=current["WT"]*current[diff_col]
        current["WDF"]=current["WDF"].astype("F")
        current.write_mtz(file)
        success=1

    return success

def add_weights(file_list, sigF_col="SIGF-obs-scaled", sigF_col_recons="SIG_recons", diff_col="diff",sigdF_pct=90.0, absdF_pct=99.99, wt_coefs=(1,1), redo=True, low_res_cutoff=999., ncpu=1):
    """
    Add difference map coefficient weights to the corresponding files in file_list in vae_reconstructed_with_phases_path. 
        Parameters:
        -----------
            file_list (list of str) : list of input files (complete path!)
            sigF_col (str) : name of column containing error estimates for measured amplitudes *scaled to the reference mtz*
            sigF_col_recons (str) : name of column containing error estimates for reconstructed ampls (default: "SIG_recons, as produced by the VAE reconstruction method)
            diff_col (str): name of column for output Fobs-Frecon (default: "diff")
            sigdF_pct (float): value of sig(deltaF) at which weights substantially diminish
            absdF_pct (float): value of abs(deltaF) at which weights substantially diminish
            wt_coefs (2-tuple): additional weight coefficients for the sig_dF and abs_dF terms in the weight denominator, respectively (default: (1,1))
            redo (bool): whether to override existing weights (default: True)
            low_res_cutoff (float) : at resolution lower (larger #) than this cutoff we set the weight to 0 (default: 999.0)
            ncpu (int): Number of CPUs to use for multiprocessing (default: 1)
            
        Returns:
        --------
            list of input files for which no matching file with phases could be found.
    """
    
    additional_args=[sigF_col, sigF_col_recons, diff_col, sigdF_pct, absdF_pct, redo, low_res_cutoff, wt_coefs]
    if ncpu>1:
        input_args = zip(file_list,repeat(additional_args))
        with Pool(ncpu) as pool:
            result = pool.starmap(add_weights_single_file, tqdm(input_args, total=len(file_list)))
        
    else:
        result=[]
        for filename in tqdm(file_list):
            result.append(add_weights_single_file(filename, additional_args))

    return result

def find_phase_file(file, apo_mtzs_path, parser=None):
    '''
    Obtain the filename of the MTZ containing phases starting from the filename of the MTZ containing reconstructed amplitudes
    
    Parameters
    ----------
    file (str or list of str): Name(s) of the MTZ with reconstructed amplitudes. If the parser returns multiple possibilities, 
                               the first match to an MTZ with phases will be used. 
    apo_mtz_path (str): path to the directory with apo refinement results
    parser (function): user-provided function that will parse the input MTZ filename (default: None)
    
    Returns:
    --------
    mtz (string or list of strings): converted MTZ name
    handler (int): index for which parsing routine was used; a value < 1 indicates failure.
    '''
    handler=0
    if parser != None:
        mtz_names=parser(os.path.basename(file)) # parser can return a list of possible values
        if type(mtz_names) is str:
             mtz_names=[mtz_names]
        n=0
        for m in mtz_names:            
            if handler ==0:
                if os.path.isfile(os.path.join(apo_mtzs_path, m)):
                    mtz=os.path.join(apo_mtzs_path,m)
                    handler=1
                    n=n+1
                else:
                    handler=0
        if n>1:
            warnings.warn("Warning: detected multiple possible MTZ files containing phases.\n"+\
                          "Make sure that the parser mapping input MTZs to MTZs containing phases is sufficiently specific.")
    if handler ==1:
        pass
    else:
        mtz=os.path.join(apo_mtzs_path, os.path.basename(file))
        if os.path.isfile(mtz):
            handler=2
        else:
            try:
                mtz=glob.glob(os.path.join(apo_mtzs_path, f"*{os.path.splitext(os.path.basename(file))[0]}*.mtz"))[0]
                if os.path.isfile(mtz):
                    handler=3
            except:
                mtz=os.path.join(apo_mtzs_path, os.path.basename(file)[0:4]+".mtz")
                if os.path.isfile(mtz):
                    handler=4
                else:
                    handler=-1
    return mtz, handler

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
    parser                             = additional_args[6]
    rfree_label_in                     = additional_args[7]
    
    current = rs.read_mtz(file)
    success=False
    
    mtz, handler = find_phase_file(file, apo_mtzs_path, parser)
    if handler > 0:
        phases_df = rs.read_mtz(mtz)

        # add Rfree from phase mtz to reconstructed mtz, if not existing, generate rfree flags
        try:
            current = rs.utils.copy_rfree(current, phases_df, rfree_key=rfree_label_in)
        except:
            current = rs.utils.add_rfree(current)
            
        try:    
            current[phase_2FOFC_col_out] = phases_df[phase_2FOFC_col_in]
            current[phase_FOFC_col_out]  = phases_df[phase_FOFC_col_in]
            current.write_mtz(vae_reconstructed_with_phases_path + os.path.basename(file))
            success=True
        except Exception as e:
            print(e,flush=True)
    else:
        print(f"Handler: {handler} for input MTZ {mtz}")
    # pass
    # if handler > 0:
    #     print(f"Reading in apo phases using handler {handler}")
    # else:
    #     print(f"Failed to read in apo phases for {file}.\n")
        
    return [file, success]


def add_phases_pool(file_list, apo_mtzs_path, vae_reconstructed_with_phases_path, phase_2FOFC_col_out='PH2FOFCWT', phase_FOFC_col_out='PHFOFCWT',phase_2FOFC_col_in='PH2FOFCWT', phase_FOFC_col_in='PHFOFCWT', prefix=None, parser=None, ncpu=None, rfree_label_in=None):
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
            rfree_label_in (None or str)  : *input* MTZ column name for Rfree flags

        Returns:
            a dataframe reporting for each dataset whether phases were successfully added.
    """

    additional_args=[apo_mtzs_path, vae_reconstructed_with_phases_path, phase_2FOFC_col_out, phase_FOFC_col_out, phase_2FOFC_col_in, phase_FOFC_col_in, parser, rfree_label_in]
    input_args = zip(file_list, repeat(additional_args))
    with Pool(ncpu) as pool:
        metrics = pool.starmap(add_phases_from_pool_map, tqdm(input_args, total=len(file_list)))
            
        metrics_df = pd.DataFrame(metrics)
        metrics_df.columns=['file', 'success']
    
    return metrics_df[~metrics_df['success']]['file'].tolist()


def french_wilson_from_pool_map(file, additional_args):
    intensity_key  = additional_args[0]
    sigma_key      = additional_args[1]
    output_columns = additional_args[2]
    isotropic      = additional_args[3]
    minimum_sigma  = additional_args[4]
    bw             = additional_args[5]
    ds=rs.read_mtz(file)

    if isotropic:
        mean_intensity_method="isotropic"
    else:
        mean_intensity_method="anisotropic"
        
    ds_out=rs.algorithms.scale_merged_intensities(\
        ds, \
        intensity_key=intensity_key, \
        sigma_key=sigma_key, \
        output_columns=output_columns, \
        dropna=True, \
        inplace=False, \
        mean_intensity_method=mean_intensity_method, \
        bw=bw,
        minimum_sigma=minimum_sigma,
    )
    ds_out.write_mtz(file)
    return 1


def apply_french_wilson(file_list, intensity_key, sigma_key, output_columns=["I-FW","SIGI-FW","FP","SIGFP"], isotropic=True, minimum_sigma=0.1, bw=2.0, ncpu=None):
    """
    Apply French-Wilson scaling to observed intensities.
        Parameters:
            file_list (list of str) : list of input files (complete path!)
            intensity_key (str): name of intensity column to apply FW scaling to
            sigma_key (str): name columnn with  error estimates for observed intensities
            output_columns (list of four str): name of output columns for I-FW, SIGI-FW, FP, SIGFP (default: ["I-FW","SIGI-FW","FP", "SIGFP"]
            isotropic (bool) : Whether to use isotropic (True, default) or anisotropic estimation of Sigma
            minimum_sigma (float) : minimum value of Sigma when using a local averaging to calculate it (default: 0.1)
            bw (float): bandwidth of the exponential weights used to locally calculate Sigma.
            ncpu (int) : Number of CPUs available

        Returns:
            list of mtz files for which FW succeeded.
    """

    additional_args=[intensity_key, sigma_key, output_columns, isotropic, minimum_sigma, bw]
    input_args = zip(file_list, repeat(additional_args))

    # with get_context("spawn").Pool(ncpu) as pool:
    with Pool(ncpu) as pool:
        success = pool.starmap(french_wilson_from_pool_map, tqdm(input_args, total=len(file_list)))
            
   
    return success

def extrapolate_single_file(file, additional_args):
    '''
    Note: the input column names 'WT' and 'WDF' are generated by add_weights_single_file() and considered fixed.
          the output column names 'ESF_*' or 'WESF_*' are likewise considered fixed.
          please file an issue or pull request to change this behavior.
    '''
    # eps is the minimal VAE reconstruction error if none is available (def: 1e-6)
    eps       = 1.0e-6
    F_col     = additional_args[0]
    sigF_col  = additional_args[1]
    recons_col= additional_args[2]
    sig_recons= additional_args[3]
    diff_col  = additional_args[4]
    wt_col    = additional_args[5]
    weighted  = additional_args[6]
    ext_fact  = additional_args[7]
    redo      = additional_args[8]

    success=0
    try:
        current = rs.read_mtz(file)
    except Exception as e:
        print(e)
    
    col_names=[]
    for n in ext_fact:
        if weighted:
            col_names.append("WESF_"+str(n))
        else:
            col_names.append("ESF_"+str(n))
    
    if not redo:
        all_cols_exist=True
        for name in col_names:
            if name not in current.columns:
                all_cols_exist=False
                break
    else:
        all_cols_exist=False

    if all_cols_exist and not redo:
        print("Not doing extrapolation.")
    else:
        # include reconstruction errors iff we have them.
        if sig_recons not in current:
            current[sig_recons]=eps
            current[sig_recons]=current[sig_recons].astype("Stddev")
        
        if weighted:
            dF = current["WDF"].to_numpy()
        else:
            dF = current[diff_col].to_numpy()

        for i in range(len(ext_fact)):
            N=ext_fact[i]
            current[col_names[i]]=current[recons_col] + N*dF
            current[col_names[i]]=current[col_names[i]].astype("SFAmplitude")
            current["SIG"+col_names[i]]=np.sqrt(N**2     * current[sigF_col  ].to_numpy()**2 + \
                                                (N-1)**2 * current[sig_recons].to_numpy()**2)
            current["SIG"+col_names[i]]=current["SIG"+col_names[i]].astype("Stddev")
        
        current.write_mtz(file)
        success=1

    return success

def extrapolate(file_list, F_col="F-obs-scaled", sigF_col="SIGF-obs-scaled", recons_col="recons", sigF_recons_col="SIG_recons", diff_col="diff", wt_col="WT",\
                redo=True, weighted=False, extrapolate_factors=[2,4,8], ncpu=1):
    """
    Calculate extrapolated differences (weighted or unweighted and add the corresponding files in file_list. 
    Note that this is relative to Frecon, but that it is easy to interconvert: N*(Fo-Fr)+Fo = (N+1)*(Fo-Fr)+Fr
        
        Parameters:
        -----------
            file_list (list of str) : list of input files (complete path!)
            F_col (str) : name of column containing observed amplitudes (*scaled to reference mtz*)
            sigF_col (str) : name of column containing error estimates for *scaled* measured amplitudes
            recons_col (str) : name of column containing VAE-reconstructed amplitudes (after rescaling to match the ref mtz)
            sigF_recons_col (str) : name of column containing error estimates for reconstructed ampls (default: "SIG_recons")
            diff_col (str): name of column for output Fobs-Frecon (default: "diff")
            wt_col (str) : name of column containing weights
            redo (bool): whether to override existing weights (default: True)
            weighted (bool) : whether to apply weights to the deltaF (default: False)
            extrapolate_factors (list of int) : list of extrapolation factors (default: [2, 4, 8]
            ncpu (int): Number of CPUs to use for multiprocessing (default: 1)
            
        Returns:
        --------
            TBD
    """
    
    additional_args=[F_col, sigF_col, recons_col, sigF_recons_col, diff_col, wt_col, weighted, extrapolate_factors, redo]
    if ncpu>1:
        input_args = zip(file_list,repeat(additional_args))
        with Pool(ncpu) as pool:
            result = pool.starmap(extrapolate_single_file, tqdm(input_args, total=len(file_list)))
        
    else:
        result=[]
        for filename in tqdm(file_list):
            result.append(extrapolate_single_file(filename, additional_args))

    return result

# avoid OpenMP oversubscription when also using multiprocessing
os.environ.setdefault("OMP_NUM_THREADS", "1")

def _pick(df_cols, *candidates, required=True, label="column"):
    """Find the first matching column name from candidates (case-insensitive)."""
    cols_l = {str(c).lower(): c for c in df_cols}
    for cand in candidates:
        if str(cand).lower() in cols_l:
            return cols_l[str(cand).lower()]
    if required:
        raise KeyError(f"Missing required {label}. Tried: {candidates}")
    return None


def _export_one_dataset(args):
    """
    Worker: writes maps for one dtag and returns (dtag, ev_rows, site_rows, log_msgs).
    Expects df_ev to have ONLY normalized columns:
      ['_event_num','_site_num','_x','_y','_z','_cluster_size']
    """
    (dtag, df_ev, mtz_dir, out_root, wdf_label, phase_label, esfn, recon_label,
     mtz_pattern, input_pdb_dir, input_pdb_pattern, input_map_mtz_dir,
     input_map_mtz_pattern, use_new_style, default_rwork, default_rfree) = args

    log = []
    ev_rows = []
    site_rows = []

    try:
        dtag = str(dtag)
        ddir = os.path.join(out_root, "processed_datasets", dtag)
        os.makedirs(ddir, exist_ok=True)

        mtz_path = os.path.join(mtz_dir, mtz_pattern.format(dtag=dtag))
        if not os.path.exists(mtz_path):
            log.append(f"[{dtag}] MTZ not found: {mtz_path}")
            return (dtag, ev_rows, site_rows, log)

        # read mtz
        try:
            ds = rs.read_mtz(mtz_path)
        except Exception as e:
            log.append(f"[{dtag}] Failed to read MTZ: {mtz_path} ({e})")
            return (dtag, ev_rows, site_rows, log)

        esf_label = f"ESF_{esfn}"
        for need in (wdf_label, phase_label, esf_label):
            if need not in ds.columns:
                raise KeyError(f"[{dtag}] Missing column '{need}' in {os.path.basename(mtz_path)}")

        # optional inputs for (2)Fo-Fc/Fo-Fc and model
        if input_pdb_dir:
            src_pdb = os.path.join(input_pdb_dir, input_pdb_pattern.format(dtag=dtag))
            if os.path.exists(src_pdb):
                shutil.copy(src_pdb, os.path.join(ddir, f"{dtag}-pandda-input.pdb"))
            else:
                log.append(f"[{dtag}] Input PDB not found: {src_pdb}")
        if input_map_mtz_dir:
            src_mapmtz = os.path.join(input_map_mtz_dir, input_map_mtz_pattern.format(dtag=dtag))
            if os.path.exists(src_mapmtz):
                shutil.copy(src_mapmtz, os.path.join(ddir, f"{dtag}-pandda-input.mtz"))
            else:
                log.append(f"[{dtag}] Input map MTZ not found: {src_mapmtz}")

        # --- Z / average MTZ ---
        out_zavg = os.path.join(ddir, f"{dtag}-pandda-output.mtz")
        zavg = ds[[wdf_label, phase_label]].copy()
        zavg = zavg.rename(columns={wdf_label: "FZVALUES", phase_label: "PHZVALUES"})
        zavg["FZVALUES"] = zavg["FZVALUES"].astype("F")
        zavg["PHZVALUES"] = zavg["PHZVALUES"].astype("P")
        if recon_label and recon_label in ds.columns:
            zavg["FGROUND"] = ds[recon_label].astype("F")
            zavg["PHGROUND"] = ds[phase_label].astype("P")
        zavg.write_mtz(out_zavg)

        one_minus_bdc = 1.0 / float(esfn)

        # Per-event MTZs + CSV rows
        for _, ev in df_ev.iterrows():
            i_ev = int(ev["_event_num"])

            if use_new_style:
                out_event = os.path.join(ddir, f"{dtag}-pandda-output-event-{i_ev:03d}.mtz")
                evds = ds[[esf_label, phase_label]].copy()
                evds = evds.rename(columns={esf_label: "FEVENT", phase_label: "PHEVENT"})
                evds["FEVENT"] = evds["FEVENT"].astype("F")
                evds["PHEVENT"] = evds["PHEVENT"].astype("P")
                evds.write_mtz(out_event)
            else:
                out_event = os.path.join(ddir, f"{dtag}-event_{i_ev}_1-BDC_{one_minus_bdc:.3f}_map.native.mtz")
                evds = ds[[esf_label, phase_label]].copy()
                evds = evds.rename(columns={esf_label: "FWT", phase_label: "PHWT"})
                evds["FWT"] = evds["FWT"].astype("F")
                evds["PHWT"] = evds["PHWT"].astype("P")
                evds.write_mtz(out_event)

            # analysed_resolution via Gemmi
            mtz_g = gemmi.read_mtz_file(os.fspath(mtz_path))
            analysed_resolution = f"{mtz_g.resolution_high():.3f}"

            ev_rows.append({
                "dtag": dtag,
                "event_num": i_ev,
                "site_num": int(ev["_site_num"]) if pd.notna(ev["_site_num"]) else i_ev,
                "1-BDC": f"{one_minus_bdc:.6f}",
                "x": float(ev["_x"]),
                "y": float(ev["_y"]),
                "z": float(ev["_z"]),
                "analysed_resolution": analysed_resolution,
                "r_work": default_rwork,
                "r_free": default_rfree,
                "Viewed": "False",
                "cluster_size": (int(ev["_cluster_size"]) if pd.notna(ev.get("_cluster_size")) else ""),
            })

        # Sites table (unique per dataset)
        for s in sorted(set(pd.to_numeric(df_ev["_site_num"], errors="coerce").dropna().astype(int).tolist())):
            site_rows.append({"site_num": int(s), "dtag": dtag})

    except Exception as e:
        log.append(f"[{dtag}] ERROR: {e}")

    return (dtag, ev_rows, site_rows, log)


def export_valdo_to_pandda_inspect(
    blobs_pickle,
    mtz_dir,
    out_root,
    *,
    wdf_label="WDF",
    phase_label="Refine_PH2FOFCWT",
    esfn=8,
    recon_label=None,
    mtz_pattern="{dtag}.mtz",
    input_pdb_dir=None,
    input_pdb_pattern="{dtag}.pdb",
    input_map_mtz_dir=None,
    input_map_mtz_pattern="{dtag}.mtz",
    use_new_style=True,
    default_rwork="NA",
    default_rfree="NA",
    top_n=200,               # export only the top-N blobs globally (by score if present)
    n_procs=1,               # number of worker processes
):
    """
    Faster VALDO → PanDDA export with top-N filter and per-dataset parallelization.

    - top_n: If 'score' exists, pick the top-N rows across ALL blobs by descending score.
             Else, take the first N rows in the pickle. Events renumber per-dtag.
    - n_procs: Number of worker processes (<= cpu_count()).
    """
    # Normalize paths
    blobs_pickle = os.fspath(blobs_pickle)
    mtz_dir = os.fspath(mtz_dir)
    out_root = os.fspath(out_root)
    if input_pdb_dir is not None:
        input_pdb_dir = os.fspath(input_pdb_dir)
    if input_map_mtz_dir is not None:
        input_map_mtz_dir = os.fspath(input_map_mtz_dir)

    os.makedirs(os.path.join(out_root, "processed_datasets"), exist_ok=True)
    os.makedirs(os.path.join(out_root, "results"), exist_ok=True)

    # Load blobs
    blobs = pd.read_pickle(blobs_pickle)
    if not isinstance(blobs, pd.DataFrame):
        raise TypeError(f"Pickle did not contain a pandas DataFrame: {type(blobs)}")

    # Resolve columns
    col_dtag  = _pick(blobs.columns, "dtag", "dataset", "sample", "id", label="dtag")
    col_x     = _pick(blobs.columns, "x", "cart_x", "x_Å", "x_A", "cenx", "center_x", label="x")
    col_y     = _pick(blobs.columns, "y", "cart_y", "y_Å", "y_A", "ceny", "center_y", label="y")
    col_z     = _pick(blobs.columns, "z", "cart_z", "z_Å", "z_A", "cenz", "center_z", label="z")
    col_site  = _pick(blobs.columns, "site", "site_id", "site_num", required=False, label="site")
    col_score = _pick(blobs.columns, "score", "zscore", "z", "peak_value", required=False, label="score")
    col_size  = _pick(blobs.columns, "cluster_size", "n_voxels", "size", "members", required=False, label="cluster_size")

    # Top-N across ALL blobs
    if col_score and col_score in blobs.columns:
        blobs_top = blobs.sort_values(col_score, ascending=False).head(int(top_n)).copy()
    else:
        blobs_top = blobs.head(int(top_n)).copy()

    # Pre-normalize for workers
    blobs_top["_dtag"] = blobs_top[col_dtag].astype(str)
    blobs_top["_x"] = pd.to_numeric(blobs_top[col_x], errors="coerce")
    blobs_top["_y"] = pd.to_numeric(blobs_top[col_y], errors="coerce")
    blobs_top["_z"] = pd.to_numeric(blobs_top[col_z], errors="coerce")
    blobs_top["_site_num"] = (
        pd.to_numeric(blobs_top[col_site], errors="coerce") if (col_site and col_site in blobs_top.columns)
        else pd.NA
    )
    blobs_top["_cluster_size"] = (
        pd.to_numeric(blobs_top[col_size], errors="coerce") if (col_size and col_size in blobs_top.columns)
        else pd.NA
    )

    # Per-dtag group with local event numbering
    groups = []
    for dtag, df in blobs_top.groupby("_dtag"):
        if col_score and col_score in df.columns:
            df = df.sort_values(col_score, ascending=False).reset_index(drop=True)
        else:
            df = df.reset_index(drop=True)
        df["_event_num"] = np.arange(1, len(df) + 1, dtype=int)
        if df["_site_num"].isna().all():
            df["_site_num"] = df["_event_num"]
        groups.append((dtag, df))

    # Build worker args
    worker_args = [
        (dtag, df_ev[["_event_num", "_site_num", "_x", "_y", "_z", "_cluster_size"]],
         mtz_dir, out_root, wdf_label, phase_label, esfn, recon_label,
         mtz_pattern, input_pdb_dir, input_pdb_pattern, input_map_mtz_dir,
         input_map_mtz_pattern, use_new_style, default_rwork, default_rfree)
        for (dtag, df_ev) in groups
    ]

    # Run workers
    results = []
    if int(n_procs) > 1:
        procs = min(int(n_procs), cpu_count())
        chunksize = max(1, len(worker_args) // (procs * 4))
        with Pool(processes=procs) as pool:
            for r in pool.imap_unordered(_export_one_dataset, worker_args, chunksize=chunksize):
                results.append(r)
    else:
        for args in worker_args:
            results.append(_export_one_dataset(args))

    # Collect rows / logs
    all_ev_rows, all_site_rows = [], []
    msgs = []
    for dtag, ev_rows, site_rows, log in results:
        all_ev_rows.extend(ev_rows)
        all_site_rows.extend(site_rows)
        msgs.extend(log)

    # Write CSVs
    ev_df = pd.DataFrame(all_ev_rows, columns=[
        "dtag","event_num","site_num","1-BDC","x","y","z",
        "analysed_resolution","r_work","r_free","Viewed","cluster_size"
    ])
    si_df = pd.DataFrame(all_site_rows, columns=["site_num","dtag"])

    ev_out = os.path.join(out_root, "results", "pandda_analyse_events.csv")
    si_out = os.path.join(out_root, "results", "pandda_analyse_sites.csv")
    ev_df.to_csv(ev_out, index=False)
    si_df.to_csv(si_out, index=False)

    for m in msgs:
        warnings.warn(m)

    print(f"Wrote {len(ev_df)} events from {ev_df['dtag'].nunique()} datasets → {out_root} (top_n={top_n}, n_procs={n_procs})")
    return out_root

