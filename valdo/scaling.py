"""
Scaling structural factor magnitudes to a reference dataset, with anisotropic scaling parameters
"""
import numpy as np
import reciprocalspaceship as rs
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from itertools import repeat

import time
import torch
from valdo.helper import try_gpu


def get_aniso_args_np(uaniso, reciprocal_cell_paras, hkl):
    U11, U22, U33, U12, U13, U23 = uaniso
    h, k, l = hkl.T
    ar, br, cr, cos_alphar, cos_betar, cos_gammar = reciprocal_cell_paras
    args = 2*np.pi**2*(
        U11 * h**2 * ar**2
        + U22 * k**2 * br**2
        + U33 * l**2 * cr**2
        + 2 * (h * k * U12 * ar * br * cos_gammar 
               + h * l * U13 * ar * cr * cos_betar 
               + k * l * U23 * br * cr * cos_alphar)
    )
    return args

def aniso_scaling_torch(uaniso, reciprocal_cell_paras, hkl):
    HKL_tensor = torch.tensor(hkl, device=uaniso.device)
    U11, U22, U33, U12, U13, U23 = uaniso
    h, k, l = HKL_tensor.T
    ar, br, cr, cos_alphar, cos_betar, cos_gammar = reciprocal_cell_paras
    args = (
        U11 * h**2 * ar**2
        + U22 * k**2 * br**2
        + U33 * l**2 * cr**2
        + 2 * (h * k * U12 * ar * br * cos_gammar 
               + h * l * U13 * ar * cr * cos_betar 
               + k * l * U23 * br * cr * cos_alphar)
    )
    return torch.exp(-2.0 * np.pi**2 * args)

class Scaler(object):
    """
    reference_mtz : path to mtz file as the reference dataset
    
    columns : list of column names to be used
        The first name is used for scaling, while the remaining 
        names will be saved as is without any alterations.
    """
    def __init__(self, reference_mtz, columns=['F-obs', 'SIGF-obs']):
        self.columns = columns
        self.base_mtz = rs.read_mtz(reference_mtz)[columns]
        self.base_mtz.dropna(inplace=True)

        # Record reciprocal space parameters
        reciprocal_cell = self.base_mtz.cell.reciprocal()
        self.reciprocal_cell_paras = [reciprocal_cell.a,
                        reciprocal_cell.b,
                        reciprocal_cell.c,
                        np.cos(np.deg2rad(reciprocal_cell.alpha)),
                        np.cos(np.deg2rad(reciprocal_cell.beta)),
                        np.cos(np.deg2rad(reciprocal_cell.gamma))]
    
    def _get_ln_k(self, FA, FB, hkl, uaniso):
        args = get_aniso_args_np(uaniso, self.reciprocal_cell_paras, hkl)
        ln_k = np.mean(args + np.log(FA/FB))
        return ln_k
    
    def _get_uaniso(self, FA, FB, hkl, ln_k):
        V = np.concatenate([hkl**2, 2 * hkl[:, [0, 2, 1]] * hkl[:, [1, 0, 2]]], axis=-1)
        Z = (np.log(FA/FB) - ln_k)/(2*np.pi**2)
        M = V.T @ V
        b = -np.sum(Z * V.T, axis=-1)
        uaniso = np.linalg.inv(M) @ b
        return uaniso

    def ana_getku(self, FA, FB, hkl, n_iter=5):
        """
        Use analytical scaling method to get parameter k and uaniso, with purely numpy.
        Afonine, P. V., et al. Acta Crystallographica Section D: Biological Crystallography 69.4 (2013): 625-634.
        """
        uaniso = np.array([0.]*6) # initialize 
        for _ in range(n_iter):
            ln_k = self._get_ln_k(FA, FB, hkl, uaniso)
            uaniso = self._get_uaniso(FA, FB, hkl, ln_k)
        return ln_k, uaniso

    def opt_getku(self, FA, FB, hkl, ln_k, uaniso, device, n_steps=50, lr=0.01):
        """
        Use adam numerical optimization for better scaling
        """
        FA_tensor = torch.tensor(FA, requires_grad=True, dtype=torch.float32, device=device)
        FB_tensor = torch.tensor(FB, requires_grad=True, dtype=torch.float32, device=device)
        ln_k_init = torch.tensor(ln_k, requires_grad=True, dtype=torch.float32, device=device)
        uaniso_init = torch.tensor(uaniso, requires_grad=True, dtype=torch.float32, device=device)
        adam_opt = torch.optim.Adam([ln_k_init, uaniso_init], lr=lr)
        for i in range(n_steps):
            adam_opt.zero_grad()
            FB_scaled = self.scaleit_torch(FB_tensor, ln_k_init, uaniso_init, hkl)
            loss = torch.sum((FA_tensor - FB_scaled)**2)
            loss.backward()
            adam_opt.step()
        return ln_k_init.detach().cpu().numpy(), uaniso_init.detach().cpu().numpy()

    def scaleit(self, FB, ln_k, uaniso, hkl):
        args = get_aniso_args_np(uaniso, self.reciprocal_cell_paras, hkl)
        FB_scaled = np.exp(ln_k) * np.exp(-args) * FB
        return FB_scaled

    def scaleit_torch(self, FB, ln_k, uaniso, hkl):
        aniso_scaling = aniso_scaling_torch(uaniso, self.reciprocal_cell_paras, hkl)
        FB_scaled = torch.exp(ln_k) * aniso_scaling * FB
        return FB_scaled

    def get_metric(self, FA, FB, uaniso, ln_k, hkl):
        # Before
        LS_i = np.sum((FA - FB)**2)
        corr_i = np.corrcoef(FA, FB)[0,1]
        
        # After
        FB_scaled = self.scaleit(FB, ln_k, uaniso, hkl)
        LS_f = np.sum((FA - FB_scaled)**2)
        corr_f = np.corrcoef(FA, FB_scaled)[0,1]
        
        return [LS_i, corr_i, LS_f, corr_f]
    
    def batch_scaling(self, mtz_path_list, 
                      outputmtz_path='./scaled_mtzs/', 
                      prefix="", 
                      verbose=True, 
                      n_iter=5, 
                      when_opt=0.2,
                      n_opt=50, 
                      lr_opt=0.01, 
                      opt_device=try_gpu()):
        """
        when_opt: "all" or "never" or float [0.0, 1.0]
            argument to control when to use numerical optimization after analytical initialization
            1. "all", use numerical optimization for every dataset
            2. "never", don't use numerical optimization at all
            3. float x between 0.0 and 1.0, apply numerical optimization when (LS_i - LS_f)/LS_i < x, e.g. analytical initialization
            is not improving LS error large enough. 
        """
        metrics = []
        for path in tqdm(mtz_path_list):
            start_time = time.time()
            concrete_filename = path.split('/')[-1].replace(".mtz", "") # "PTP1B_yxxx_idxs"
            temp_mtz = rs.read_mtz(path)[self.columns].dropna()
            merge = self.base_mtz.merge(temp_mtz, left_index=True, right_index=True, 
                                        suffixes=('ref', 'target'), check_isomorphous=False)
            

            FA = merge[self.columns[0]+"ref"].to_numpy()
            FB = merge[self.columns[0]+"target"].to_numpy()
            hkl = merge.get_hkls()

            ln_k, uaniso = self.ana_getku(FA, FB, hkl, n_iter=n_iter)
            metric = self.get_metric(FA, FB, uaniso, ln_k, hkl)

            # apply numerical optimization when condition satisfied
            if type(when_opt) is str:
                if when_opt == "all":
                    ln_k, uaniso = self.opt_getku(FA, FB, hkl, ln_k, uaniso, opt_device, n_steps=n_opt, lr=lr_opt)
                    metric = self.get_metric(FA, FB, uaniso, ln_k, hkl)
                elif when_opt == "never":
                    pass
                else:
                    raise ValueError("when_opt has to be 'all', 'never' or float between 0.0 and 1.0!")
            elif type(when_opt) is float:
                assert when_opt >= 0.0 and when_opt <= 1.0, "when_opt has to be 'all', 'never' or float between 0.0 and 1.0!"
                ana_improvement = (metric[0] - metric[2] + 1e-3) / (metric[0] + 1e-3) # (LS_i - LS_f) / LS_i
                if ana_improvement < when_opt:
                    ln_k, uaniso = self.opt_getku(FA, FB, hkl, ln_k, uaniso, opt_device, n_steps=n_opt, lr=lr_opt)
                    metric = self.get_metric(FA, FB, uaniso, ln_k, hkl)
                else:
                    pass
            else:
                raise ValueError("when_opt has to be 'all', 'never' or float between 0.0 and 1.0!")

            FB_complete = temp_mtz[self.columns[0]].to_numpy() 
            SIGFB_complete = temp_mtz[self.columns[1]].to_numpy() 
            hkl_complete = temp_mtz.get_hkls()
            
            temp_mtz = temp_mtz.reset_index()
            temp_mtz[self.columns[0]+'-scaled'] = rs.DataSeries(self.scaleit(   FB_complete, ln_k, uaniso, hkl_complete), dtype="SFAmplitude")
            temp_mtz[self.columns[1]+'-scaled'] = rs.DataSeries(self.scaleit(SIGFB_complete, ln_k, uaniso, hkl_complete), dtype="Stddev")
            temp_mtz = temp_mtz.set_index(['H', 'K', 'L'])
            # Save the scaled mtz file
            temp_mtz.write_mtz(outputmtz_path+concrete_filename+".mtz")

            str_ = f"Time: {time.time()-start_time:.3f}"
            if verbose:
                print(f"LS before:  {metric[0]:.1f}", f"LS after: {metric[2]:.0f}", flush=True)
                print(f"Corr before:  {metric[1]:.3f}", f"Corr after: {metric[3]:.3f}", flush=True)
                print(str_, flush=True)
                print("="*20)
                
            metrics.append([concrete_filename, *metric])
        
        metrics_df = pd.DataFrame(metrics)
        metrics_df.columns=['file', 'start_LS', 'start_corr', 'end_LS', 'end_corr']
        metrics_df.to_pickle(outputmtz_path + prefix + 'scaling_metrics.pkl')
        print("Scaling metrics have been saved at:", outputmtz_path + prefix + 'scaling_metrics.pkl', flush=True)
        return metrics

# Multiprocessing variant #

class Scaler_pool(object):
    """
    reference_mtz : path to mtz file as the reference dataset
    
    columns : list of column names to be used
        The first name is used for scaling, while the remaining 
        names will be saved as is without any alterations.
    """
    def __init__(self, reference_mtz, columns=['F-obs', 'SIGF-obs'],n_iter=5,verbose=False, ncpu=None):
        # self.starmap = Pool(ncpu).starmap
        self.columns = columns
        self.base_mtz = rs.read_mtz(reference_mtz)[columns]
        self.base_mtz.dropna(inplace=True)
        self.n_iter=n_iter
        self.verbose=verbose
        self.ncpu=ncpu
        # Record reciprocal space parameters
        reciprocal_cell = self.base_mtz.cell.reciprocal()
        self.reciprocal_cell_paras = [reciprocal_cell.a,
                        reciprocal_cell.b,
                        reciprocal_cell.c,
                        np.cos(np.deg2rad(reciprocal_cell.alpha)),
                        np.cos(np.deg2rad(reciprocal_cell.beta)),
                        np.cos(np.deg2rad(reciprocal_cell.gamma))]
        # self.start()
    
    def _get_ln_k(self, FA, FB, hkl, uaniso):
        args = get_aniso_args_np(uaniso, self.reciprocal_cell_paras, hkl)
        ln_k = np.mean(args + np.log(FA/FB))
        return ln_k
    
    def _get_uaniso(self, FA, FB, hkl, ln_k):
        V = np.concatenate([hkl**2, 2 * hkl[:, [0, 2, 1]] * hkl[:, [1, 0, 2]]], axis=-1)
        Z = (np.log(FA/FB) - ln_k)/(2*np.pi**2)
        M = V.T @ V
        b = -np.sum(Z * V.T, axis=-1)
        uaniso = np.linalg.inv(M) @ b
        return uaniso

    def ana_getku(self, FA, FB, hkl):
        """
        Use analytical scaling method to get parameter k and uaniso, with purely numpy.
        Afonine, P. V., et al. Acta Crystallographica Section D: Biological Crystallography 69.4 (2013): 625-634.

        TODO: opt_getku, use stepwise optimizer to further optimize the parameters, in pytorch
        """
        uaniso = np.array([0.]*6) # initialize 
        for _ in range(self.n_iter):
            ln_k = self._get_ln_k(FA, FB, hkl, uaniso)
            uaniso = self._get_uaniso(FA, FB, hkl, ln_k)
        return ln_k, uaniso

    def scaleit(self, FB, ln_k, uaniso, hkl):
        args = get_aniso_args_np(uaniso, self.reciprocal_cell_paras, hkl)
        FB_scaled = np.exp(ln_k) * np.exp(-args) * FB
        return FB_scaled

    def get_metric(self, FA, FB, uaniso, ln_k, hkl):
        # Before
        LS_i = np.sum((FA - FB)**2)
        corr_i = np.corrcoef(FA, FB)[0,1]
        
        # After
        FB_scaled = self.scaleit(FB, ln_k, uaniso, hkl)
        LS_f = np.sum((FA - FB_scaled)**2)
        corr_f = np.corrcoef(FA, FB_scaled)[0,1]
        
        return [LS_i, corr_i, LS_f, corr_f]
    
    def batch_scaling_from_pool_map(self, mtz_path, additional_args):
        outputmtz_path = additional_args[0]
        
        concrete_filename = mtz_path.split('/')[-1].replace(".mtz", "") # "PTP1B_yxxx_idxs"
        temp_mtz = rs.read_mtz(mtz_path)[self.columns].dropna()
        merge = self.base_mtz.merge(temp_mtz, left_index=True, right_index=True, 
                                    suffixes=('ref', 'target'), check_isomorphous=False)

        FA = merge[self.columns[0]+"ref"].to_numpy()
        FB = merge[self.columns[0]+"target"].to_numpy()
        hkl = merge.get_hkls()

        ln_k, uaniso = self.ana_getku(FA, FB, hkl)
        metric = self.get_metric(FA, FB, uaniso, ln_k, hkl)

        FB_complete = temp_mtz[self.columns[0]].to_numpy() 
        SIGFB_complete = temp_mtz[self.columns[1]].to_numpy() 
        hkl_complete = temp_mtz.get_hkls()
        
        temp_mtz = temp_mtz.reset_index()
        temp_mtz[self.columns[0]+'-scaled'] = rs.DataSeries(self.scaleit(   FB_complete, ln_k, uaniso, hkl_complete), dtype="SFAmplitude")
        temp_mtz[self.columns[1]+'-scaled'] = rs.DataSeries(self.scaleit(SIGFB_complete, ln_k, uaniso, hkl_complete), dtype="Stddev")
        temp_mtz = temp_mtz.set_index(['H', 'K', 'L'])
        # Save the scaled mtz file
        temp_mtz.write_mtz(outputmtz_path+concrete_filename+".mtz")

        if self.verbose:
            print(f"LS before:  {metric[0]:.1f}", f"LS after: {metric[2]:.0f}", flush=True)
            print(f"Corr before:  {metric[1]:.3f}", f"Corr after: {metric[3]:.3f}", flush=True)
            print("="*20)
        return [concrete_filename] + metric

    
    def batch_scaling(self, mtz_path_list, outputmtz_path='./scaled_mtzs/', prefix=None):

        additional_args=[outputmtz_path]
        input_args = zip(mtz_path_list, repeat(additional_args))
        with Pool(self.ncpu) as pool:
            metrics = pool.starmap(self.batch_scaling_from_pool_map, tqdm(input_args, total=len(mtz_path_list)))
            
        metrics_df = pd.DataFrame(metrics)
        metrics_df.columns=['file', 'start_LS', 'start_corr', 'end_LS', 'end_corr']
        metrics_df.to_pickle(outputmtz_path + prefix + 'scaling_metrics.pkl')
        print("Scaling metrics have been saved at:", outputmtz_path + prefix + 'scaling_metrics.pkl', flush=True)
        return metrics_df
    