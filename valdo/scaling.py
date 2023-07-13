"""
Scaling structural factor magnitudes to a reference dataset, with anisotropic scaling parameters
"""
import numpy as np
import reciprocalspaceship as rs
import pandas as pd

import time


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

        TODO: opt_getku, use stepwise optimizer to further optimize the parameters, in pytorch
        """
        uaniso = np.array([0.]*6) # initialize 
        for _ in range(n_iter):
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
    
    def batch_scaling(self, mtz_path_list, outputmtz_path='./scaled_mtzs/', reportfile='./scaling_data.json', verbose=True, n_iter=5):

        metrics = []
        for path in mtz_path_list:
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

            FB_complete = temp_mtz[self.columns[0]].to_numpy() 
            hkl_complete = temp_mtz.get_hkls()
            
            temp_mtz = temp_mtz.reset_index()
            temp_mtz[self.columns[0]+'-scaled'] = rs.DataSeries(self.scaleit(FB_complete, ln_k, uaniso, hkl_complete), dtype="SFAmplitude")
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
        
        pd.DataFrame(metrics).to_pickle(outputmtz_path + 'metrics.pkl')
        return metrics

            
    







            

        




