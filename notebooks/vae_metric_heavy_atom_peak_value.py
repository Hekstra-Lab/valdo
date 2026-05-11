import pandas as pd
import numpy as np
import gemmi
import reciprocalspaceship as rs
from tqdm import tqdm

# Use the lig log from DK's results
lig_log = pd.read_pickle("./lig_heavy_atoms.pkl")
DK_bound = lig_log[lig_log['author'] == 'Keedy'].copy()

phyllis_dir="/n/holyscratch01/hekstra_lab/phyllis/"
my_dir     ="/n/holyscratch01/hekstra_lab/dhekstra/valdo-tests/"

basepath = my_dir + 'pipeline/'
vae_reconstructed_path             = basepath + 'vae/reconstructed/'
vae_reconstructed_with_phases_path = '/n/holyscratch01/hekstra_lab/dhekstra/valdo-tests/pipeline/vae/reconstructed_w_phases/'
bound_models_standardized_path     = '/n/holyscratch01/hekstra_lab/dhekstra/phyllis/PTP1B_DK/all_bound_models_reindexed_v2/short_setting_0049/'

# Get a mean peak value as a VAE metric

DK_bound.loc[:, 'lig_heavy_peak'] = 0.0
DK_bound.loc[:, 'is_highest_peak(<5A)'] = 0.0

diff_col="WDF"
for pdbid in tqdm(DK_bound['sample']):
    try:
        try:
            mtz_file = gemmi.read_mtz_file(vae_reconstructed_with_phases_path + f'{pdbid}_0.mtz')
        except:
            try:
                mtz_file = gemmi.read_mtz_file(vae_reconstructed_with_phases_path + f'{pdbid}_1.mtz')
            except Exception as e:
                print(e)

        st = gemmi.read_pdb(bound_models_standardized_path + f'{pdbid}.pdb')

        real_grid = mtz_file.transform_f_phi_to_map(diff_col, 'refine_PH2FOFCWT', sample_rate=3.0)
        real_grid.normalize()

        sel = gemmi.Selection('[CL,Br,S,I]')
        sel_model = sel.copy_model_selection(st[0])
        lig_heavy_atoms = [i for i in list(sel_model.all()) if i.residue.name == 'LIG']

        dis_lists = []
        peak_values = []
        for cra in lig_heavy_atoms:

            eq_points = []
            ops = real_grid.spacegroup.operations()
            atom = cra.atom

            a,b,c = np.unravel_index(real_grid.array.argmax(), real_grid.array.shape)
            tmp = real_grid.get_fractional(a,b,c)
            peak_pos = st.cell.orthogonalize(gemmi.Fractional(tmp.x, tmp.y, tmp.z))
            dis_list = []

            for op in ops:
                SG_mapped=op.apply_to_xyz(st.cell.fractionalize(atom.pos).tolist())
                tmp = SG_mapped-np.floor(np.array(SG_mapped))
                SG_mapped = gemmi.Fractional(*tmp)
                eq_points.append(SG_mapped)
                SG_mapped_orth = st.cell.orthogonalize(SG_mapped)
                dis_list.append(np.sqrt(np.sum(np.array((peak_pos - SG_mapped_orth).tolist())**2)))

            peak_value = []
            for pos in eq_points:
                a = round(pos.x * real_grid.nu)
                b = round(pos.y * real_grid.nv)
                c = round(pos.z * real_grid.nw)
                peak_value.append(real_grid.get_value(a, b, c))

            dis_lists.append(dis_list)
            peak_values.append(peak_value)

        log_peak = np.max(peak_values)
        log_ismaxpeak = np.any(np.array(dis_lists) < 5.0)

        DK_bound.loc[DK_bound['sample']==pdbid, 'lig_heavy_peak'] = log_peak
        DK_bound.loc[DK_bound['sample']==pdbid, 'is_highest_peak(<5A)'] = log_ismaxpeak
    except Exception as e:
        print(e)

print(DK_bound['lig_heavy_peak'])
print(np.mean(DK_bound['lig_heavy_peak']))

# Same metric for Fo-Fc maps

DK_bound.loc[:, ('lig_heavy_peak')] = 0.0
DK_bound.loc[:, ('is_highest_peak(<5A)')] = 0.0

for pdbid in tqdm(DK_bound['sample']):
    mtz_file = gemmi.read_mtz_file(phyllis_dir+f'PTP1B_DK/pandda_input_models_refined_waters/PTP1B_y{pdbid}_pandda_input_reindexed_refine_001.mtz')
    st = gemmi.read_pdb(phyllis_dir+f'PTP1B_DK/all_bound_models_reindexed/PTP1B_y{pdbid}_bound_state_reindexed.pdb')

    real_grid = mtz_file.transform_f_phi_to_map('FOFCWT', 'PHFOFCWT', sample_rate=3.0)
    real_grid.normalize()

    sel = gemmi.Selection('[CL,Br,S,I]')
    sel_model = sel.copy_model_selection(st[0])
    lig_heavy_atoms = [i for i in list(sel_model.all()) if i.residue.name == 'LIG']

    dis_lists = []
    peak_values = []
    for cra in lig_heavy_atoms:

        eq_points = []
        ops = real_grid.spacegroup.operations()
        atom = cra.atom

        a,b,c = np.unravel_index(real_grid.array.argmax(), real_grid.array.shape)
        tmp = real_grid.get_fractional(a,b,c)
        peak_pos = st.cell.orthogonalize(gemmi.Fractional(tmp.x, tmp.y, tmp.z))
        dis_list = []

        for op in ops:
            SG_mapped=op.apply_to_xyz(st.cell.fractionalize(atom.pos).tolist())
            tmp = SG_mapped-np.floor(np.array(SG_mapped))
            SG_mapped = gemmi.Fractional(*tmp)
            eq_points.append(SG_mapped)
            SG_mapped_orth = st.cell.orthogonalize(SG_mapped)
            dis_list.append(np.sqrt(np.sum(np.array((peak_pos - SG_mapped_orth).tolist())**2)))

        peak_value = []
        for pos in eq_points:
            a = round(pos.x * real_grid.nu)
            b = round(pos.y * real_grid.nv)
            c = round(pos.z * real_grid.nw)
            peak_value.append(real_grid.get_value(a, b, c))

        dis_lists.append(dis_list)
        peak_values.append(peak_value)

    log_peak = np.max(peak_values)
    log_ismaxpeak = np.any(np.array(dis_lists) < 5.0)

    DK_bound.loc[DK_bound['sample']==pdbid, 'lig_heavy_peak'] = log_peak
    DK_bound.loc[DK_bound['sample']==pdbid, 'is_highest_peak(<5A)'] = log_ismaxpeak

print(DK_bound['lig_heavy_peak'])
print(np.mean(DK_bound['lig_heavy_peak']))

# Zmap mean peak value as metric

DK_bound.loc[:, ('lig_heavy_peak')] = 0.0
DK_bound.loc[:, ('is_highest_peak(<5A)')] = 0.0

for pdbid in tqdm(DK_bound['sample']):
    zmap = gemmi.read_ccp4_map(f'/n/hekstra_lab/people/minhuan/projects/drug/minhuan_backup/pipeline/data/z_maps/PTP1B-y{pdbid}-z_map.native.ccp4')
    st = gemmi.read_structure(f'/n/hekstra_lab/people/minhuan/projects/drug/minhuan_backup/pipeline/data/bound_models_DK/PTP1B-y{pdbid}_refmac_input.split.bound-state.pdb')

    real_grid = zmap.grid
    real_grid.normalize()

    sel = gemmi.Selection('[CL,Br,S,I]')
    sel_model = sel.copy_model_selection(st[0])
    lig_heavy_atoms = [i for i in list(sel_model.all()) if i.residue.name == 'LIG']

    dis_lists = []
    peak_values = []
    for cra in lig_heavy_atoms:

        eq_points = []
        ops = real_grid.spacegroup.operations()
        atom = cra.atom

        a,b,c = np.unravel_index(real_grid.array.argmax(), real_grid.array.shape)
        tmp = real_grid.get_fractional(a,b,c)
        peak_pos = st.cell.orthogonalize(gemmi.Fractional(tmp.z, tmp.y, tmp.x))
        dis_list = []

        for op in ops:
            SG_mapped=op.apply_to_xyz(st.cell.fractionalize(atom.pos).tolist())
            tmp = SG_mapped-np.floor(np.array(SG_mapped))
            SG_mapped = gemmi.Fractional(*tmp)
            eq_points.append(SG_mapped)
            SG_mapped_orth = st.cell.orthogonalize(SG_mapped)
            dis_list.append(np.sqrt(np.sum(np.array((peak_pos - SG_mapped_orth).tolist())**2)))

        peak_value = []
        for pos in eq_points:
            a = round(pos.z * real_grid.nu)
            b = round(pos.y * real_grid.nv)
            c = round(pos.x * real_grid.nw)
            peak_value.append(real_grid.get_value(a, b, c))

        dis_lists.append(dis_list)
        peak_values.append(peak_value)

    log_peak = np.max(peak_values)
    log_ismaxpeak = np.any(np.array(dis_lists) < 5.0)

    DK_bound.loc[DK_bound['sample']==pdbid, 'lig_heavy_peak'] = log_peak
    DK_bound.loc[DK_bound['sample']==pdbid, 'is_highest_peak(<5A)'] = log_ismaxpeak

print(DK_bound['lig_heavy_peak'])
print(np.mean(DK_bound['lig_heavy_peak']))
