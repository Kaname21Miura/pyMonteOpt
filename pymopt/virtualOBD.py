from pymopt.modeling import TuringPattern
from pymopt.voxel import VoxelTuringModel
from pymopt.utils import generate_variable_params
from pymopt.optics import OBD

import datetime,time
import os,gc
import numpy as np
import pandas as pa
from multiprocessing import Pool


repetitions = 8
pool_num = 8
nPh = 1e7

range_params = {
    'th_dermis':[1,2],             # 皮膚厚さの範囲
    'ma_dermis':[0.00633,0.08560], # 皮膚吸収係数の範囲
    'msd_dermis':[1.420,2.506],    # 皮膚減衰散乱係数の範囲
    'th_subcutaneus':[1,6],        # 皮下組織厚さの範囲
    'ma_subcutaneus':[0.005,0.012],# 皮下組織吸収係数の範囲
    'msd_subcutaneus':[0.83,1.396],# 皮下組織減衰散乱係数の範囲
    'ma_marrow':[0.005,0.012],     # 骨髄吸収係数の範囲
    'msd_marrow':[0.83,1.396],     # 骨髄減衰散乱係数の範囲
    'bv_tv':[0.115,0.02],          # 海綿骨BV/TVの平均と分散
    'th_cortical':[0.669, 0.133],  # 皮質骨厚さの平均と分散
    'corr':0.54,                   # 皮質骨と海綿骨の相関係数 Boutry2005
}

model_params ={
    'grid':40,
    'dx':1/40,
    'dt':1,
    'du':0.0002,
    'dv':0.01,
    'length':13,
    'repetition':150,
    'voxelsize':0.024,
    'seed':False,
    'ct_coef':4.5e4,
    'tile_num':2,
}

monte_params = {
    'voxel_space':model_params['voxelsize'],
    'n_space':1.4,
    'n_trabecular':1.55,
    'n_cortical':1.55,
    'n_subcutaneus':1.4,
    'n_dermis':1.4,
    'n_air':1.,

    'ma_space':0.00862,
    'ma_trabecular':0.02374,
    'ma_cortical':0.02374,
    #'ma_subcutaneus':0.011,'ma_dermis':0.05925,

    'ms_space':11.13,
    'ms_trabecular':20.588,
    'ms_cortical':20.588,
    #'ms_subcutaneus':20,'ms_dermis':19.63,

    'g_space':0.90,
    'g_trabecular':0.90,
    'g_cortical':0.90,
    'g_subcutaneus':0.90,
    'g_dermis':.90,
}

opt_params ={
    'start':15,
    'end':90,
    'split':0.25,

    'wavelength':850,

    'outerD_1':50,
    'efl_1':100,
    'bfl_1':93.41,
    'ct_1':10,
    'et_1':3.553,
    'r_1':51.68,
    'substrate_1':'N-BK7',

    'outerD_2' : 50,
    'efl_2' : 50,
    'bfl_2' : 43.28,
    'ct_2':12,
    'et_2':3.01,
    'r_2':39.24,
    'substrate_2':'N-SF11',

    'slit_outerD':50,
    'slit_D':20,
    'slit_width':2,
    'slit_thickness':3,
    'd_pd':3,

    'distance_2slits':32,
    'pd_poit_correction':0.22,
    'ld_fix_part':False,
}


def get_variable_params(n,range_params):
    gvp = generate_variable_params()
    gvp.set_params(range_params)
    gvp.generate(n)
    return gvp.get_variable_params()

def generate_bone_model(bv_tv,path,model_params):
    model_params['bv_tv']=bv_tv
    tp = TuringPattern()
    tp.set_params(model_params)
    if not os.path.exists(path):
        os.makedirs(path)
    u = tp.modeling(path,save_dicom=True)

    del tp
    gc.collect()
    return u

def calc_montecalro(vp,iteral,params,path,u):
    print()
    print('###################################')
    print('# %s'%iteral)

    lamda = 850
    deg = 0

    params['th_dermis'] = vp['th_dermis'][iteral]
    params['ma_dermis'] = vp['ma_dermis'][iteral]
    params['ms_dermis'] = (1-params['g_dermis'])*vp['msd_dermis'][iteral]

    params['th_subcutaneus'] = vp['th_subcutaneus'][iteral]
    params['ma_subcutaneus'] = vp['ma_subcutaneus'][iteral]
    params['ms_subcutaneus'] = (1-params['g_subcutaneus'])*vp['msd_subcutaneus'][iteral]

    params['ma_space'] = vp['ma_marrow'][iteral]
    params['ms_space'] = (1-params['g_space'])*vp['msd_marrow'][iteral]

    params['bv_tv'] = vp['bv_tv'][iteral]
    params['th_cortical'] = vp['th_cortical'][iteral]

    model = VoxelTuringModel(
        nPh = nPh,
        z_max_mode = False,
        beam_angle = deg*np.pi/180,
        wavelength = lamda,
        beam_posision = 10,
        lens_curvature_radius = 51.68,
        grass_type = 'N-BK7',
        initial_refrect_by_angle = True,
    )
    model.set_model(u)

    model.build(**params)
    start = time.time()
    model = model.start()
    print('%s sec'%(time.time()-start))
    print('# %s'%iteral)
    print("Save -> %s"%path)
    model.save_result(path,coment='for machine learning')
    res = model.get_result()

    del model
    gc.collect()
    return res

def calc_ray_tracing(res,opt_params,path):
    obd = OBD()
    obd.set_monte_data(res)
    obd.set_params(opt_params)
    obd.start()

    print('# Ray Tracing save -> %s'%path)
    obd.save_result(path)
    del obd
    gc.collect()


def calc(iteral):
    alias_name = "-".join((str(datetime.datetime.now().isoformat()).split('.')[0]).split(':'))+'_it'+f'{iteral:04}'

    print('')
    print('### iteral number ',iteral)
    print('Alias name: ',alias_name)
    model_path = './model_result/'
    monte_path = './monte_result/'
    opt_path = './opt_result/'

    path_ = model_path+alias_name+'_dicom'
    u = generate_bone_model(vp['bv_tv'][iteral],path_,model_params)

    path_ = monte_path+alias_name
    res = calc_montecalro(vp,iteral,monte_params,path_,u)

    path_ = opt_path+alias_name
    calc_ray_tracing(res,opt_params,path_)
    print('')
    print('############### End %s it ###################'%iteral)
    print('')

vp = get_variable_params(repetitions,range_params)
iteral_num=np.arange(repetitions)
pa.DataFrame(vp).to_csv('variable_params.csv')

if __name__ == "__main__":

    p = Pool(pool_num)
    p.map(calc, iteral_num)

    print()
    print('######################')
    print(datetime.datetime.now())
