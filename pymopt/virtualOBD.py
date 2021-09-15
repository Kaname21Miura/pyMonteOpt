from pymopt.modeling import TuringPattern
from pymopt.voxel import VoxelTuringModel
from pymopt.utils import generate_variable_params
from pymopt.optics import OBD

import datetime,time
import os,gc
import numpy as np
import pandas as pa
from multiprocessing import Pool


repetitions = 130
pool_num = 8
nPh = 1e7
iteral_num=np.arange(repetitions)

range_params = {
    'th_dermis':[1,2],             # 皮膚厚さの範囲
    'ma_dermis':[0.00633,0.08560], # 皮膚吸収係数の範囲
    'msd_dermis':[1.420,2.506],    # 皮膚減衰散乱係数の範囲
    'th_subcutaneus':[1,6],        # 皮下組織厚さの範囲
    'ma_subcutaneus':[0.00485,0.01239],# 皮下組織吸収係数の範囲
    'msd_subcutaneus':[0.83,1.396],# 皮下組織減衰散乱係数の範囲
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
    'length':9,
    'repetition':100,
    'voxelsize':0.0245,
    'seed':False,
    'ct_coef':4.5e4,
    'tile_num':2,
}

th_coef = np.array([-7.61194835,1.62003258,-0.44989454,0.60428882])

monte_params = {
    'voxel_space':model_params['voxelsize'],
    'symmetrization':True,
    'enclosure':True,

    'n_space':1.4,
    'n_trabecular':1.55,
    'n_cortical':1.55,
    'n_subcutaneus':1.4,
    'n_dermis':1.4,
    'n_air':1.,

    'ma_space':0.00862,
    'ma_trabecular':0.02374,
    'ma_cortical':0.02374,
    'ma_subcutaneus':0.011,
    'ma_dermis':0.05925,

    'ms_space':11.13,
    'ms_trabecular':20.588,
    'ms_cortical':20.588,
    'ms_subcutaneus':20,
    'ms_dermis':19.63,

    'g_space':0.90,
    'g_trabecular':0.90,
    'g_cortical':0.90,
    'g_subcutaneus':0.90,
    'g_dermis':0.90,
}

opt_params ={
    'start':15,
    'end':85,
    'split':0.5,

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
    'inversion':False,
    'side':False,
}

opt_params_inv = opt_params.copy()
opt_params_inv['inversion']=True
opt_params_inv['side']=False
opt_params_inv['start']=0
opt_params_side = opt_params.copy()
opt_params_side['inversion']=False
opt_params_side['side']=True
opt_params_side['start']=0

def generate_bone_model(bv_tv,path,model_params):
    model_params['bv_tv']=bv_tv
    tp = TuringPattern()
    tp.set_params(model_params)
    tp.set_threshold_func_coef(th_coef)
    if not os.path.exists(path):
        os.makedirs(path)
    u = tp.modeling(path,save_dicom=False)
    bvtv_ = tp.bv_tv_real
    del tp
    gc.collect()
    return u,bvtv_

def calc_montecalro(vp,iteral,params,path,u):
    print()
    print('###################################')
    print('# %s'%iteral)

    lamda = 850
    deg = 0
    nn = 0
    params['th_dermis'] = vp['th_dermis'][nn]
    params['ma_dermis'] = vp['ma_dermis'][nn]
    params['ms_dermis'] = vp['msd_dermis'][nn]/(1-params['g_dermis'])

    params['th_subcutaneus'] = vp['th_subcutaneus'][nn]
    params['ma_subcutaneus'] = vp['ma_subcutaneus'][nn]
    params['ms_subcutaneus'] = vp['msd_subcutaneus'][nn]/(1-params['g_subcutaneus'])

    params['ma_space'] = vp['ma_subcutaneus'][nn]
    params['ms_space'] = vp['msd_subcutaneus'][nn]/(1-params['g_space'])

    params['bv_tv'] = vp['bv_tv'][nn]
    params['th_cortical'] = vp['th_cortical'][nn]

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
    gvp = generate_variable_params()
    gvp.set_params(range_params)
    vp = gvp.generate(1)

    print('### iteral number ',iteral)
    print('Alias name: ',alias_name)
    model_path = './model_result/'
    monte_path = './monte_result/'
    opt_path = './opt_result/'

    path_ = model_path+alias_name+'_dicom'
    u,bv_tv = generate_bone_model(vp['bv_tv'][0],path_,model_params)
    print('it: ',iteral,', change bvtv: ',vp['bv_tv'][0],'-->',bv_tv)
    vp['bv_tv'][0] = bv_tv
    path_ = monte_path+alias_name
    res = calc_montecalro(vp,iteral,monte_params,path_,u)
    print('###### end monte calro in it: ',iteral)

    path_ = opt_path+alias_name
    calc_ray_tracing(res,opt_params,path_)

    path_ = opt_path+alias_name+'_inv'
    calc_ray_tracing(res,opt_params_inv,path_)

    path_ = opt_path+alias_name+'_side'
    thickness = (vp['th_subcutaneus'][0]+vp['th_dermis'][0]+\
                vp['th_cortical'][0])/2+\
                model_params['grid']*model_params['length']*\
                model_params['tile_num']*model_params['voxelsize']
    opt_params_side['side']=thickness
    calc_ray_tracing(res,opt_params_side,path_)
    print('')
    print('############### End %s it ###################'%iteral)
    print('')


if __name__ == "__main__":

    p = Pool(pool_num)
    p.map(calc, iteral_num)

    print()
    print('######################')
    print(datetime.datetime.now())
