from pymopt.modeling import TuringPattern
from pymopt.voxel import VoxelTuringModel
from pymopt.utils import generate_variable_params
from pymopt.optics import OBD

import datetime,time
import os,gc
import numpy as np
import pandas as pa
from multiprocessing import Pool


repetitions = 500
pool_num = 8
nPh = 1e7
iteral_num=np.arange(repetitions)

range_params_norm = {
    'th_dermis':[1,2],             # 皮膚厚さの範囲
    'ma_dermis':[0.00633,0.08560], # 皮膚吸収係数の範囲
    'msd_dermis':[1.420,2.506],    # 皮膚減衰散乱係数の範囲
    'th_subcutaneus':[1,6],        # 皮下組織厚さの範囲
    'ma_subcutaneus':[0.00485,0.01239],# 皮下組織吸収係数の範囲
    'msd_subcutaneus':[0.83,1.396],# 皮下組織減衰散乱係数の範囲
    'bv_tv':[0.134,0.028],          # 海綿骨BV/TVの平均と分散
    'th_cortical':[0.804, 0.149],  # 皮質骨厚さの平均と分散
    'corr':0.54,                   # 皮質骨と海綿骨の相関係数 Boutry2005
}
range_params_osteoporosis = range_params_norm.copy()
range_params_osteoporosis['bv_tv'] = [0.085,0.022]
range_params_osteoporosis['th_cortical'] = [0.487,0.138]
range_params_osteopenia = range_params_norm.copy()
range_params_osteopenia['bv_tv'] = [0.103,0.030]
range_params_osteopenia['th_cortical'] = [0.571,0.173]
def determining_params_range():
    a = np.random.rand()
    range_params = 0
    if a <= 0.4:
        print('Osteoporosis')
        range_params = range_params_osteoporosis
    elif a >= 0.6:
        print('Normal')
        range_params = range_params_norm
    else:
        print('Osteoprnia')
        range_params = range_params_osteopenia
    return range_params

model_params ={
    'grid':40,
    'dx':1/40,
    'dt':1,
    'du':0.0002,
    'dv':0.01,
    'length':10,
    'repetition':100,
    'voxelsize':0.0245,
    'seed':False,
    'ct_coef':4.5e4,
    'tile_num_xz':2,
    'tile_num_y':4,
}

th_coef = np.array([-7.6618293,1.64450117,-0.45237661,0.60426539])


monte_params = {
    'voxel_space':model_params['voxelsize'],
    'r_bone':9.17,

    'n_space':1.4,
    'n_trabecular':1.55,
    'n_cortical':1.55,
    'n_subcutaneus':1.4,
    'n_dermis':1.4,
    'n_air':1.,

    'ma_space':0.00862,
    'ma_trabecular':0.02374,
    'ma_cortical':0.02374,
    'ma_air':1e-5,

    'ms_space':11.13,
    'ms_trabecular':20.588,
    'ms_cortical':20.588,
    'ms_air':1e-5,

    'g_space':0.90,
    'g_trabecular':0.90,
    'g_cortical':0.90,
    'g_subcutaneus':0.90,
    'g_dermis':0.90,
    'g_air':.90,
}


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
        model_name = 'TuringModel_cylinder'
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


def calc(iteral):
    gvp = generate_variable_params()
    range_params = determining_params_range()

    gvp.set_params(range_params)
    vp = gvp.generate(1)

    if vp['bv_tv'][0] > 0 and vp['th_cortical'][0] > 0:
        alias_name = "-".join((str(datetime.datetime.now().isoformat()).split('.')[0]).split(':'))+'_it'+f'{iteral:04}'


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

        print('')
        print('############### End %s it ###################'%iteral)
        print('')
    else:
        print("Invalid parameter was generated")
        print("BV/TV : ",vp['bv_tv'][0])
        print("th_cortical : ",vp['th_cortical'][0])



if __name__ == "__main__":

    p = Pool(pool_num)
    p.map(calc, iteral_num)

    print()
    print('######################')
    print(datetime.datetime.now())
