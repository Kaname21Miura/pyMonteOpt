from pymopt.modeling_gpu import TuringPattern
from pymopt.voxel_gpu import VoxelTuringModel
from pymopt.utils import generate_variable_params
from pymopt import metrics as met

import datetime,time
import os,gc
import numpy as np
import pandas as pa

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks", {'grid.linestyle': '--'})
import pandas as pd


import datetime,time
import os,gc
import numpy as np
import pandas as pa
from multiprocessing import Pool


repetitions = 10000
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
    'grid':30,
    'dx':1/30,
    'dt':1,
    'du':0.0002,
    'dv':0.01,
    'length':10,
    'repetition':100,
    'voxelsize':0.0306,
    'seed':False,
    'ct_coef':4.5e4,
    'tile_num_xz':2,
    'tile_num_y':4,
}


#th_coef = np.array([-7.6618293,1.64450117,-0.45237661,0.60426539])
th_coef = np.array([-10.93021385,   2.62630274,  -0.50913966,   0.60371039])


monte_params = {
    'voxel_space':model_params['voxelsize'],
    'r_bone':9.,

    'n_space':1.4,
    'n_trabecular':1.55,
    'n_cortical':1.55,
    'n_subcutaneus':1.4,
    'n_dermis':1.4,
    'n_air':1.,

    'ma_space':0.00862,
    'ma_trabecular':0.02374,
    'ma_cortical':0.02374,
    'ma_air':1e-6,

    'ms_space':11.13,
    'ms_trabecular':20.588,
    'ms_cortical':20.588,
    'ms_air':1e-6,

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
    #print(params)
    model = VoxelTuringModel(
        nPh = nPh,
        model_name = 'TuringModel_cylinder'
    )
    model.set_model(u)

    model.build(**params)
    start = time.time()
    model = model.start(iteral)
    print('%s sec'%(time.time()-start))
    print('# %s'%iteral)
    print("Save -> %s"%path)
    model.save_result(path,coment='for machine learning')
    res = model.get_result()

    del model
    gc.collect()
    return res,params

def calc_ray_tracing(res,monte_params,path,alias_name):
    l = monte_params["r_bone"]*2+monte_params["th_subcutaneus"]*2+monte_params["th_dermis"]*2
    nn = 300
    dr = 30/nn
    nn_ = 400
    dr_ = 40/nn_
    margin = 1e-8
    ind = np.where((res["v"][2]<0)&(res["p"][2]<margin))[0]
    alphaRd,Rd = met.radialDistance(res["p"][:,ind],res["w"][ind],nn,dr,res["nPh"])
    
    ind = np.where(res["v"][2]>0&(res["p"][2]>l-margin))[0]
    alphaTt,Tt = met.radialDistance(res["p"][:,ind],res["w"][ind],nn,dr,res["nPh"])

    ind = np.where((res["v"][0]<0))[0]
    alpha_ssyz,Ssyz = met.lineDistance(res["p"][:,ind],res["w"][ind],nn_,dr_,res["nPh"],y_range=5)

    print('# Ray Tracing save -> %s'%path)
    
    path_ = path+"_B"
    aa = alias_name+"_B"
    df = pd.DataFrame()
    df[aa] = Rd
    df.index = alphaRd
    df.to_csv(path_+".csv")

    path_ = path+"_F"
    aa = alias_name+"_F"
    df = pd.DataFrame()
    df[aa] = Tt
    df.index = alphaTt
    df.to_csv(path_+".csv")
    
    path_ = path+"_L"
    aa = alias_name+"_L"
    df = pd.DataFrame()
    df[aa] = Ssyz
    df.index = alpha_ssyz
    df.to_csv(path_+".csv")
    


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
        res,params_ = calc_montecalro(vp,iteral,monte_params,path_,u)
        print('###### end monte calro in it: ',iteral)
        
        path_ = opt_path+alias_name
        calc_ray_tracing(res,params_,path_,alias_name)
        print('')
        print('############### End %s it ###################'%iteral)
        print('')
    else:
        print("Invalid parameter was generated")
        print("BV/TV : ",vp['bv_tv'][0])
        print("th_cortical : ",vp['th_cortical'][0])



if __name__ == "__main__":

    for iteral in range(repetitions):
        calc(iteral)

    print()
    print('######################')
    print(datetime.datetime.now())
