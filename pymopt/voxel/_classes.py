#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from ._kernel import vmc_kernel

import numpy as np
import os
import pydicom
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from abc import ABCMeta, abstractmethod

import datetime,time
import json,pickle,bz2

from ..utils import readDicom,reConstArray_8,reConstArray
from ..utils.validation import _deprecate_positional_args
from ..fluence import Fluence2D,Fluence3D
from ..utils.utilities import calTime,set_params,ToJsonEncoder
from ..optics._classes import Grass
import gc
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
#os.system("taskset -p 0xff %d" % os.getpid())

__all__ = [
'VoxelPlateModel','VoxelTuringModel'
]

# =============================================================================
# Base solid model
# =============================================================================

class BaseVoxelMonteCarlo(metaclass = ABCMeta):
    #@_deprecate_positional_args
    @abstractmethod
    def __init__(self,*,nPh,model,dtype_f=np.float32,dtype=np.int32,
                 beam_type = 'TEM00',w_beam = 0,
                 beam_angle = 0,initial_refrect_by_angle = False,
                 first_layer_clear = False,
                 ):

        def __check_list_name(name,name_list):
            if not(name in name_list):
                raise ValueError('%s is not a permitted for factor. Please choose from %s.'%(name,name_list))

        self.beam_type_list=['TEM00',False]
        __check_list_name(beam_type,self.beam_type_list)
        self.beam_type = beam_type

        self.dtype = dtype
        self.dtype_f = dtype_f
        self.nPh = nPh
        self.w_beam = w_beam

        self.initial_refrect_by_angle = initial_refrect_by_angle
        self.beam_angle = beam_angle

        self.model = model
        self.first_layer_clear=first_layer_clear

    def start(self):
        self.nPh = int(self.nPh)
        self._reset_results()
        self._generate_initial_coodinate(self.nPh)

        self.add = self.add.astype(np.int32)
        self.p = self.p.astype(np.float32)
        self.v = self.v.astype(np.float32)
        self.w = self.w.astype(np.float32)
        print("")
        print("###### Start ######")
        print("")
        start_ = time.time()
        self.add,self.p,self.v,self.w = vmc_kernel(
            self.add, self.p,self.v, self.w,
            self.model.ma, self.model.ms, self.model.n, self.model.g,
            self.model.voxel_model, self.model.voxel_space,
            np.int32(self.nPh), np.int8(self.model.end_point)
        )

        self._end_process()
        print("###### End ######")
        self.getRdTtRate()
        calTime(time.time(), start_)
        #del func
        return self

    def _end_process(self):#書き換え
        #index = np.where(~np.isnan(self.w))[0]
        self.v_result = self.v#[:,index]
        self.p_result = self.p#[:,index]
        self.add_result = self.add#[:,index]
        self.w_result = self.w#[index]

    def _reset_results(self):
        self.v_result = np.empty((3,1)).astype(self.dtype_f)
        self.p_result = np.empty((3,1)).astype(self.dtype_f)
        self.add_result = np.empty((3,1)).astype(self.dtype)
        self.w_result = np.empty(1).astype(self.dtype_f)
        return self

    def get_voxel_model(self):
        return self.model.voxel_model

    def _generate_initial_coodinate(self,nPh):
        self._set_inital_add()
        self._set_beam_distribution()
        self._set_inital_vector()
        self._set_inital_w()


    def _set_inital_add(self):
        if self.beam_type == 'TEM00':
            self.add =  np.zeros((3, self.nPh),dtype = self.dtype)
        self.add[0] = self._get_center_add(self.model.voxel_model.shape[0])
        self.add[1] = self._get_center_add(self.model.voxel_model.shape[1])
        if self.first_layer_clear:
            self.add[2] = self.model.get_second_layer_addz()
        else:
            self.add[2] = 1

    def _get_center_add(self,length):
        #addの中心がローカル座標（ボックス内）の中心となるため、
        #ボクセル数が偶数の時は、1/2小さい位置を中心とし光を照射し、
        #逆変換時（_encooder）も同様に1/2小さい位置を中心として元のマクロな座標に戻す。
        return int((length-1)/2)

    def _set_inital_vector(self):
        if self.beam_type == 'TEM00':
            self.v = np.zeros((3,self.nPh)).astype(self.dtype_f)
            self.v[2] = 1
            if self.beam_angle!=0 and self.w_beam==0:
                #ビーム径がある場合はとりあえず無視
                #角度はrad表記
                ni = self.model.n[-1]
                nt = self.model.n[0]
                ai = self.beam_angle
                at = np.arcsin(np.sin(ai)*ni/nt)
                self.v[0] = np.sin(at)
                self.v[2] = np.cos(at)
                if self.initial_refrect_by_angle:
                    Ra = ((np.sin(ai-at)/np.sin(ai+at))**2\
                    +(np.tan(ai-at)/np.tan(ai+at))**2)/2

                    self.inital_del_num = np.count_nonzero(Ra>=np.random.rand(self.nPh))
                    self.v = np.delete(self.v, np.arange(self.inital_del_num), 1)
                    self.p = np.delete(self.p, np.arange(self.inital_del_num), 1)
                    self.add = np.delete(self.add, np.arange(self.inital_del_num), 1)
                    sub_v = np.zeros((3,self.inital_del_num)).astype(self.dtype_f)
                    sub_v[0] = np.sin(ai)
                    sub_v[2] = -np.cos(ai)
                    self.v_result = np.concatenate([self.v_result,
                    sub_v],axis = 1)
                    self.p_result = np.concatenate([self.p_result,
                    self.p[:,:self.inital_del_num]],axis = 1)
                    self.add_result = np.concatenate([self.add_result,
                    self.add[:,:self.inital_del_num]],axis = 1)
        else:
            print("ビームタイプが設定されていません")

    def _set_inital_w(self):
        if self.beam_type == 'TEM00':
            self.w = np.ones(self.nPh).astype(self.dtype_f)
            Rsp = 0
            n1 = self.model.n[-1]
            n2 = self.model.n[0]
            if n1 != n2:
                Rsp = ((n1-n2)/(n1+n2))**2
                if self.beam_angle!=0 and self.w_beam==0:
                    ai = self.beam_angle
                    at = np.arcsin(np.sin(ai)*n1/n2)
                    Rsp = ((np.sin(ai-at)/np.sin(ai+at))**2\
                    +(np.tan(ai-at)/np.tan(ai+at))**2)/2
                elif self.first_layer_clear:
                    n3=self.model.n[1]
                    r2 = ((n3-n2)/(n3+n2))**2
                    Rsp = Rsp+r2*(1-Rsp)**2/(1-Rsp*r2)
                self.w -= Rsp

            if self.beam_angle!=0 and self.w_beam==0:
                if self.initial_refrect_by_angle:
                    self.w[:] = 1
                    self.w = np.delete(self.w, np.arange(self.inital_del_num), 0)
                    self.w_result = np.concatenate([self.w_result,
                    self.w[:self.inital_del_num]],axis = 0)
        else:
            print("ビームタイプが設定されていません")

    def _set_beam_distribution(self):
        if self.beam_type == 'TEM00':
            self.p = np.zeros((3,self.nPh)).astype(self.dtype_f)
            self.p[2] = -self.model.voxel_space/2
            if self.w_beam!= 0:
                print("%sを入力"%self.beam_type)
                #ガウシアン分布を生成
                gb = np.array(self.gaussianBeam(self.w_beam)).astype(self.dtype_f)
                #ガウシアン分布を各アドレスに振り分ける

                l = self.model.voxel_space
                pp = (gb/l).astype("int16")
                ind = np.where(gb<0)
                pp[ind[0].tolist(),ind[1].tolist()] -= 1
                pa = gb - (pp+1/2)*l
                ind = np.where((np.abs(pa)>=l/2))
                pa[ind[0].tolist(),ind[1].tolist()] = \
                    np.sign(pa[ind[0].tolist(),ind[1].tolist()])*(l/2)
                pa += l/2
                self.add[:2] = self.add[:2] + pp
                self.p[:2] = pa.astype(self.dtype_f)
        else:
            print("ビームタイプが設定されていません")

    def _get_beam_dist(self,x,y):
        fig = plt.figure(figsize=(10,6),dpi=70)
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        H = ax.hist2d(x,y, bins=100,cmap="plasma")
        ax.set_title('Histogram for laser light intensity')
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        fig.colorbar(H[3],ax=ax)
        plt.show()

    def gaussianBeam(self,w=0.54):
        #TEM00のビームを生成します
        r = np.linspace(-w*2,w*2,100)
        #Ir = 2*np.exp(-2*r**2/(w**2))/(np.pi*(w**2))
        Ir = np.exp(-2*r**2/(w**2))
        normd = stats.norm(0, w/2)
        x = normd.rvs(self.nPh)
        y = normd.rvs(self.nPh)
        #z = np.zeros(self.nPh)

        fig, ax1 = plt.subplots()
        ax1.set_title('Input laser light distribution')
        ax1.hist(x, bins=100, color="C0")
        ax1.set_ylabel('Number of photon')
        ax2 = ax1.twinx()
        ax2.plot(r, Ir, color="k")
        ax2.set_xlabel('X [mm]')
        ax2.set_ylabel('Probability density')
        plt.show()
        self._get_beam_dist(x,y)
        return x,y

    def get_result(self):
        encoded_position = self._encooder(self.p_result,self.add_result)
        df_result = {
            'p':encoded_position,
            'v':self.v_result,
            'w':self.w_result,
            'nPh':self.nPh
        }
        return df_result

    def get_model_params(self):
        return self.model.get_params()

    def _encooder(self,p,add):
        space = self.model.voxel_space
        center_add_x = self._get_center_add(self.model.voxel_model.shape[0])
        center_add_y = self._get_center_add(self.model.voxel_model.shape[1])
        encoded_position = p.copy()
        encoded_position[0] = space*(add[0]-center_add_x)+p[0]
        encoded_position[1] = space*(add[1]-center_add_y)+p[1]
        encoded_position[2] = np.round(space*(add[2]-1)+p[2]+space/2,6)
        return encoded_position

    def set_monte_params(self,*,nPh,model, dtype_f=np.float32, dtype=np.int32,w_beam = 0):
        self.dtype_f = dtype_f
        self.dtype = dtype
        self.nPh = nPh
        self.w_beam = w_beam
        self.model = model

    def build(self,*initial_data, **kwargs):
        if initial_data == () and kwargs == {}:
            pass
        else:
            self.model.set_params(*initial_data, **kwargs)
        self.model.build()

    def getRdTtRate(self):
        self.Tt_index = np.where(self.v_result[2]>0)[0]
        self.Rd_index = np.where(self.v_result[2]<0)[0]
        self.Rdw = self.w_result[self.Rd_index].sum()/self.nPh
        self.Ttw = self.w_result[self.Tt_index].sum()/self.nPh
        print('######')
        print('Mean Rd %0.6f'% self.Rdw)
        print('Mean Td %0.6f'% self.Ttw)
        print()

    def save_result(self,fname,
    *,coment='',save_monte = True,save_params = True,):
        start_ = time.time()

        if save_monte:
            res = self.get_result()
            save_name = fname+"_LID.pkl.bz2"
            with bz2.open(save_name, 'wb') as fp:
                fp.write(pickle.dumps(res))
            print("Monte Carlo results saved in ")
            print("-> %s" %(save_name))
            print('')

        if save_params :
            info = self._calc_info(coment)
            save_name = fname+"_info.json"
            with open(save_name, 'w') as fp:
                json.dump(info,fp,indent=4,cls= ToJsonEncoder)
            print("Calculation conditions are saved in")
            print("-> %s" %(save_name))
            print('')

        calTime(time.time(), start_)

    def _calc_info(self,coment=''):
        _params = self.model.get_params()
        calc_info = {
            'Date':datetime.datetime.now().isoformat(),
            'coment':coment,
            'number_of_photons':self.nPh,
            'calc_dtype':"32 bit",
            'model':{
                'model_name':self.model.model_name,
                'model_params':_params,
                'model_voxel_space':self.model.voxel_space,
                'model_xy_size':self.model.xy_size,
            },
            'w_beam':self.w_beam,
            'beam_angle':self.beam_angle,
            'initial_refrect_mode':self.initial_refrect_by_angle,
            'beam_mode':'TEM00',
            'fluence_mode':self.fluence_mode,
        }
        return calc_info
# =============================================================================
# Modeling class
# =============================================================================

class VoxelModel:
    def build(self):
        pass
    def set_params(self):
        pass

    def getModelSize(self):
        print("Memory area size for voxel storage: %0.3f Mbyte" % (self.voxel_model.nbytes*1e-6))

class PlateModel(VoxelModel):
    @_deprecate_positional_args
    def __init__(self):
        self.model_name = 'PlateModel'
        self.dtype_f = np.float32
        self.dtype = np.int8
        self.params = {
            'x_size':40,'y_size':40,#X-Yはintralipidの領域を示す
            'voxel_space':0.1,
            'thickness':[1.3,40],
            'n':[1.5,1.4],
            'n_air':1.,
            'ma':[1e-5,0.02374],
            'ms':[1e-5,0.02374],
            'g':[0.9,0.9],
            }
        self.keys = list(self.params.keys())
        self._param_instantiating()
        self.voxel_model = np.zeros((3,3,3),dtype = self.dtype)

    def _param_instantiating(self):
        f = self.dtype_f
        self.thickness = self.params['thickness']
        self.n =np.array(self.params['n']+[self.params['n_air']]).astype(f)
        self.ms = np.array(self.params['ms']).astype(f)
        self.ma = np.array(self.params['ma']).astype(f)
        self.g = np.array(self.params['g']).astype(f)
        self.voxel_space = self.params['voxel_space']
        self.x_size = np.int32(round(self.params['x_size']/self.params['voxel_space']))
        self.y_size = np.int32(round(self.params['y_size']/self.params['voxel_space']))
        self.z_size = np.int32(round(np.array(self.thickness).sum()/self.params['voxel_space']))

    def build(self):
        #thickness,xy_size,voxel_space,ma,ms,g,n,n_air
        del self.voxel_model
        gc.collect()
        self._make_voxel_model()
        self.getModelSize()

    def set_params(self,*initial_data, **kwargs):
        set_params(self.params,self.keys,*initial_data, **kwargs)
        self._param_instantiating()

    def _make_voxel_model(self):

        self.voxel_model = np.empty((
            self.x_size+2,
            self.y_size+2,
            self.z_size+2
        )).astype(self.dtype)

        val = 1
        for n_,i in enumerate(self.thickness):
            val_ = round(i/self.voxel_space)
            self.voxel_model[:,:,val:val_+val] = np.int8(n_)
            val+=val_

        self.end_point = np.int8(np.array(self.thickness).size)
        self.voxel_model[0,:,:] = self.end_point
        self.voxel_model[-1,:,:] = self.end_point
        self.voxel_model[:,0,:] = self.end_point
        self.voxel_model[:,-1,:] = self.end_point
        self.voxel_model[:,:,0] = self.end_point
        self.voxel_model[:,:,-1] = self.end_point

    def get_params(self):
        return {
                'th':self.thickness,
                'ms':self.ms,
                'ma':self.ma,
                'n':self.n,
                'g':self.g
                }

class PlateExModel(VoxelModel):
    #ガラスとイントラリピッドの２層構造のみを対象とする
    #ガラスはイントラリピッドを取り囲んでいるものとする
    def __init__(self):
        self.model_name = 'PlateExModel'
        self.dtype = 'int8'
        self.dtype_f = 'float32'
        self.grass_num=0
        self.intra_num=1
        self.end_point = 2
        self.params = {
            'x_size':40,'y_size':40,#X-Yはintralipidの領域を示す
            'voxel_space':0.1,
            'thickness':[1.3,40],
            'n':[1.5,1.4],
            'n_air':1.,
            'ma':[1e-5,0.02374],
            'ms':[1e-5,0.02374],
            'g':[0.9,0.9],
            }
        self.keys = list(self.params.keys())
        self._param_instantiating()
        self.voxel_model = np.zeros((3,3,3),dtype = self.dtype)
        self.model_shape = (3,3,3)

    def _param_instantiating(self):
        f = self.dtype_f
        self.thickness = self.params['thickness']
        self.n =np.array(self.params['n']+[self.params['n_air']]).astype(f)
        self.ms = np.array(self.params['ms']).astype(f)
        self.ma = np.array(self.params['ma']).astype(f)
        self.g = np.array(self.params['g']).astype(f)
        self.voxel_space = self.params['voxel_space']
        self.x_size = int(self.params['x_size']/self.params['voxel_space'])
        self.y_size = int(self.params['y_size']/self.params['voxel_space'])
        self.z_size = int(self.params['thickness'][1]/self.params['voxel_space'])

    def build(self):
        #thickness,xy_size,voxel_space,ma,ms,g,n,n_air
        del self.voxel_model
        gc.collect()
        self._make_voxel_model()
        self.getModelSize()

    def set_params(self,*initial_data, **kwargs):
        set_params(self.params,self.keys,*initial_data, **kwargs)
        self._param_instantiating()

    def _make_voxel_model(self):
        #まずイントラリピッドを定義し、その後、ガラス面を定義する
        self.voxel_model = np.ones((self.x_size,self.y_size,self.z_size),dtype = self.dtype)
        self.num_pix = int(self.params['thickness'][0]/self.params['voxel_space'])

        #z方向の追加
        ct = np.ones((
        self.voxel_model.shape[0],self.voxel_model.shape[1],self.num_pix+1),
        dtype = self.dtype)*self.grass_num
        self.voxel_model = np.concatenate((ct,self.voxel_model),2)
        self.voxel_model = np.concatenate((self.voxel_model,ct),2)
        #y方向の追加
        ct = np.ones((
        self.voxel_model.shape[0],self.num_pix+1,self.voxel_model.shape[2]
        ),dtype = self.dtype)*self.grass_num
        self.voxel_model = np.concatenate((self.voxel_model,ct),1)
        self.voxel_model = np.concatenate((ct,self.voxel_model),1)
        #x方向の追加
        ct = np.ones((
        self.num_pix+1,self.voxel_model.shape[1],self.voxel_model.shape[2]
        ),dtype = self.dtype)*self.grass_num
        self.voxel_model = np.concatenate((self.voxel_model,ct),0)
        self.voxel_model = np.concatenate((ct,self.voxel_model),0)

        # end cooding
        self.voxel_model[0,:,:] = self.end_point
        self.voxel_model[-1,:,:] = self.end_point
        self.voxel_model[:,0,:] = self.end_point
        self.voxel_model[:,-1,:] = self.end_point
        self.voxel_model[:,:,0] = self.end_point
        self.voxel_model[:,:,-1] = self.end_point

    def get_second_layer_addz(self):
        return self.num_pix+1


class TuringModel_Rectangular(VoxelModel):
    def __init__(self):
        self.model_name = 'TuringModel_Rectangular'
        self.dtype_f = np.float32
        self.dtype = np.int8
        self.ct_num=2
        self.subc_num=3
        self.skin_num=4
        self.end_point = 5
        self.voxel_model = np.zeros((3,3,3),dtype = self.dtype)

        self.params = {
            'xz_size':17.15,'voxel_space':0.0245,'dicom_path':False,'bv_tv':0.138,
            'th_cortical':1.,'th_subcutaneus':2.6,'th_dermis':1.4,
            'n_space':1.,'n_trabecular':1.4,'n_cortical':1.4,'n_subcutaneus':1.4,'n_dermis':1.4,'n_air':1.,
            'ma_space':1e-8,'ma_trabecular':0.02374,'ma_cortical':0.02374,'ma_subcutaneus':0.011,'ma_dermis':0.037,
            'ms_space':1e-8,'ms_trabecular':20.54,'ms_cortical':17.67,'ms_subcutaneus':20,'ms_dermis':20,
            'g_space':0.90,'g_trabecular':0.90,'g_cortical':0.90,'g_subcutaneus':0.90,'g_dermis':.90,
            }
        self.keys = list(self.params.keys())
        self._make_model_params()
        self.voxel_space = self.params['voxel_space']

    def build(self,bone_model):
        #thickness,xy_size,voxel_space,ma,ms,g,n,n_air
        del self.voxel_model
        gc.collect()
        self.voxel_model = bone_model

        self.voxel_space = self.params['voxel_space']
        self._make_voxel_model()
        self.getModelSize()

    def set_params(self,*initial_data, **kwargs):
        set_params(self.params,self.keys,*initial_data, **kwargs)
        self._make_model_params()

    def _make_model_params(self):
        # パラメーターは、[骨梁間隙, 海綿骨, 緻密骨, 皮下組織, 皮膚, 外気]のように設定されています。
        name_list = ['_space','_trabecular','_cortical','_subcutaneus','_dermis']
        _n = [];_ma = [];_ms = [];_g = []
        for i in name_list:
            _n.append(self.params['n'+i])
            _ma.append(self.params['ma'+i])
            _ms.append(self.params['ms'+i])
            _g.append(self.params['g'+i])
        _n.append(self.params['n_air'])
        self.n = np.array(_n).astype(self.dtype_f)
        self.ma = np.array(_ma).astype(self.dtype_f)
        self.ms = np.array(_ms).astype(self.dtype_f)
        self.g = np.array(_g).astype(self.dtype_f)

    def get_params(self):
        return {
                'ms':self.ms,
                'ma':self.ma,
                'n':self.n,
                'g':self.g
                }
    def _read_dicom(self):
        path = self.params['dicom_path']
        files = os.listdir(path)
        files.sort()
        self.params['voxel_space'] = round(float(pydicom.dcmread(path+"/"+files[0],force=True).PixelSpacing[0]),5)

        ds = []
        for i in files:
            ds.append(pydicom.dcmread(path+"/"+i,force=True).pixel_array)
        ds = np.array(ds).astype("int8")
        return ds

    def add_array(self,X,num_pix,val,dtype,y_axis = False):
        # Z方向
        ct = np.zeros((X.shape[0],X.shape[1],num_pix),dtype = dtype)+val
        X = np.concatenate((ct,X),2)
        X = np.concatenate((X,ct),2)
        # X方向
        ct = np.zeros((num_pix,X.shape[1],X.shape[2]),dtype = dtype)+val
        X = np.concatenate((ct,X),0)
        X = np.concatenate((X,ct),0)
        # Y方向
        if y_axis:
            ct = np.zeros((X.shape[0],num_pix,X.shape[2]),dtype = dtype)+val
            X = np.concatenate((ct,X),1)
            X = np.concatenate((X,ct),1)

        return X

    def _make_voxel_model(self):
        if self.params['dicom_path']:
            self.voxel_model = self._read_dicom()
        A = np.zeros_like(self.voxel_model).astype(bool)
        list_num = [self.ct_num,self.subc_num,self.skin_num]
        num_s = np.round(np.array(
        [self.params['th_cortical'],self.params['th_subcutaneus'],self.params['th_dermis']]
        )/self.params["voxel_space"]).astype(np.int)

        int_num = int(self.voxel_model.shape[0]/2-round(self.params["xz_size"]/(self.params["voxel_space"]*2)))+num_s[0]
        A[int_num:-int_num,:,int_num:-int_num] = 1

        x=0
        for i in A[:,int(A.shape[2]/2),int(A.shape[0]/2)]:
            if i:
                break
            x+=1
        A = A[x:-x,:,x:-x]
        self.voxel_model = self.voxel_model[x:-x,:,x:-x]

        for i in tqdm(range(3)):
            self.voxel_model = self.add_array(self.voxel_model,num_s[i],list_num[i],np.int8)

        self.voxel_model = self.add_array(self.voxel_model,1,self.end_point,np.int8,y_axis=True)
        print("Shape of voxel_model ->",self.voxel_model.shape)


class TuringModel_Cylinder(TuringModel_Rectangular):
    def __init__(self):
        self.model_name = 'TuringModel_Cylinder'
        self.dtype_f = np.float32
        self.dtype = np.int8
        self.ct_num=2
        self.subc_num=3
        self.skin_num=4
        self.air_num=5
        self.end_point = 6
        self.voxel_model = np.zeros((3,3,3),dtype = self.dtype)

        self.params = {
            'r_bone':9.14,'voxel_space':0.0245,'dicom_path':False,'bv_tv':0.138,
            'th_cortical':1.,'th_subcutaneus':2.6,'th_dermis':1.4,
            'n_space':1.,'n_trabecular':1.4,'n_cortical':1.4,'n_subcutaneus':1.4,'n_dermis':1.4,'n_air':1.,
            'ma_space':1e-8,'ma_trabecular':0.02374,'ma_cortical':0.02374,'ma_subcutaneus':0.011,'ma_dermis':0.037,'ma_air':1e-5,
            'ms_space':1e-8,'ms_trabecular':20.54,'ms_cortical':17.67,'ms_subcutaneus':20,'ms_dermis':20,'ms_air':1e-5,
            'g_space':0.90,'g_trabecular':0.90,'g_cortical':0.90,'g_subcutaneus':0.90,'g_dermis':.90,'g_air':.90,
            }
        self.keys = list(self.params.keys())
        self._make_model_params()
        self.voxel_space = self.params['voxel_space']

    def _make_model_params(self):
        # パラメーターは、[骨梁間隙, 海綿骨, 緻密骨, 皮下組織, 皮膚, 外気]のように設定されています。
        name_list = ['_space','_trabecular','_cortical','_subcutaneus','_dermis','_air']
        _n = [];_ma = [];_ms = [];_g = []
        for i in name_list:
            _n.append(self.params['n'+i])
            _ma.append(self.params['ma'+i])
            _ms.append(self.params['ms'+i])
            _g.append(self.params['g'+i])
        _n.append(self.params['n_air'])
        self.n = np.array(_n).astype(self.dtype_f)
        self.ma = np.array(_ma).astype(self.dtype_f)
        self.ms = np.array(_ms).astype(self.dtype_f)
        self.g = np.array(_g).astype(self.dtype_f)

    def round_index(self,X,num_r):
        size_x = int(X.shape[0]/2)
        size_y = int(X.shape[1])
        size_z = int(X.shape[2]/2)
        x_lab = np.tile(np.arange(X.shape[0]),(X.shape[2], 1)).T-size_x
        y_lab = np.tile(np.arange(X.shape[2]),(X.shape[0], 1))-size_z
        r_ = np.sqrt(x_lab**2+y_lab**2)
        return np.where(r_<num_r)

    def _make_voxel_model(self):
        #骨梁間隙を0,海綿骨を1、緻密骨を2、 皮下組織を3、皮膚を4、大気を５、領域外を-1に設定する
        if self.params['dicom_path']:
            self.voxel_model = self._read_dicom()
        A = np.zeros_like(self.voxel_model).astype(bool)
        list_num = [self.ct_num,self.subc_num,self.skin_num]
        num_s = np.round(np.array(
        [self.params['th_cortical'],self.params['th_subcutaneus'],self.params['th_dermis']]
        )/self.params["voxel_space"]).astype(np.int)

        num_tr = round(self.params["r_bone"]/self.params["voxel_space"])-num_s[0]
        ind = self.round_index(A,num_tr)
        for i in range(A.shape[1]):
            A[ind[0],i,ind[1]] = 1
        ind = np.where(A==0)
        self.voxel_model[ind] = self.air_num
        x=0
        for i in A[:,int(A.shape[2]/2),int(A.shape[0]/2)]:
            if i:
                break
            x+=1
        A = A[x:-x,:,x:-x]
        self.voxel_model = self.voxel_model[x:-x,:,x:-x]

        for i in tqdm(range(3)):
            B = self.add_array(A,num_s[i],False,bool)
            self.voxel_model = self.add_array(self.voxel_model,num_s[i],self.air_num,np.int8)
            num_tr+=num_s[i]
            ind = self.round_index(B,num_tr)
            A = np.zeros_like(B).astype(bool)
            for j in range(A.shape[1]):
                A[ind[0],j,ind[1]] = 1
            ind = np.where((A&~B)==1)
            self.voxel_model[ind] = list_num[i]

        self.voxel_model = self.add_array(self.voxel_model,1,self.end_point,np.int8,y_axis=True)
        print("Shape of voxel_model ->",self.voxel_model.shape)

class TuringModel_RnC(TuringModel_Cylinder):
    def __init__(self):
        self.model_name = 'TuringModel_RnC'
        self.dtype_f = np.float32
        self.dtype = np.int8
        self.ct_num=2
        self.subc_num=3
        self.skin_num=4
        self.air_num=5
        self.end_point = 6
        self.voxel_model = np.zeros((3,3,3),dtype = self.dtype)
        self.params = {
            'r_bone':9.14,'xz_size':17.15,'voxel_space':0.0245,'dicom_path':False,'bv_tv':0.138,
            'th_cortical':1.,'th_subcutaneus':2.6,'th_dermis':1.4,
            'n_space':1.,'n_trabecular':1.4,'n_cortical':1.4,'n_subcutaneus':1.4,'n_dermis':1.4,'n_air':1.,
            'ma_space':1e-8,'ma_trabecular':0.02374,'ma_cortical':0.02374,'ma_subcutaneus':0.011,'ma_dermis':0.037,'ma_air':1e-5,
            'ms_space':1e-8,'ms_trabecular':20.54,'ms_cortical':17.67,'ms_subcutaneus':20,'ms_dermis':20,'ms_air':1e-5,
            'g_space':0.90,'g_trabecular':0.90,'g_cortical':0.90,'g_subcutaneus':0.90,'g_dermis':.90,'g_air':.90,
            }
        self.keys = list(self.params.keys())
        self._make_model_params()
        self.voxel_space = self.params['voxel_space']

    def _make_voxel_model(self):
        #骨梁間隙を0,海綿骨を1、緻密骨を2、 皮下組織を3、皮膚を4、大気を５、領域外を-1に設定する
        if self.params['dicom_path']:
            self.voxel_model = self._read_dicom()
        A = np.zeros_like(self.voxel_model).astype(bool)
        B = np.zeros_like(A).astype(bool)

        list_num = [self.ct_num,self.subc_num,self.skin_num]
        num_s = np.round(np.array(
        [self.params['th_cortical'],self.params['th_subcutaneus'],self.params['th_dermis']]
        )/self.params["voxel_space"]).astype(np.int)

        int_num = int(self.voxel_model.shape[0]/2-round(self.params["xz_size"]/(self.params["voxel_space"]*2)))+num_s[0]
        A[int_num:-int_num,:,int_num:-int_num] = 1

        num_tr = round(self.params["r_bone"]/self.params["voxel_space"])-num_s[0]
        ind = self.round_index(B,num_tr)
        for i in range(B.shape[1]):
            B[ind[0],i,ind[1]] = 1

        A = A&B
        ind = np.where(A==0)
        self.voxel_model[ind] = self.air_num
        x=0
        for i in A[:,int(A.shape[2]/2),int(A.shape[0]/2)]:
            if i:
                break
            x+=1
        A = A[x:-x,:,x:-x]
        self.voxel_model = self.voxel_model[x:-x,:,x:-x]

        for i in tqdm(range(3)):
            B = self.add_array(A,num_s[i],False,bool)
            self.voxel_model = self.add_array(self.voxel_model,num_s[i],self.air_num,np.int8)
            num_tr+=num_s[i]
            ind = self.round_index(B,num_tr)
            A = np.zeros_like(B).astype(bool)
            for j in range(A.shape[1]):
                A[ind[0],j,ind[1]] = 1
            ind = np.where((A&~B)==1)
            self.voxel_model[ind] = list_num[i]

        self.voxel_model = self.add_array(self.voxel_model,1,self.end_point,np.int8,y_axis=True)
        print("Shape of voxel_model ->",self.voxel_model.shape)

# =============================================================================
# Public montecalro model
# =============================================================================

class VoxelPlateModel(BaseVoxelMonteCarlo):
    def __init__(
        self,*,nPh=1000,dtype_f=np.float32,dtype=np.int32,
        beam_type = 'TEM00',w_beam = 0,
        beam_angle = 0,initial_refrect_by_angle = False,
        first_layer_clear = False,
    ):
        super().__init__(
            nPh = nPh, model = PlateModel(),dtype_f=dtype_f,dtype=dtype,
            w_beam=w_beam,beam_angle = beam_angle,beam_type = beam_type,
            initial_refrect_by_angle = initial_refrect_by_angle,
            first_layer_clear=first_layer_clear,
        )

class VoxelPlateExModel(BaseVoxelMonteCarlo):
    #ガラスとイントラリピッドの２層構造のみを対象とする
    #ガラスはイントラリピッドを取り囲んでいるものとする
    def __init__(
        self,*,nPh=1000,dtype_f=np.float32,dtype=np.int32,
        beam_type = 'TEM00',w_beam = 0,
        beam_angle = 0,initial_refrect_by_angle = False,
        first_layer_clear = True,
        ):
        super().__init__(
            nPh = nPh, model = PlateExModel(),dtype_f=dtype_f,dtype=dtype,
            w_beam=w_beam,beam_angle = beam_angle,beam_type = beam_type,
            initial_refrect_by_angle = initial_refrect_by_angle,
            first_layer_clear=first_layer_clear,
        )


    def build(self,*initial_data, **kwargs):
        if initial_data == () and kwargs == {}:
            pass
        else:
            self.model.set_params(*initial_data, **kwargs)
        self.model.build()

    def set_params(self,*initial_data, **kwargs):
        self.model.set_params(*initial_data, **kwargs)

    def _calc_info(self,coment=''):
        calc_info = {
            'Date':datetime.datetime.now().isoformat(),
            'coment':coment,
            'number_of_photons':self.nPh,
            'calc_dtype':self.dtype,
            'model':{
                'model_name':self.model.model_name,
                'model_params':self.model.params,
            },
            'w_beam':self.w_beam,
            'beam_angle':self.beam_angle,
            'initial_refrect_mode':self.initial_refrect_by_angle,
            'beam_mode':'TEM00',
            'fluence_mode':self.fluence_mode,
        }
        if self.beam_angle_mode:
            calc_info['wavelength'] = self.wavelength
            calc_info['beam_posision'] = self.beam_posision
            calc_info['lens_curvature_radius'] = self.lens_curvature_radius
            calc_info['grass_type'] = self.grass_type
        return calc_info

class VoxelTuringModel(BaseVoxelMonteCarlo):
    def __init__(
        self,*,nPh=1000,dtype_f=np.float32,dtype=np.int32,
        beam_type = 'TEM00',w_beam = 0,
        beam_angle = 0,initial_refrect_by_angle = False,
        first_layer_clear = False,
        model_name = 'TuringModel'
        ):
        self.namelist = ['TuringModel_Rectangular','TuringModel_Cylinder','TuringModel_RnC']
        if model_name==self.namelist[0]:
            model = TuringModel_Rectangular()
        elif model_name == self.namelist[1]:
            model = TuringModel_Cylinder()
        elif model_name == self.namelist[2]:
            model = TuringModel_RnC()
        else:
            print('Invalid name: ',model_name)

        super().__init__(
            nPh = nPh, model = model,dtype_f=dtype_f,dtype=dtype,
            w_beam=w_beam,beam_angle = beam_angle,beam_type = beam_type,
            initial_refrect_by_angle = initial_refrect_by_angle,
            first_layer_clear=first_layer_clear,
        )
        self.bone_model = False

    def _set_inital_add(self):

        if self.beam_type == 'TEM00':
            self.add =  np.zeros((3, self.nPh),dtype = self.dtype)
        self.add[0] = self._get_center_add(self.model.voxel_model.shape[0])
        self.add[1] = self._get_center_add(self.model.voxel_model.shape[1])
        if self.first_layer_clear:
            self.add[2] = self.model.get_second_layer_addz()
        else:
            self.add[2] = 1

        if self.model.model_name==self.namelist[1]:
            def _get_first_num_z(a,x):
                if a[x]==(self.model.end_point-2):
                    return x
                return _get_first_num_z(a,x+1)
            aa = self.add[:,0]
            a = self.model.voxel_model[aa[0],aa[1]]
            x=0
            zz = _get_first_num_z(a,x)
            print("Inital add for z-axis is ",zz)
            self.add[2] = zz

    def build(self,*initial_data, **kwargs):
        if initial_data == () and kwargs == {}:
            pass
        else:
            self.model.set_params(*initial_data, **kwargs)
        self.model.build(self.bone_model)
        del self.bone_model
        gc.collect()

    def set_model(self,u):
        self.bone_model = u

    def set_params(self,*initial_data, **kwargs):
        self.model.set_params(*initial_data, **kwargs)

    def get_model_fig(self,*,dpi=300,save_path = [False,False],):
        image = self.model.voxel_model
        resol0 = (image.shape[0]+1)*self.model.params['voxel_space']/2-\
        np.array([self.model.params['voxel_space']*i for i in range(image.shape[0]+1)])
        resol1 = (image.shape[1]+1)*self.model.params['voxel_space']/2-\
        np.array([self.model.params['voxel_space']*i for i in range(image.shape[1]+1)])
        resol2 = np.array([self.model.params['voxel_space']*i for i in range(image.shape[2]+1)])

        plt.figure(figsize=(5,5),dpi=100)
        plt.set_cmap(plt.get_cmap('gray'))
        plt.pcolormesh(resol0,resol2,image[:,int(image.shape[1]/2),:].T)
        plt.xlabel('X [mm]')
        plt.ylabel('Z [mm]')
        plt.ylim(resol2[-1],resol2[0])
        if save_path[0]:
            plt.savefig(
                save_path[0],
                dpi=dpi,
                orientation='portrait',
                transparent=False,
                pad_inches=0.0)
        plt.show()

        plt.figure(figsize=(6,5),dpi=100)
        plt.set_cmap(plt.get_cmap('gray'))
        plt.pcolormesh(resol1,resol2,image[int(image.shape[0]/2),:,:].T)
        plt.xlabel('Y [mm]')
        plt.ylabel('Z [mm]')
        plt.ylim(resol2[-1],resol2[0])
        if save_path[1]:
            plt.savefig(
                save_path[1],
                dpi=dpi,
                orientation='portrait',
                transparent=False,
                pad_inches=0.0)
        plt.show()

    def _calc_info(self,coment=''):
        calc_info = {
            'Date':datetime.datetime.now().isoformat(),
            'coment':coment,
            'number_of_photons':self.nPh,
            'calc_dtype':"32 bit",
            'model':{
                'model_name':self.model.model_name,
                'model_params':self.model.params,
            },
            'w_beam':self.w_beam,
            'beam_angle':self.beam_angle,
            'initial_refrect_mode':self.initial_refrect_by_angle,
            'beam_mode':'TEM00',
        }
        return calc_info
