#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 17:12:05 2020

@author: kaname
"""
## *** All parameters should be defined in millimeters ***

import numpy as np
from scipy import stats
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors
from abc import ABCMeta, abstractmethod

import datetime,time
import json,pickle,bz2

from ..utils.montecalro import MonteCalro
from ..utils import readDicom,reConstArray_8,reConstArray
from ..utils.validation import _deprecate_positional_args
from ..fluence import IntarnalFluence
from ..utils.utilities import calTime,set_params
import gc
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

__all__ = ['VoxelPlateModel','PlateModel','VoxelDicomModel']

# =============================================================================
# Base solid model
# =============================================================================

class BaseVoxelMonteCarlo(MonteCalro,metaclass = ABCMeta):
    #@_deprecate_positional_args
    @abstractmethod
    def __init__(self,*,nPh,model,fluence = False,dtype='float32',
                 nr=50,nz=20,dr=0.1,dz=0.1,d_beam = 0):
        super().__init__()
        self.dtype = dtype
        self.nPh = nPh
        self.d_beam = d_beam
        self.model = model
        self.fluence = fluence
        if self.fluence:
            self.fluence = IntarnalFluence(nr=nr,nz=nz,dr=dr,dz=dz)
    def start(self):
        self._generate_inisal_coodinate(self.nPh)
        super().start()

    def get_voxel_model(self):
        return self.model.voxel_model

    def _generate_inisal_coodinate(self,nPh,f = 'float32'):
        self.add =  np.zeros((3, nPh),dtype = 'int16')
        self.add[0] = int(self.model.voxel_model.shape[0]/2)
        self.add[1] = int(self.model.voxel_model.shape[1]/2)
        self.add[2] = 1
        self.p = np.zeros((3,nPh)).astype(f)
        self.p[2] = -self.model.voxel_space/2
        self._ser_beam_diameter()
        self.v = np.zeros((3,nPh)).astype(f)
        self.v[2] = 1
        self.w = np.ones(nPh).astype(f)
        self.w = self._initial_weight(self.w)


    def _set_beam_diameter(self):
        if self.d_beam!= 0:
            print("TEM00を入力")
            #ガウシアン分布を生成
            gb = np.array(self.gaussianBeam(self.w0)).astype(f_bit)
            #ガウシアン分布を各アドレスに振り分ける
            pp = (gb/l).astype("int16")
            ind = np.where(gb<0)
            pp[ind[0].tolist(),ind[1].tolist()] = \
                pp[ind[0].tolist(),ind[1].tolist()]-1
            pa = gb - pp*l -l/2
            ind = np.where((np.abs(pa)>=l/2))
            pa[ind[0].tolist(),ind[1].tolist()] = \
                np.sign(pa[ind[0].tolist(),ind[1].tolist()])*(l/2)

            self.add[:2] = self.add[:2] + pp
            self.p[:2] = pa.astype(self.dtype)

    def gaussianBeam(self,w=0.54):
        #TEM00のビームを生成します
        r = np.linspace(-w*2,w*2,100)
        #Ir = 2*np.exp(-2*r**2/(w**2))/(np.pi*(w**2))
        Ir = np.exp(-2*r**2/(w**2))
        normd = stats.norm(0, w/2)
        x = normd.rvs(self.nPh)
        y = normd.rvs(self.nPh)
        z = np.zeros(self.nPh)

        fig, ax1 = plt.subplots()
        ax1.set_title('Input laser light distribution')
        ax1.hist(x, bins=100, color="C0")
        ax1.set_ylabel('Number of photon')
        ax2 = ax1.twinx()
        ax2.plot(r, Ir, color="k")
        ax2.set_xlabel('X [mm]')
        ax2.set_ylabel('Probability density')
        plt.show()

        fig = plt.figure(figsize=(10,6),dpi=70)
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        H = ax.hist2d(x,y, bins=100,cmap="plasma")
        ax.set_title('Histogram for laser light intensity')
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        fig.colorbar(H[3],ax=ax)
        plt.show()
        return x,y

    def get_result(self):
        encoded_position = self._encooder(self.p_result,self.add_result)
        return{
            'p':encoded_position,
            'v':self.v_result,
            'w':self.w_result,
            'nPh':self.nPh
        }

    def get_fluence(self):
        return {'Arz':self.fluence.getArz(),
                'r':self.fluence.getArrayR(),
                'z':self.fluence.getArrayZ(),
                }

    def get_model_params(self):
        return self.model.get_params()

    def _encooder(self,p,add):
        space = self.model.voxel_space
        center_add_xy = int(self.model.voxel_model.shape[0]/2)
        encoded_position = p.copy()
        encoded_position[0] = space*(add[0]-center_add_xy)+p[0]
        encoded_position[1] = space*(add[1]-center_add_xy)+p[1]
        encoded_position[2] = np.round(space*(add[2]-1)+p[2]+space/2,6)
        return encoded_position

    def _w_update(self,w,ma,mt,p,add):
        dw = w*ma/mt
        if self.fluence != False:
            encoded_position = self._encooder(p,add)
            self.fluence.saveFluesnce(encoded_position,dw)
        return w-dw

    def _end_process(self):
        self.v_result = self.v_result[:,1:]
        self.p_result = self.p_result[:,1:]
        self.add_result = self.add_result[:,1:]
        self.w_result = self.w_result[1:]

    def _initial_weight(self,w):
        Rsp = 0
        n1 = self.model.n[-1]
        n2 = self.model.n[0]
        if n1 != n2:
            Rsp = ((n1-n2)/(n1+n2))**2
        return w-Rsp

    def set_monte_params(self,*,nPh,fluence = False,
                        dtype='float32',nr=50,nz=20,dr=0.1,dz=0.1):
        self.dtype = dtype
        self.nPh = nPh
        self.fluence = fluence
        if self.fluence:
            self.fluence = IntarnalFluence(nr=nr,nz=nz,dr=dr,dz=dz)

    @_deprecate_positional_args
    def build(self,**kwargs):
        self.model.build(**kwargs)

    def _photon_vanishing(self,p,v,w,add):
        #photn waight が0.0001以下
        del_index = np.where(w<=0.0001)[0]
        if list(del_index) != []:
            w[del_index] = self.russianRoulette(w[del_index])
            del_index = np.where(w==0)[0]
            v = np.delete(v, del_index, axis = 1)
            p = np.delete(p, del_index, axis = 1)
            w = np.delete(w, del_index)
            add = np.delete(add,del_index, axis = 1)
        return p,v,w,add

    def _border_out(self,p,v,w,add,index):
        self.v_result = np.concatenate([self.v_result, v[:,index]],axis = 1)
        self.p_result = np.concatenate([self.p_result, p[:,index]],axis = 1)
        self.add_result = np.concatenate([self.add_result, add[:,index]],axis = 1)
        self.w_result = np.concatenate([self.w_result,w[index]])

        p = np.delete(p,index, axis = 1)
        v = np.delete(v,index, axis = 1)
        add = np.delete(add,index, axis = 1)

        w = np.delete(w,index)
        return p,v,w,add

    def _border_act(self,v,p,add,s,db):
        l = self.model.voxel_space
        A = self.create01Array(np.argmin(db,axis=0),m=3)
        B = ~A

        sigAv = A*np.sign(v).astype("int16")
        ds = np.min(db,axis=0)
        s -= ds
        p = l/2*A*np.sign(v)+B*(p+v*ds)

        ni = self.model.getReflectiveIndex(add)
        nt = self.model.getReflectiveIndex(add+sigAv)
        pb_index = np.where(ni!=nt)[0]

        if list(pb_index) != []:
            #透過判別
            #入射角の計算
            ai = np.abs(A[:,pb_index]*v[:,pb_index])
            ai = np.arccos(ai[ai!=0])#Aから算出される0を排除して1次元配列にする\
            Ra = np.zeros_like(ai)

            #全反射するときのインデックスを取得
            sub_index = np.where(nt[pb_index]-ni[pb_index]<0)[0]
            subsub_index = np.where(ai[sub_index]>=np.arcsin(nt[pb_index][sub_index]/ni[pb_index][sub_index]))[0]
            #屈折角の計算
            ##(ai,at)=(0,0)の時は透過させ方向ベクトルは変化させない
            ###今の状態だと、subsub_indexもarcsinで計算しているため、しょうがないけどRuntimeWarningが出ます
            at = np.arcsin((ni[pb_index]/nt[pb_index])*np.sin(ai))
            Ra = np.random.rand(ai.size)-0.5*(np.add((np.sin(ai-at)/np.sin(ai+at))**2,
                                                     (np.tan(ai-at)/np.tan(ai+at))**2))

            #全反射はRaを強制的に0以下にし反射させる
            Ra[sub_index[subsub_index]] = -1

            #透過と反射のindexを取得
            vl_index = pb_index[np.where(Ra<=0)[0].tolist()].tolist()#反射
            sub_index = np.where(Ra>0)[0]#Raに対する透過のindex
            vt_index = pb_index[sub_index]#全体ベクトルvに対する透過のindex

            #反射するときのベクトルを更新　
            v[:,vl_index] = v[:,vl_index]*(B[:,vl_index]*1-A[:,vl_index]*1)

            #透過するときのベクトルを更新
            if vt_index !=[]:
                v[:,vt_index] = (ni[vt_index]/nt[vt_index])\
                    *((v[:,vt_index]*B[:,vt_index])\
                      +(A[:,vt_index]*np.sign(v[:,vt_index])*np.cos(at[sub_index])))

            #境界を超えた時の位置更新,番地の更新
            pt_index = np.delete(np.arange(ni.size),vl_index)
            p[:,pt_index] = p[:,pt_index]*(B[:,pt_index]*1-A[:,pt_index]*1)
            add[:,pt_index] += sigAv[:,pt_index]
        else:
            p = p*(B*1-A*1)
            add += sigAv

        return v,p,add,s

    def _boundary_judgment(self,v,p,add):
        box_model = self.model.voxel_model
        l = self.model.voxel_space
        s = -np.log(np.random.rand(self.w.size))
        v_,p_,add_,s_ = v,p,add,s
        index = np.arange(s.size)
        out_index = []
        while True:
            ###今の状態だと、ｖは最初Ｚ軸方向以外0なので、dbはしょうがないけどRuntimeWarningが出ます
            db = (l/2-p_*np.sign(v_))/np.abs(v_)
            ma = self.model.getAbsorptionCoeff(add_)
            ms = self.model.getScatteringCoeff(add_)
            mt = ma+ms
            s_ /= mt
            pb_index = np.where(s_-db.min(0)>=0)[0]
            pn_index = np.delete(np.arange(s_.size),pb_index)
            if list(pb_index) != []:
                v[:,index[pn_index]] = v_[:,pn_index]
                p[:,index[pn_index]] = p_[:,pn_index]\
                +s_[pn_index]*v_[:,pn_index]

                add[:,index[pn_index]] = add_[:,pn_index]
                index = index[pb_index]

                v_,p_,add_,s_ = self._border_act(
                    v_[:,pb_index],p_[:,pb_index],
                    add_[:,pb_index],s_[pb_index],
                    db[:,pb_index])
                v_ = v_/np.linalg.norm(v_,axis=0)
                s_ *= mt[pb_index]
                out_index_ = np.where(box_model[add_[0],add_[1],add_[2]] == -1)[0]
                if list(out_index_) != []:
                    v[:,index[out_index_]] = v_[:,out_index_]
                    p[:,index[out_index_]] = p_[:,out_index_]
                    add[:,index[out_index_]] = add_[:,out_index_]
                    out_index += list(index[out_index_])
                    v_ = np.delete(v_,out_index_, axis = 1)
                    p_ = np.delete(p_,out_index_, axis = 1)
                    add_ = np.delete(add_,out_index_, axis = 1)
                    s_ = np.delete(s_,out_index_)
                    index = np.delete(index,out_index_)
            else:
                v[:,index] = v_
                p[:,index] = p_+s_*v_
                break
        return v,p,add,out_index

    def stepMovement(self):
        self.v,self.p,self.add,index = self._boundary_judgment(self.v,self.p,self.add)

        self.p,self.v,self.w,self.add = self._border_out(self.p,self.v,self.w,self.add,index)

        G = self.model.getAnisotropyCoeff(self.add)
        self.v = self.vectorUpdate(self.v,G)
        ma = self.model.getAbsorptionCoeff(self.add)
        ms = self.model.getScatteringCoeff(self.add)
        self.w = self._w_update(self.w,ma,ma+ms,self.p,self.add)

        self.p,self.v,self.w,self.add = self._photon_vanishing(self.p,self.v,self.w,self.add)

        return self.w.size


# =============================================================================
# Modeling class
# =============================================================================

class VoxelModel:
    def build(self):
        pass
    def set_params(self):
        pass
    def getAbsorptionCoeff(self,add):
        index = self.voxel_model[add[0],add[1],add[2]]
        return self.ma[index]
    def getScatteringCoeff(self,add):
        index = self.voxel_model[add[0],add[1],add[2]]
        return self.ms[index]
    def getAnisotropyCoeff(self,add):
        index = self.voxel_model[add[0],add[1],add[2]]
        return self.g[index]
    def getReflectiveIndex(self,add):
        index = self.voxel_model[add[0],add[1],add[2]]
        return self.n[index]


class DicomBinaryModel(VoxelModel):
    def __init__(self):
        self.model_name = 'DicomBinaryModel'
        self.dtype = 'int8'
        self.dtype_f = 'float32'
        self.cort_num=2
        self.skin_num=3
        self.params = {
            'th_skin':2,'th_ct':0.03,
            'n_sp':1.,'n_tr':1.37,'n_ct':1.37,'n_skin':1.37,'n_air':1.,
            'ma_sp':0.00001,'ma_tr':0.011,'ma_ct':0.011,'ma_skin':0.037,
            'ms_sp':0.00001,'ms_tr':19.1,'ms_ct':19.1,'ms_skin':18.8,
            'g_sp':0.99,'g_tr':0.93,'g_ct':0.93,'g_skin':.93,
            }
        self.keys = list(self.params.keys())

        self.set_params()
        self.voxel_space = 0.01
        self.voxel_model = np.zeros((3,3,3),dtype = self.dtype)
        self.model_shape = (3,3,3)

    def build(self,model,voxel_space):
        del self.voxel_model
        gc.collect()

        self.dtype = model.dtype
        self.model_shape = model.shape
        self.model_size = model.size
        self.voxel_space = voxel_space
        self._make_voxel_model(model)


    def get_params(self):
        _head = {'n':0,'ma':1,'ms':2,'g':3,'th':4}
        _part = {'sp':0,'tr':1,'ct':2,'skin':3,'air':4}
        df_ = np.zeros((len(_head),len(_part)))
        for i in self.params:
            val = i.split('_')
            df_[_head[val[0]],_part[val[1]]] = self.params[i]
        df_ = pd.DataFrame(df_)
        df_.columns = _part.keys()
        df_.index = _head.keys()
        return df_

    def set_params(self,*initial_data, **kwargs):
        set_params(self.params,self.keys,*initial_data, **kwargs)
        self._make_model_params()

    def _make_model_params(self):
        # パラメーターは、[骨梁間隙, 海綿骨, 緻密骨, 皮膚, 外気]のように設定されています。
        name_list = ['_sp','_tr','_ct','_skin']
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

    def _make_voxel_model(self,model):
        # バイナリーモデルでは、骨梁間隙を0,海綿骨を1、緻密骨を2、皮膚を３、領域外を-1に設定する
        self.voxel_model = model.flatten()
        del model
        gc.collect
        index = np.where(self.voxel_model!=0)[0]
        self.voxel_model[index] = 1
        self.voxel_model = self.voxel_model.reshape(
            self.model_shape[0],self.model_shape[1],self.model_shape[2])

        if not self.params['th_ct'] !=0:
            ct = np.ones((self.model_shape[0],
                         self.model_shape[1],
                         int(self.params['th_ct']/self.voxel_space)),
                         dtype = self.dtype)*self.cort_num
            self.voxel_model = np.concatenate((ct.T,self.voxel_model.T)).T

        if self.params['th_skin'] != 0:
            skin = np.ones((self.model_shape[0],
                           self.model_shape[1],
                           int(self.params['th_skin']/self.voxel_space)+1),
                           dtype = self.dtype)*self.skin_num
            self.voxel_model = np.concatenate((skin.T,self.voxel_model.T)).T

        self.voxel_model[0,:,:] = -1
        self.voxel_model[-1,:,:] = -1
        self.voxel_model[:,0,:] = -1
        self.voxel_model[:,-1,:] = -1
        self.voxel_model[:,:,0] = -1
        self.voxel_model[:,:,-1] = -1

class DicomLinearModel(DicomBinaryModel):
    def __init__(self):
        self.model_name = 'DicomBinaryModel'
        self.dtype = 'int8'
        self.dtype_f = 'float32'
        self.cort_num=20
        self.skin_num=30
        self.params = {
            'th_skin':2,'th_ct':0.03,
            'n_sp':1.,'n_tr':1.37,'n_ct':1.37,'n_skin':1.37,'n_air':1.,
            'ma_sp':0.00001,'ma_tr':0.011,'ma_ct':0.011,'ma_skin':0.037,
            'ms_sp':0.00001,'ms_tr':np.nan,'ms_ct':19.1,'ms_skin':18.8,
            'g_sp':0.99,'g_tr':0.93,'g_ct':0.93,'g_skin':.93,
            }
        self.keys = list(self.params.keys())

        self.set_params()
        self.voxel_space = 0.01
        self.voxel_model = np.zeros((3,3,3),dtype = self.dtype)
        self.model_shape = (3,3,3)


    def getAbsorptionCoeff(self,add):
        # パラメーターは、[骨梁間隙, 海綿骨, 緻密骨, 皮膚, 外気]
        val = self.voxel_model[add[0],add[1],add[2]]
        val[val== 0 ]=self.ma[0]
        val[val > 0 ]=self.ma[1]
        val[val==-20]=self.ma[2]
        val[val==-30]=self.ma[3]
        return self.ma[index]

    def getScatteringCoeff(self,add):
        val = self.voxel_model[add[0],add[1],add[2]]
        #### 確認が必要 #####
        val[val > 0 ]=(0.016*val[val>0]*256-119.75)/10
        val[val== 0 ]=self.ms[0]
        val[val==-20]=self.ms[2]
        val[val==-30]=self.ms[3]
        return self.ms[index]

    def getAnisotropyCoeff(self,add):
        val = self.voxel_model[add[0],add[1],add[2]]
        val[val== 0 ]=self.g[0]
        val[val > 0 ]=self.g[1]
        val[val==-20]=self.g[2]
        val[val==-30]=self.g[3]
        return val

    def getReflectiveIndex(self,add):
        # パラメーターは、[骨梁間隙, 海綿骨, 緻密骨, 皮膚, 外気]のように設定されています。
        val = self.voxel_model[add[0],add[1],add[2]]
        val[val== -1]=self.n[-1]
        val[val== 0 ]=self.n[0]
        val[val > 0 ]=self.n[1]
        val[val==-20]=self.n[2]
        val[val==-30]=self.n[3]
        return val


class PlateModel(VoxelModel):
    @_deprecate_positional_args
    def __init__(
        self,*,thickness=[0.2,] ,xy_size=0.1 ,voxel_space = 0.1,
        ma=[1,],ms=[100,],g=[0.9,],n=[1.37,],n_air=1,f = 'float32'):
        self.model_name = 'PlateModel'
        self.n =np.array(n+[n_air]).astype(f)
        self.ms = np.array(ms).astype(f)
        self.ma = np.array(ma).astype(f)
        self.g = np.array(g).astype(f)
        self.voxel_space = voxel_space
        self.xy_size = xy_size
        self.borderposit = self._make_borderposit(thickness,f)
        self._make_voxel_model()

    def _make_borderposit(self,thickness,f):
        thickness = [0]+thickness
        b = 0; b_list = []
        for i in  thickness:
            b += i
            b_list.append(b)
        return np.array(b_list).astype(f)

    def _make_voxel_model(self):
        nxy_box = np.round(self.xy_size/self.voxel_space).astype(int)
        nz_box = np.round(self.borderposit/self.voxel_space).astype(int)
        self.voxel_model = np.zeros((nxy_box+2,nxy_box+2,nz_box[-1]+2),dtype = 'int8')
        for i in range(nz_box.size-1):
            self.voxel_model[:,:,nz_box[i]+1:nz_box[i+1]+1] = i
        self.voxel_model[0] = -1;self.voxel_model[-1] = -1
        self.voxel_model[:,0] = -1; self.voxel_model[:,-1] = -1
        self.voxel_model[:,:,0] = -1; self.voxel_model[:,:,-1] = -1

    @_deprecate_positional_args
    def build(self,thickness,xy_size,voxel_space,ma,ms,g,n,n_air,f = 'float32'):
        del self.voxel_model
        gc.collect()
        #-1はモデルの外側
        self.voxel_space = voxel_space
        self.xy_size = xy_size
        self.borderposit = self._make_borderposit(thickness,f)
        self._make_voxel_model()

        self.n =np.array([n_air]+n).astype(f)
        self.ms = np.array(ms).astype(f)
        self.ma = np.array(ma).astype(f)
        self.g = np.array(g).astype(f)
        self.getModelSize()

    def get_params(self):
        return {'ms':self.ms,
                'ma':self.ma,
                'n':self.n,
                'g':self.g}

    def getModelSize(self):
        print("Memory area size for voxel storage: %0.3f Mbyte" % (self.voxel_model.nbytes*1e-6))

# =============================================================================
# Public montecalro model
# =============================================================================
class VoxelPlateModel(BaseVoxelMonteCarlo):
    def __init__(self,*,nPh=1000,fluence=False,dtype='float32',
                 nr=50,nz=20,dr=0.1,dz=0.1,):

        super().__init__(nPh = nPh,fluence =fluence, model = PlateModel(),
                         dtype='float32',nr=nr,nz=nz,dr=dr,dz=dz)

class VoxelDicomModel(BaseVoxelMonteCarlo):

    def __init__(self,*,nPh,fluence=False,dtype='float32',
                 nr=50,nz=20,dr=0.1,dz=0.1,model_type = 'binary'):

        self.model_type = model_type
        if model_type == 'binary':
            model = DicomBinaryModel()
        elif model_type == 'liner':
            model = DicomLinearModel()

        super().__init__(nPh = nPh,fluence =fluence, model = model,
                         dtype='float32',nr=nr,nz=nz,dr=dr,dz=dz)

        self._right = 0
        self._left = 0
        self._upper = 0
        self._lower = 0
        self._bottom = 0
        self._top = 0
        self.threshold = 9500
        self.array_dicom = np.zeros((3,3,3),dtype='int8')

        self.rot180_y_status = False
        self.rot90_z_status = 0

    def build(self,*initial_data, **kwargs):
        self.model.set_params(*initial_data, **kwargs)
        try:
            self.model.build(self.array_dicom,self.ConstPixelSpacing[0])
            del self.array_dicom
            gc.collect()
        except:
            warnings.warn('New voxel_model was not built')

    def set_params(self,*initial_data, **kwargs):
        self.model.set_params(*initial_data, **kwargs)

    def _calc_info(self):
        calc_info = {
            'Date':datetime.datetime.now().isoformat(),
            'number_of_photons':self.nPh,
            'calc_dtype':self.dtype,
            'model':{
                'model_type':self.model_type,
                'model_name':self.model.model_name,
                'model_params':self.model.params,
                'model_shape':self.model_shape,
                'model_dtype':self.model.dtype,
            },
            'dicom'{
                'dicom_path':self.path,
                'ConstPixelSpacing':self.ConstPixelSpacing,
                'ConstPixelDims':self.ConstPixelDims,
                'voxcel_processing':{
                    'rot180_y':self.rot180_y_status,
                    'rot90_z':self.rot90_z_status,
                    'trimd_size':self.trimd_size,
                    'trim_right_pixel':self._right,
                    'trim_left_pixel':self._left,
                    'trim_upper_pixel':self._upper,
                    'trim_lower_pixel':self._lower,
                    'trim_top_pixel':self._top,
                    'trim_bottom_pixel':self._bottom,
                    'bone_threshold':self.threshold,
                },
                'fluence_mode':self.fluence_mode,
            }
        }
        return calc_info

    def save_result(fname = "test"):
        start_ = time.time()

        res = self.get_result()
        with bz2.open(fname+".pkl.bz2", 'wb') as fp:
            fp.write(pickle.dumps(res))

        info = self._calc_info()
        with open(fname+".json", 'w') as fp:
            json.dump(info,fp,indent=4)

        calTime(time.time(), start_)

    def display_histmap(self):
        fig = plt.figure(figsize=(10,6),dpi=200)
        ax = fig.add_subplot(111)
        ax.set_aspect('equal')
        _index = self.getRdTtIndex()
        res = self.get_result()
        y = res['p'][1,_index['Rd']]
        x = res['p'][0,_index['Rd']]
        H = ax.hist2d(x,y, bins=2**10,cmap="plasma",norm=colors.LogNorm())
        ax.set_title('Hist map at XY sureface on Z = 0 mm')
        ax.set_xlabel('X [mm]')
        ax.set_ylabel('Y [mm]')
        fig.colorbar(H[3],ax=ax)
        plt.show()

    def import_dicom(self,path,dtype = 'int8'):
        del self.array_dicom
        gc.collect()
        self.path = path
        self.dtype = dtype
        array_dicom,ConstPixelDims,ConstPixelSpacing = readDicom(path)
        self.array_dicom = array_dicom
        self.ConstPixelSpacing = list(ConstPixelSpacing)
        self.ConstPixelDims = ConstPixelDims
        print("Memory area size for voxel storage: %0.3f Mbyte" % (self.array_dicom.nbytes*1e-6))

    def display_cross_section(self,*,xx = 0,yy = 0,zz = 0,
                              int_pix = True, section_line = True,
                              graph_type = 'ALL', hist_type = 'XY',cmap = 'gray',
                              image = False,):
        if image is False:
            image = self.array_dicom

        if int_pix:
            x_size = xx*self.ConstPixelSpacing[0]
            y_size = yy*self.ConstPixelSpacing[1]
            z_size = zz*self.ConstPixelSpacing[2]
        else:
            x_size = int(xx)
            y_size = int(yy)
            z_size = int(zz)
            xx = int(x_size/self.ConstPixelSpacing[0])
            yy = int(y_size/self.ConstPixelSpacing[1])
            zz = int(z_size/self.ConstPixelSpacing[2])

        resol0 = [self.ConstPixelSpacing[0]*i for i in range(image.shape[0]+1)]
        resol1 = [self.ConstPixelSpacing[1]*i for i in range(image.shape[1]+1)]
        resol2 = [self.ConstPixelSpacing[2]*i for i in range(image.shape[2]+1)]
        if graph_type == 'XY':
            plt.figure(figsize=(6,6),dpi=100)
            plt.set_cmap(plt.get_cmap(cmap))
            plt.title('X-Y pic in Z = %0.3f mm' %(z_size))
            plt.pcolormesh(resol1,resol0,image[:,:,zz])
            plt.xlabel('X mm')
            plt.ylabel("Y mm")
            plt.show()
        if graph_type == 'XY-YZ':
            fig,ax = plt.subplots(1,2,figsize=[12,6],dpi=90)
            plt.set_cmap(plt.get_cmap(cmap))
            ax[0].set_title('X-Y pic in Z = %0.3f mm' %(z_size))
            ax[0].pcolormesh(resol1,resol0,image[:,:,zz])
            ax[0].set_xlabel("X mm")
            ax[0].set_ylabel("Y mm")
            if section_line:
                ax[0].plot([x_size,x_size],[0,resol0[-1]],'-',c = 'r')

            ax[1].set_title('Y-Z pic in X = %0.3f mm' %(x_size))
            ax[1].pcolormesh(resol2,resol0,image[:,xx,:])
            ax[1].set_xlabel("Z mm")
            ax[1].set_ylabel("Y mm")
            if section_line:
                ax[1].plot([z_size,z_size],[0,resol0[-1]],'-',c = 'r')
            plt.show()
        if graph_type == 'ALL' or graph_type == 'NO_HIST':
            fig,ax = plt.subplots(2,2,figsize=[12,12],dpi=90)
            plt.set_cmap(plt.get_cmap(cmap))
            ax[0,0].set_title('X-Y pic in Z = %0.3f mm' %(z_size))
            ax[0,0].pcolormesh(resol1,resol0,image[:,:,zz])
            ax[0,0].set_xlabel("X mm")
            ax[0,0].set_ylabel("Y mm")
            if section_line:
                ax[0,0].plot([x_size,x_size],[0,resol0[-1]],'-',c = 'r')
                ax[0,0].plot([0,resol1[-1]],[y_size,y_size],'-',c = 'r')

            ax[0,1].set_title('Y-Z pic in X = %0.3f mm' %(x_size))
            ax[0,1].pcolormesh(resol2,resol0,image[:,xx,:])
            ax[0,1].set_xlabel("Z mm")
            ax[0,1].set_ylabel("Y mm")
            if section_line:
                ax[0,1].plot([z_size,z_size],[0,resol0[-1]],'-',c = 'r')

            ax[1,0].set_title('X-Z pic in Y = %0.3f mm' %(y_size))
            ax[1,0].pcolormesh(resol1,resol2,image[yy,:,:].T)
            ax[1,0].set_ylim(resol2[-1],resol2[0])
            ax[1,0].set_xlabel("X mm")
            ax[1,0].set_ylabel("Z mm")
            if section_line:
                ax[1,0].plot([0,resol1[-1]],[z_size,z_size],'-',c = 'r')

            if graph_type == 'ALL':
                if hist_type == 'XY':
                    ax[1,1].set_title('Histogram of X-Y pic pixel values')
                    ax[1,1].hist(image[:,:,zz].flatten(),bins=50, color='c')
                elif hist_type == 'YZ':
                    ax[1,1].set_title('Histogram of Y-Z pic pixel values')
                    ax[1,1].hist(image[:,xx,].flatten(),bins=50, color='c')
                elif hist_type == 'XZ':
                    ax[1,1].set_title('Histogram of X-Z pic pixel values')
                    ax[1,1].hist(image[yy,:,].flatten(),bins=50, color='c')
                elif hist_type =='ALL':
                    ax[1,1].set_title('Histogram of X-Z pic pixel values')
                    ax[1,1].hist(image.flatten(),bins=50, color='c')
                ax[1,1].set_yscale('log')
                ax[1,1].set_xlabel("Voxel values")
                ax[1,1].set_ylabel("Frequency")
        plt.show()

    def rot180_y(self):
        self.rot180_y_status = not self.rot180_y_status
        self.array_dicom = np.flip(self.array_dicom)
        self.array_dicom = np.flipud(self.array_dicom)

    def rot90_z(self,k=1):
        self.rot90_z_status += 90
        if self.rot90_z_status == 360:
            self.rot90_z_status = 0
        self.array_dicom = np.rot90(self.array_dicom,k=k)


    def trim_area(self,*, right = 0, left = 0,
                  upper ,lower =0,
                  top = 0,bottom = 0,
                  int_pix = True,cmap = 'gray'):
        img = self.array_dicom.copy()

        if int_pix:
            right = right*-1
            upper = upper*-1
        else:
            right = int(right/self.ConstPixelSpacing[0])*-1
            left = int(left/self.ConstPixelSpacing[0])
            upper = int(upper/self.ConstPixelSpacing[1])*-1
            lower = int(lower/self.ConstPixelSpacing[1])
            bottom = int(bottom/self.ConstPixelSpacing[2])*-1
            top = int(top/self.ConstPixelSpacing[2])

        d_rate = 2
        for i,po in enumerate([upper,right,bottom]):
            if i == 0:
                if po != 0:
                    img[po:,:,:] = img[po:,:,:]/d_rate
            elif i == 1:
                if po != 0:
                    img[:,po:,:] = img[:,po:,:]/d_rate
            elif i == 2:
                if po != 0:
                    img[:,:,po:] = img[:,:,po:]/d_rate
        img[:lower,:,:] = img[:lower,:,:]/d_rate
        img[:,:left,:] = img[:,:left,:]/d_rate
        img[:,:,:top] = img[:,:,:top]/d_rate

        df = pd.DataFrame(columns = ['right','left','upper','lower','top','bottom'])
        df_array = np.array([right,left,upper,lower,top,bottom])
        df.loc['Pixel number'] = df_array
        df.loc['Position [mm]'] = np.round(df_array*self.ConstPixelSpacing[0],3)
        print('Trimming parameters')
        print(abs(df))

        self.display_cross_section(image = img,zz = top,
                              xx = int(img.shape[0]/2),
                              yy = int(img.shape[1]/2),
                              cmap = cmap)
        del img
        gc.collect()

        self._right = right
        self._left = left
        self._upper = upper
        self._lower = lower
        self._bottom = bottom
        self._top = top


    def set_trim(self,threshold=False,cmap = 'gray'):
        if not threshold is False:
            if self.dtype == 'int8':
                self.threshold = int(threshold*2**8)
            elif self.dtype == 'int16':
                self.threshold = int(threshold*2**8)
        self.array_dicom = reConstArray_8(self.array_dicom,self.threshold)

        d_num = -10
        d_size = self.array_dicom.nbytes*1e-6

        for i,po in enumerate([self._upper,self._right,self._bottom]):
            if i == 0:
                if po != 0:
                    self.array_dicom[po:,:,:] = d_num
            elif i == 1:
                if po != 0:
                    self.array_dicom[:,po:,:] = d_num
            elif i == 2:
                if po != 0:
                    self.array_dicom[:,:,po:] = d_num
        self.array_dicom[:self._lower,:,:] = d_num
        self.array_dicom[:,:self._left,:] = d_num
        self.array_dicom[:,:,:self._top] = d_num

        img_shape = self.array_dicom.shape

        self.array_dicom = self.array_dicom.ravel()
        index = np.where(self.array_dicom==d_num)[0]
        self.array_dicom = np.delete(self.array_dicom,index)
        L0 = self.ConstPixelDims[0]-abs(self._lower)-abs(self._upper )
        L1 = self.ConstPixelDims[1]-abs(self._left )-abs(self._right)
        L2 = self.ConstPixelDims[2]-abs(self._top)-abs(self._bottom)
        self.array_dicom = self.array_dicom.reshape([L0,L1,L2])

        self.display_cross_section(zz = 0,
                              xx = int(self.array_dicom.shape[0]/2),
                              yy = int(self.array_dicom.shape[1]/2),
                              cmap = cmap)

        print("#########  Size  #########")
        print('* Image shape was changed')
        print('  from (%d,%d,%d)' %img_shape)
        self.trimd_size = self.array_dicom.shape
        print('  to   (%d,%d,%d)' %self.trimd_size)
        print()
        print('* Memory area size for')
        print('  voxel storage was changed')
        print('  from %0.3f Mbyte' %d_size)
        print('  to   %0.3f Mbyte' %(self.array_dicom.nbytes*1e-6))

    def reset_setting(self):
        del self.array_dicom
        gc.collect()
        array_dicom,ConstPixelDims,ConstPixelSpacing = readDicom(
            self.path,self.dtype)
        self.array_dicom = array_dicom
        self.ConstPixelSpacing = list(ConstPixelSpacing)
        self.ConstPixelDims = ConstPixelDims

    def check_threshold(self,threshold=37,cmap = 'gray',graph_type = 'XY'):
        image = reConstArray(self.array_dicom,threshold)
        self.display_cross_section(image = image,zz = 0,
                              xx = int(image.shape[0]/2),
                              yy = int(image.shape[1]/2),
                              cmap = cmap,graph_type = graph_type)
        self.threshold = int(threshold*2**8)

    def set_threshold(self,threshold=37,cmap = 'gray',graph_type = 'XY'):
        self.array_dicom = reConstArray(self.array_dicom,threshold)
        self.display_cross_section(zz = 0,
                              xx = int(self.array_dicom.shape[0]/2),
                              yy = int(self.array_dicom.shape[1]/2),
                              cmap = cmap,graph_type = graph_type)
        self.threshold = int(threshold*2**8)

    def _initial_weight(self,w):
        Rsp = 0
        n2 = -1
        n1 = self.model.n[-1]
        if self.model.params['th_skin'] != 0:
            n2 = self.model.n[3]
        elif self.model.params['th_ct'] != 0:
            n2 = self.model.n[2]

        if n1 != n2 and n2 > 0:
            Rsp = ((n1-n2)/(n1+n2))**2
        elif n2 < 0:
            n1 = np.ones(self.w.size)
            n2 = self.model.getReflectiveIndex(self.add)
            Rsp = ((n1-n2)/(n1+n2))**2
        return w-Rsp
