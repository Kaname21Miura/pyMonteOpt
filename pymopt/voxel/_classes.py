#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 17:12:05 2020

@author: kaname
"""
## *** All parameters should be defined in millimeters ***

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

from ..utils.montecalro import MonteCalro
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
'VoxelPlateModel',
'PlateModel',
'VoxelDicomModel',
'VoxelSeparatedPlateModel',
'VoxelPlateLedModel',
'VoxelWhiteNoiseModel',
'VoxelTuringModel']

# =============================================================================
# Base solid model
# =============================================================================

class BaseVoxelMonteCarlo(MonteCalro,metaclass = ABCMeta):
    #@_deprecate_positional_args
    @abstractmethod
    def __init__(self,*,nPh,model,fluence_mode =False, dtype='float32',
                 nr=50,nz=20,dr=0.1,dz=0.1,
                 beam_type = 'TEM00',w_beam = 0,
                 beam_angle = 0,initial_refrect_by_angle = False,
                 z_max_mode = False,
                 wavelength = 850,beam_posision = 10,
                 lens_curvature_radius = 51.68,grass_type = 'N-BK7',
                 beam_dist=False,
                 beam_direct='positive',intermediate_buffer=False):
        super().__init__()

        def __check_list_name(name,name_list):
            if not(name in name_list):
                raise ValueError('%s is not a permitted for factor. Please choose from %s.'%(name,name_list))

        self.beam_type_list=['TEM00','Free',False]
        __check_list_name(beam_type,self.beam_type_list)
        self.beam_type = beam_type

        self.beam_direct_list = ['positive','negative',False]
        __check_list_name(beam_direct,self.beam_direct_list)
        self.beam_direct = beam_direct
        self.intermediate_buffer = intermediate_buffer
        self.set_beam_dist(beam_dist)

        self.z_max_mode = z_max_mode

        self.dtype = dtype
        self.nPh = nPh
        self.w_beam = w_beam

        self.initial_refrect_by_angle = initial_refrect_by_angle
        self.wavelength = wavelength
        self.grass_type = grass_type
        self.lens_curvature_radius = lens_curvature_radius
        self.beam_posision = beam_posision
        self.beam_angle_mode = False
        if beam_angle == 'lens_f':
            grass = Grass()
            self.beam_angle_mode = True
            self.beam_angle = grass.get_inital_angle(
                slit_radius = self.beam_posision,
                lens_curvature_radius = self.lens_curvature_radius,
                wavelength = self.wavelength,
                grass_type = self.grass_type
            )
        else:
            self.beam_angle = beam_angle

        self.model = model
        self.fluence = fluence_mode
        self.fluence_mode = fluence_mode
        self.generate_initial = True
        if self.fluence:
            if fluence_mode == '2D':
                self.fluence = Fluence2D(nr=nr,nz=nz,dr=dr,dz=dz)
            elif fluence_mode == '3D':
                self.fluence = Fluence3D(nr=nr,nz=nz,dr=dr,dz=dz)

    def start(self):
        self.nPh = int(self.nPh)
        self._reset_results()
        if self.generate_initial:
            self._generate_initial_coodinate(self.nPh)
        super().start()
        return self

    def _reset_results(self):
        self.v_result = np.empty((3,1)).astype(self.f_bit)
        self.p_result = np.empty((3,1)).astype(self.f_bit)
        self.add_result = np.empty((3,1)).astype('int16')
        self.w_result = np.empty(1).astype(self.f_bit)
        if self.z_max_mode:
            self.z_max_result = np.empty(1).astype(self.f_bit)
        return self

    def get_voxel_model(self):
        return self.model.voxel_model

    def _generate_initial_coodinate(self,nPh,f = 'float32'):
        self._set_inital_add()
        self._set_beam_distribution()
        self._set_inital_vector()
        if self.z_max_mode:
            self.z_max = np.zeros(self.nPh).astype(self.dtype)
            if self.beam_angle!=0 and self.w_beam==0:
                if self.initial_refrect_by_angle:
                    self.z_max = np.delete(self.z_max, np.arange(self.inital_del_num), 0)
                    self.z_max_result = np.concatenate([self.z_max_result,
                    self.z_max[:self.inital_del_num]],axis = 0)

        self._set_inital_w()


    def set_beam_dist(self,beam_dist):
        if beam_dist:
            self.beam_dist=beam_dist.copy()
            if self.beam_direct==self.beam_direct_list[0]:
                #positive
                if self.intermediate_buffer:
                    self.target_index = np.where((beam_dist['v'][2]>0)\
                    &(beam_dist['p'][2]>=self.intermediate_buffer-1e-8))[0]
                else:
                    self.target_index = np.where(beam_dist['v'][2]>0)[0]

            elif self.beam_direct==self.beam_direct_list[1]:
                #negative
                if self.intermediate_buffer:
                    self.target_index = np.where((beam_dist['v'][2]<0)\
                    &(beam_dist['p'][2]<=0+1e-8))[0]
                else:
                    self.target_index = np.where(beam_dist['v'][2]<0)[0]
                    self.beam_dist['v'][2] *= -1

            self.beam_dist['v'] = beam_dist['v'][:,self.target_index]
            self.beam_dist['p'] = beam_dist['p'][:,self.target_index]
            self.beam_dist['w'] = beam_dist['w'][self.target_index]

    def _set_inital_add(self):
        if self.beam_type == 'TEM00':
            self.add =  np.zeros((3, self.nPh),dtype = 'int16')
        elif self.beam_type == 'Free':
            self.add =  np.zeros((3, self.beam_dist['p'].shape[1]),dtype = 'int16')
        self.add[0] = self._get_center_add(self.model.voxel_model.shape[0])
        self.add[1] = self._get_center_add(self.model.voxel_model.shape[1])
        self.add[2] = 1

    def _get_center_add(self,length):
        #addの中心がローカル座標（ボックス内）の中心となるため、
        #ボクセル数が偶数の時は、1/2小さい位置を中心とし光を照射し、
        #逆変換時（_encooder）も同様に1/2小さい位置を中心として元のマクロな座標に戻す。
        return int((length-1)/2)

    def _set_inital_vector(self):
        if self.beam_type == 'TEM00':
            self.v = np.zeros((3,self.nPh)).astype(self.dtype)
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
                    sub_v = np.zeros((3,self.inital_del_num)).astype(self.dtype)
                    sub_v[0] = np.sin(ai)
                    sub_v[2] = -np.cos(ai)
                    self.v_result = np.concatenate([self.v_result,
                    sub_v],axis = 1)
                    self.p_result = np.concatenate([self.p_result,
                    self.p[:,:self.inital_del_num]],axis = 1)
                    self.add_result = np.concatenate([self.add_result,
                    self.add[:,:self.inital_del_num]],axis = 1)

        elif self.beam_type == 'Free':
            self.v = self.beam_dist['v']

    def _set_inital_w(self):


        if self.beam_type == 'TEM00':
            self.w = np.ones(self.nPh).astype(self.dtype)

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

                self.w -= Rsp

            if self.beam_angle!=0 and self.w_beam==0:
                if self.initial_refrect_by_angle:
                    self.w[:] = 1
                    self.w = np.delete(self.w, np.arange(self.inital_del_num), 0)
                    self.w_result = np.concatenate([self.w_result,
                    self.w[:self.inital_del_num]],axis = 0)


        elif self.beam_type == 'Free':
            self.w= self.beam_dist['w']

    def _initial_weight(self,w):
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
        return w-Rsp
    def _set_beam_distribution(self):

        if self.beam_type == 'TEM00':
            self.p = np.zeros((3,self.nPh)).astype(self.dtype)
            self.p[2] = -self.model.voxel_space/2
            if self.w_beam!= 0:
                print("%sを入力"%self.beam_type)
                #ガウシアン分布を生成
                gb = np.array(self.gaussianBeam(self.w_beam)).astype(self.dtype)
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
                self.p[:2] = pa.astype(self.dtype)

        elif self.beam_type == 'Free':
            self.p = np.zeros((3,self.beam_dist['p'].shape[1])).astype(self.dtype)
            self.p[2] = -self.model.voxel_space/2
            print("%sを入力"%self.beam_type)
            gb = self.beam_dist['p'][:2]
            self._get_beam_dist(gb[0],gb[1])

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
            self.p[:2] = pa.astype(self.dtype)

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
        if self.z_max_mode:
            df_result['z_max'] = self.z_max_result
        return df_result


    def get_fluence(self):
        return {'Arz':self.fluence.getArz(),
                'r':self.fluence.getArrayR(),
                'z':self.fluence.getArrayZ(),
                }

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

    def _w_update(self,w,ma,mt,p,add):
        dw = w*ma/mt
        if self.fluence != False:
            encoded_position = self._encooder(p,add)
            self.fluence.saveFluesnce(encoded_position,dw/ma)
        return w-dw

    def _end_process(self):
        self.v_result = self.v_result[:,1:]
        self.p_result = self.p_result[:,1:]
        self.add_result = self.add_result[:,1:]
        self.w_result = self.w_result[1:]
        if self.z_max_mode:
            self.z_max_result = self.z_max_result[1:]

    def set_monte_params(self,*,nPh,model,
        fluence_mode =False, dtype='float32',
        nr=50,nz=20,dr=0.1,dz=0.1,w_beam = 0):
        self.dtype = dtype
        self.nPh = nPh
        self.w_beam = w_beam
        self.model = model
        self.fluence = fluence_mode
        self.fluence_mode = fluence_mode
        if self.fluence:
            if fluence_mode == '2D':
                self.fluence = Fluence2D(nr=nr,nz=nz,dr=dr,dz=dz)
            elif fluence_mode == '3D':
                self.fluence = Fluence3D(nr=nr,nz=nz,dr=dr,dz=dz)

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
            if self.z_max_mode:
                self.z_max = np.delete(self.z_max, del_index)

        # nanデータを削除
        # なぜnanになるのかは究明する必要がある。
        # 現在以下のモデルでp,vがnanになる可能性が確認されている。
        # VoxelSeparatedPlateModel,
        if np.isnan(p).any():
            print('Nan occurs in vector p')
            del_index = np.where(np.isnan(p)[0])[0]
            v = np.delete(v, del_index, axis = 1)
            p = np.delete(p, del_index, axis = 1)
            w = np.delete(w, del_index)
            add = np.delete(add,del_index, axis = 1)
            if self.z_max_mode:
                self.z_max = np.delete(self.z_max, del_index)
        return p,v,w,add

    def _border_out(self,p,v,w,add,index):
        self.v_result = np.concatenate([self.v_result, v[:,index]],axis = 1)
        self.p_result = np.concatenate([self.p_result, p[:,index]],axis = 1)
        self.add_result = np.concatenate([self.add_result, add[:,index]],axis = 1)
        self.w_result = np.concatenate([self.w_result,w[index]])
        if self.z_max_mode:
            self.z_max_result = np.concatenate([self.z_max_result,self.z_max[index]])
            self.z_max = np.delete(self.z_max,index)

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
                #外側に出る？
                out_index_ = np.where(
                    (box_model[add_[0],add_[1],add_[2]] <= -1)\
                    &(box_model[add_[0],add_[1],add_[2]] >= -5)
                )[0]
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
        if self.z_max_mode:
            encoded_p = self._encooder(self.p.copy(),self.add.copy())[2]
            ind_zmax = np.where(encoded_p-self.z_max>0)[0]
            if list(ind_zmax) != []:
                self.z_max[ind_zmax] = encoded_p[ind_zmax]

        self.p,self.v,self.w,self.add = self._border_out(self.p,self.v,self.w,self.add,index)

        G = self.model.getAnisotropyCoeff(self.add)
        self.v = self.vectorUpdate(self.v,G)
        ma = self.model.getAbsorptionCoeff(self.add)
        ms = self.model.getScatteringCoeff(self.add)
        self.w = self._w_update(self.w,ma,ma+ms,self.p,self.add)

        self.p,self.v,self.w,self.add = self._photon_vanishing(self.p,self.v,self.w,self.add)

        return self.w.size

    def save_result(self,fname,coment=''):
        start_ = time.time()

        res = self.get_result()
        save_name = fname+"_LID.pkl.bz2"
        with bz2.open(save_name, 'wb') as fp:
            fp.write(pickle.dumps(res))
        print("Monte Carlo results saved in ")
        print("-> %s" %(save_name))
        print('')
        info = self._calc_info(coment)
        save_name = fname+"_info.json"
        with open(save_name, 'w') as fp:
            json.dump(info,fp,indent=4,cls= ToJsonEncoder)
        print("Calculation conditions are saved in")
        print("-> %s" %(save_name))
        print('')
        if self.fluence_mode:
            fe = self.fluence.getArz()
            save_name = fname+'_fluence'+self.fluence_mode+".pkl.bz2"
            with bz2.open(save_name, 'wb') as fp:
                fp.write(pickle.dumps(fe))
            print("Internal Fluence saved in ")
            print("-> %s" %(save_name))
            print('')

        calTime(time.time(), start_)

    def _calc_info(self,coment=''):
        _params = self.model.get_params()
        calc_info = {
            'Date':datetime.datetime.now().isoformat(),
            'coment':coment,
            'number_of_photons':self.nPh,
            'calc_dtype':self.dtype,
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
        if self.beam_angle_mode:
            calc_info['wavelength'] = self.wavelength
            calc_info['beam_posision'] = self.beam_posision
            calc_info['lens_curvature_radius'] = self.lens_curvature_radius
            calc_info['grass_type'] = self.grass_type
        return calc_info


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
            'ma_sp':1e-8,'ma_tr':0.011,'ma_ct':0.011,'ma_skin':0.037,
            'ms_sp':1e-8,'ms_tr':19.1,'ms_ct':19.1,'ms_skin':18.8,
            'g_sp':0.90,'g_tr':0.93,'g_ct':0.93,'g_skin':.93,
            }
        self.keys = list(self.params.keys())

        self._make_model_params()
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

        if self.params['th_ct'] !=0:
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
        self.model_name = 'DicomLinearModel'
        self.dtype = 'int8'
        self.dtype_f = 'float32'
        self.cort_num=-20
        self.skin_num=-30
        self.params = {
            'th_skin':2,'th_ct':0.3,
            'n_sp':1.,'n_tr':1.37,'n_ct':1.37,'n_skin':1.37,'n_air':1.,
            'ma_sp':0.00001,'ma_tr':0.011,'ma_ct':0.011,'ma_skin':0.037,
            'ms_sp':0.00001,'ms_tr':np.nan,'ms_ct':19.1,'ms_skin':18.8,
            'g_sp':0.99,'g_tr':0.93,'g_ct':0.93,'g_skin':.93,
            }
        self.keys = list(self.params.keys())

        self._make_model_params()
        self.voxel_space = 0.01
        self.voxel_model = np.zeros((3,3,3),dtype = self.dtype)
        self.model_shape = (3,3,3)

    def _make_voxel_model(self,model):
        # バイナリーモデルでは、骨梁間隙を0,海綿骨を1、緻密骨を2、皮膚を３、領域外を-1に設定する
        self.voxel_model = model
        del model
        gc.collect
        if self.params['th_ct'] !=0:
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

    def getAbsorptionCoeff(self,add):
        # パラメーターは、[骨梁間隙, 海綿骨, 緻密骨, 皮膚, 外気]
        val = self.voxel_model[add[0],add[1],add[2]].astype('float32')
        val[val>0]=self.ma[1]
        val[val==0]=self.ma[0]
        val[val==self.cort_num]=self.ma[2]
        val[val==self.skin_num]=self.ma[3]
        return val

    def getScatteringCoeff(self,add):
        val = self.voxel_model[add[0],add[1],add[2]].astype('float32')
        #val[val>0]=1.6676*1e-3*val[val>0]*2**8-10.932 #海綿骨と緻密骨を直線近似
        #val[val>0]=1.55*1e-3*val[val>0]*2**8-10.6 #若森モデル
        val[val>0]=8.46*1e-4*val[val>0]*2**8+6.25 #緻密のみを直線近似
        #val[val>0]=3.92*1e-8*(val[val>0]*2**8)**2+2.77*1e-3*(val[val>0]*2**8)-17.5#海綿骨と緻密骨を2次近似

        val[val==0]=self.ms[0]
        val[val==self.cort_num]=self.ms[2]
        val[val==self.skin_num]=self.ms[3]
        return val

    def getAnisotropyCoeff(self,add):
        val = self.voxel_model[add[0],add[1],add[2]].astype('float32')
        val[val > 0 ]=self.g[1]
        val[val== 0 ]=self.g[0]
        val[val==self.cort_num]=self.g[2]
        val[val==self.skin_num]=self.g[3]
        return val

    def getReflectiveIndex(self,add):
        # パラメーターは、[骨梁間隙, 海綿骨, 緻密骨, 皮膚, 外気]のように設定されています。
        val = self.voxel_model[add[0],add[1],add[2]].astype('float32')
        val[val > 0 ]=self.n[1]
        val[val== 0 ]=self.n[0]
        val[val== -1]=self.n[4]
        val[val==self.cort_num]=self.n[2]
        val[val==self.skin_num]=self.n[3]
        return val


class PlateModel(VoxelModel):
    @_deprecate_positional_args
    def __init__(
        self,*,thickness=[0.2,] ,xy_size=[0.1,0.1],voxel_space = 0.1,
        ma=[1,],ms=[100,],g=[0.9,],n=[1.37,],n_air=1,f = 'float32'):
        self.model_name = 'PlateModel'
        self.thickness = thickness
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
        nx_box = np.round(self.xy_size[0]/self.voxel_space).astype(int)
        ny_box = np.round(self.xy_size[1]/self.voxel_space).astype(int)
        nz_box = np.round(self.borderposit/self.voxel_space).astype(int)
        self.voxel_model = np.zeros((nx_box+2,ny_box+2,nz_box[-1]+2),dtype = 'int8')
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
        self.thickness = thickness
        self.borderposit = self._make_borderposit(thickness,f)
        self._make_voxel_model()

        self.n =np.array(n+[n_air]).astype(f)
        self.ms = np.array(ms).astype(f)
        self.ma = np.array(ma).astype(f)
        self.g = np.array(g).astype(f)
        self.getModelSize()

    def get_params(self):
        return {
                'th':self.thickness,
                'ms':self.ms,
                'ma':self.ma,
                'n':self.n,
                'g':self.g
                }

    def getModelSize(self):
        print("Memory area size for voxel storage: %0.3f Mbyte" % (self.voxel_model.nbytes*1e-6))

class WhiteNoiseModel(VoxelModel):
    def __init__(self):
        self.model_name = 'WhiteNoiseModel'
        self.dtype = 'int8'
        self.dtype_f = 'float32'
        self.ct_num=2
        self.subc_num=3
        self.skin_num=4
        self.params = {
            'xy_size':[40,40],'voxel_space':0.01,'bv_tv':0.138,'symmetrization':False,
            'th_trabecular':40,'th_cortical':1.,'th_subcutaneus':2.6,'th_dermis':1.4,
            'n_space':1.,'n_trabecular':1.4,'n_cortical':1.4,'n_subcutaneus':1.4,'n_dermis':1.4,'n_air':1.,
            'ma_space':1e-8,'ma_trabecular':0.02374,'ma_cortical':0.02374,'ma_subcutaneus':0.011,'ma_dermis':0.037,
            'ms_space':1e-8,'ms_trabecular':20.54,'ms_cortical':17.67,'ms_subcutaneus':20,'ms_dermis':20,
            'g_space':0.90,'g_trabecular':0.90,'g_cortical':0.90,'g_subcutaneus':0.90,'g_dermis':.90,
            }
        self.keys = list(self.params.keys())

        self._make_model_params()
        self.bmd = self._get_bone_vbmd()
        self.voxel_space = self.params['voxel_space']
        self.voxel_model = np.zeros((3,3,3),dtype = self.dtype)
        self.model_shape = (3,3,3)

    def _get_bone_vbmd(self):
        ms_ugru = self.params['ms_trabecular']*(1-self.params['g_trabecular'])
        ms_ugru = ms_ugru/(1-0.93)
        bmd_wet = 0.0382*ms_ugru+0.8861
        bmd_ash = 0.9784*bmd_wet - 0.8026
        return bmd_ash*self.params['bv_tv']*1000


    def build(self):
        #thickness,xy_size,voxel_space,ma,ms,g,n,n_air
        del self.voxel_model
        gc.collect()

        self.voxel_space = self.params['voxel_space']
        self._make_voxel_model()
        self.getModelSize()


    def set_params(self,*initial_data, **kwargs):
        set_params(self.params,self.keys,*initial_data, **kwargs)
        self._make_model_params()
        self.bmd = self._get_bone_vbmd()
        print('Trabecular vBMD = %4f [mg/cm^3]'%self.bmd)

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

    def _make_voxel_model(self):
        #骨梁間隙を0,海綿骨を1、緻密骨を2、 皮下組織を3、皮膚を4、領域外を-1に設定する
        self.xyz_size = [
            int(self.params['xy_size'][0]/self.voxel_space),
            int(self.params['xy_size'][1]/self.voxel_space),
            int(self.params['th_trabecular']/self.voxel_space)
        ]
        self.voxel_model = np.zeros(
            self.xyz_size[0]*self.xyz_size[1]*self.xyz_size[2]
            ,dtype = self.dtype)

        index = np.where(
            np.random.rand(
            self.xyz_size[0]*self.xyz_size[1]*self.xyz_size[2]
            )<=self.params['bv_tv'])[0]

        self.voxel_model[index] = 1
        self.voxel_model = self.voxel_model.reshape(
            self.xyz_size[0],self.xyz_size[1],self.xyz_size[2])

        if self.params['th_cortical'] !=0:
            ct = np.ones((self.xyz_size[0],
                         self.xyz_size[1],
                         int(self.params['th_cortical']/self.voxel_space)),
                         dtype = self.dtype)*self.ct_num
            self.voxel_model = np.concatenate((ct.T,self.voxel_model.T)).T
            if self.params['symmetrization']:
                self.voxel_model = np.concatenate((self.voxel_model.T,ct.T)).T

        if self.params['th_subcutaneus'] !=0:
            subc = np.ones((self.xyz_size[0],
                         self.xyz_size[1],
                         int(self.params['th_subcutaneus']/self.voxel_space)),
                         dtype = self.dtype)*self.subc_num
            self.voxel_model = np.concatenate((subc.T,self.voxel_model.T)).T
            if self.params['symmetrization']:
                self.voxel_model = np.concatenate((self.voxel_model.T,subc.T)).T

        if self.params['th_dermis'] != 0:
            skin = np.ones((self.xyz_size[0],
                           self.xyz_size[1],
                           int(self.params['th_dermis']/self.voxel_space)+1),
                           dtype = self.dtype)*self.skin_num
            self.voxel_model = np.concatenate((skin.T,self.voxel_model.T)).T
            if self.params['symmetrization']:
                self.voxel_model = np.concatenate((self.voxel_model.T,skin.T)).T

        self.voxel_model[0,:,:] = -1
        self.voxel_model[-1,:,:] = -1
        self.voxel_model[:,0,:] = -1
        self.voxel_model[:,-1,:] = -1
        self.voxel_model[:,:,0] = -1
        self.voxel_model[:,:,-1] = -1

    def getModelSize(self):
        print("Memory area size for voxel storage: %0.3f Mbyte" % (self.voxel_model.nbytes*1e-6))

class TuringModel(VoxelModel):
    def __init__(self):
        self.model_name = 'WhiteNoiseModel'
        self.dtype = 'int8'
        self.dtype_f = 'float32'
        self.ct_num=2
        self.subc_num=3
        self.skin_num=4
        self.params = {
            'xy_size':[40,40],'voxel_space':0.01,'dicom_path':"dicom_p085",'bv_tv':0.138,
            'th_trabecular':40,'th_cortical':1.,'th_subcutaneus':2.6,'th_dermis':1.4,
            'n_space':1.,'n_trabecular':1.4,'n_cortical':1.4,'n_subcutaneus':1.4,'n_dermis':1.4,'n_air':1.,
            'ma_space':1e-8,'ma_trabecular':0.02374,'ma_cortical':0.02374,'ma_subcutaneus':0.011,'ma_dermis':0.037,
            'ms_space':1e-8,'ms_trabecular':20.54,'ms_cortical':17.67,'ms_subcutaneus':20,'ms_dermis':20,
            'g_space':0.90,'g_trabecular':0.90,'g_cortical':0.90,'g_subcutaneus':0.90,'g_dermis':.90,
            }
        self.keys = list(self.params.keys())

        self._make_model_params()
        self.bmd = self._get_bone_vbmd()
        self.voxel_space = self.params['voxel_space']
        self.voxel_model = np.zeros((3,3,3),dtype = self.dtype)
        self.model_shape = (3,3,3)

    def _get_bone_vbmd(self):
        ms_ugru = self.params['ms_trabecular']*(1-self.params['g_trabecular'])
        ms_ugru = ms_ugru/(1-0.93)
        bmd_wet = 0.0382*ms_ugru+0.8861
        bmd_ash = 0.9784*bmd_wet - 0.8026
        return bmd_ash*self.params['bv_tv']*1000


    def build(self):
        #thickness,xy_size,voxel_space,ma,ms,g,n,n_air
        del self.voxel_model
        gc.collect()

        self.voxel_space = self.params['voxel_space']
        self._make_voxel_model()
        self.getModelSize()


    def set_params(self,*initial_data, **kwargs):
        set_params(self.params,self.keys,*initial_data, **kwargs)
        self._make_model_params()
        self.bmd = self._get_bone_vbmd()
        print('Trabecular vBMD = %4f [mg/cm^3]'%self.bmd)

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

    def _read_dicom(self):
        path = self.params['dicom_path']
        files = os.listdir(path)
        files.sort()
        self.params['voxel_space'] = round(float(pydicom.dcmread(path+"/"+files[0],force=True).PixelSpacing[0]),5)

        ds = []
        for i in files:
            ds.append(pydicom.dcmread(path+"/"+i,force=True).pixel_array)
        ds = np.array(ds).astype("int8")
        self.xyz_size = [
        ds.shape[0],ds.shape[1],ds.shape[2]
        ]

        self.params['xy_size'][0] = self.xyz_size[0]*self.params['voxel_space']
        self.params['xy_size'][1] = self.xyz_size[1]*self.params['voxel_space']
        self.params['th_trabecular'] = self.xyz_size[2]*self.params['voxel_space']

        return ds

    def _make_voxel_model(self):
        #骨梁間隙を0,海綿骨を1、緻密骨を2、 皮下組織を3、皮膚を4、領域外を-1に設定する
        self.voxel_model = self._read_dicom()

        if self.params['th_cortical'] !=0:
            ct = np.ones((self.xyz_size[0],
                         self.xyz_size[1],
                         int(self.params['th_cortical']/self.voxel_space)),
                         dtype = self.dtype)*self.ct_num
            self.voxel_model = np.concatenate((ct.T,self.voxel_model.T)).T

        if self.params['th_subcutaneus'] !=0:
            ct = np.ones((self.xyz_size[0],
                         self.xyz_size[1],
                         int(self.params['th_subcutaneus']/self.voxel_space)),
                         dtype = self.dtype)*self.subc_num
            self.voxel_model = np.concatenate((ct.T,self.voxel_model.T)).T

        if self.params['th_dermis'] != 0:
            skin = np.ones((self.xyz_size[0],
                           self.xyz_size[1],
                           int(self.params['th_dermis']/self.voxel_space)+1),
                           dtype = self.dtype)*self.skin_num
            self.voxel_model = np.concatenate((skin.T,self.voxel_model.T)).T

        self.voxel_model[0,:,:] = -1
        self.voxel_model[-1,:,:] = -1
        self.voxel_model[:,0,:] = -1
        self.voxel_model[:,-1,:] = -1
        self.voxel_model[:,:,0] = -1
        self.voxel_model[:,:,-1] = -1

    def getModelSize(self):
        print("Memory area size for voxel storage: %0.3f Mbyte" % (self.voxel_model.nbytes*1e-6))

class SeparatedPlateModel(PlateModel):
    def __init__(
        self,*,thickness=[0.2,] ,xy_size=[0.1,0.1],voxel_space = 0.1,
        ma=[1,],ms=[100,],g=[0.9,],n=[1.37,],n_air=1,n_front=1,n_back=1,f = 'float32'):
        self.model_name = 'SeparatedPlateModel'
        self.thickness = thickness
        self.n =np.array(n+[n_back,n_front,n_air]).astype(f)
        self.ms = np.array(ms+[0,0,0]).astype(f)
        self.ma = np.array(ma+[0,0,0]).astype(f)
        self.g = np.array(g+[0,0,0]).astype(f)
        self.voxel_space = voxel_space
        self.xy_size = xy_size
        self.borderposit = self._make_borderposit(thickness,f)
        self._make_voxel_model()

    def _make_voxel_model(self):
        nx_box = np.round(self.xy_size[0]/self.voxel_space).astype('int')
        ny_box = np.round(self.xy_size[1]/self.voxel_space).astype('int')
        nz_box = np.round(self.borderposit/self.voxel_space).astype('int')
        self.voxel_model = np.zeros((nx_box+2,ny_box+2,nz_box[-1]+2),dtype = 'int8')
        for i in range(nz_box.size-1):
            self.voxel_model[:,:,nz_box[i]+1:nz_box[i+1]+1] = i
        self.voxel_model[0] = -1;self.voxel_model[-1] = -1
        self.voxel_model[:,0] = -1; self.voxel_model[:,-1] = -1
        self.voxel_model[:,:,0] = -2; self.voxel_model[:,:,-1] = -3

    @_deprecate_positional_args
    def build(self,
        thickness,
        xy_size,
        voxel_space,
        ma,ms,g,
        n,n_air,n_front,n_back,
        f = 'float32'):

        del self.voxel_model
        gc.collect()
        #-1はモデルの外側
        self.thickness = thickness
        self.voxel_space = voxel_space
        self.xy_size = xy_size
        self.borderposit = self._make_borderposit(thickness,f)
        self._make_voxel_model()

        self.n =np.array(n+[n_back,n_front,n_air]).astype(f)
        self.ms = np.array(ms+[0,0,0]).astype(f)
        self.ma = np.array(ma+[0,0,0]).astype(f)
        self.g = np.array(g+[0,0,0]).astype(f)
        self.getModelSize()


# =============================================================================
# Public montecalro model
# =============================================================================

class VoxelWhiteNoiseModel(BaseVoxelMonteCarlo):
    def __init__(
        self,*,nPh=1000,fluence_mode=False,dtype='float32',
        nr=50,nz=20,dr=0.1,dz=0.1,w_beam =0,
        z_max_mode = False,
        beam_angle = 0,initial_refrect_by_angle = False,
        wavelength = 850,beam_posision = 10,
        lens_curvature_radius = 51.68,grass_type = 'N-BK7',
    ):
        super().__init__(
            nPh = nPh,fluence_mode =fluence_mode, model = WhiteNoiseModel(),
            dtype='float32',nr=nr,nz=nz,dr=dr,dz=dz,z_max_mode = z_max_mode,
            w_beam=w_beam,beam_angle = beam_angle,
            initial_refrect_by_angle = initial_refrect_by_angle,
            wavelength = wavelength,
            beam_posision = beam_posision,
            lens_curvature_radius = lens_curvature_radius,
            grass_type = grass_type,
        )
    def build(self,*initial_data, **kwargs):
        if initial_data == () and kwargs == {}:
            pass
        else:
            self.model.set_params(*initial_data, **kwargs)
        self.model.build()
        """
        try:
            self.model.build()
        except:
            warnings.warn('New voxel_model was not built')"""

    def set_params(self,*initial_data, **kwargs):
        self.model.set_params(*initial_data, **kwargs)

    def get_model_fig(self,*,dpi=300,save_path = False,):
        image = self.model.voxel_model
        resol0 = (image.shape[0]+1)*self.model.params['voxel_space']/2-\
        np.array([self.model.params['voxel_space']*i for i in range(image.shape[1]+1)])

        resol2 = np.array([self.model.params['voxel_space']*i for i in range(image.shape[2]+1)])

        plt.figure(figsize=(5,5),dpi=100)
        plt.set_cmap(plt.get_cmap('gray'))
        plt.pcolormesh(resol0,resol2,image[:,int(image.shape[1]/2),:].T)
        plt.xlabel('X [mm]')
        plt.ylabel('Z [mm]')
        plt.ylim(resol2[-1],resol2[0])
        if save_path:
            plt.savefig(
                save_path,
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
            'calc_dtype':self.dtype,
            'model':{
                'model_name':self.model.model_name,
                'model_params':self.model.params,
                'model_bmd':self.model.bmd,
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
        self,*,nPh=1000,fluence_mode=False,dtype='float32',
        nr=50,nz=20,dr=0.1,dz=0.1,w_beam =0,
        z_max_mode = False,
        beam_angle = 0,initial_refrect_by_angle = False,
        wavelength = 850,beam_posision = 10,
        lens_curvature_radius = 51.68,grass_type = 'N-BK7',
    ):
        super().__init__(
            nPh = nPh,fluence_mode =fluence_mode, model = TuringModel(),
            dtype='float32',nr=nr,nz=nz,dr=dr,dz=dz,z_max_mode = z_max_mode,
            w_beam=w_beam,beam_angle = beam_angle,
            initial_refrect_by_angle = initial_refrect_by_angle,
            wavelength = wavelength,
            beam_posision = beam_posision,
            lens_curvature_radius = lens_curvature_radius,
            grass_type = grass_type,
        )
    def build(self,*initial_data, **kwargs):
        if initial_data == () and kwargs == {}:
            pass
        else:
            self.model.set_params(*initial_data, **kwargs)
        self.model.build()
        """
        try:
            self.model.build()
        except:
            warnings.warn('New voxel_model was not built')"""

    def set_params(self,*initial_data, **kwargs):
        self.model.set_params(*initial_data, **kwargs)

    def get_model_fig(self,*,dpi=300,save_path = False,):
        image = self.model.voxel_model
        resol0 = (image.shape[0]+1)*self.model.params['voxel_space']/2-\
        np.array([self.model.params['voxel_space']*i for i in range(image.shape[1]+1)])

        resol2 = np.array([self.model.params['voxel_space']*i for i in range(image.shape[2]+1)])

        plt.figure(figsize=(5,5),dpi=100)
        plt.set_cmap(plt.get_cmap('gray'))
        plt.pcolormesh(resol0,resol2,image[:,int(image.shape[1]/2),:].T)
        plt.xlabel('X [mm]')
        plt.ylabel('Z [mm]')
        plt.ylim(resol2[-1],resol2[0])
        if save_path:
            plt.savefig(
                save_path,
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
            'calc_dtype':self.dtype,
            'model':{
                'model_name':self.model.model_name,
                'model_params':self.model.params,
                'model_bmd':self.model.bmd,
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



class VoxelPlateModel(BaseVoxelMonteCarlo):
    def __init__(
        self,*,nPh=1000,fluence_mode=False,dtype='float32',
        nr=50,nz=20,dr=0.1,dz=0.1,w_beam =0,
        z_max_mode = False,
        beam_angle = 0,initial_refrect_by_angle = False,
        wavelength = 850,beam_posision = 10,
        lens_curvature_radius = 51.68,grass_type = 'N-BK7',
    ):

        super().__init__(
            nPh = nPh,fluence_mode =fluence_mode, model = PlateModel(),
            dtype='float32',nr=nr,nz=nz,dr=dr,dz=dz,z_max_mode = z_max_mode,
            w_beam=w_beam,beam_angle = beam_angle,
            initial_refrect_by_angle = initial_refrect_by_angle,
            wavelength = wavelength,
            beam_posision = beam_posision,
            lens_curvature_radius = lens_curvature_radius,
            grass_type = grass_type,
        )



class VoxelSeparatedPlateModel(BaseVoxelMonteCarlo):
    #時に光入射面と透過面の屈折率を変更したい場合が生じる。
    #その時は、このモデルを用いる。
    #SeparatedPlateModelは、光入射面と透過面の屈折率を独立して設定することが可能である。
    def __init__(self,*,nPh=1000,fluence_mode=False,dtype='float32',
                 nr=50,nz=20,dr=0.1,dz=0.1,w_beam = 0,
                 beam_angle = 0,
                 beam_type = 'TEM00',beam_dist=False,
                 beam_direct='positive',intermediate_buffer=False):

        super().__init__(nPh = nPh,fluence_mode =fluence_mode, model = SeparatedPlateModel(),
                         dtype='float32',nr=nr,nz=nz,dr=dr,dz=dz,w_beam=w_beam,beam_angle = beam_angle,
                         beam_type = beam_type,beam_dist=beam_dist,beam_direct=beam_direct,
                         intermediate_buffer=intermediate_buffer)




class VoxelPlateLedModel(BaseVoxelMonteCarlo):
    def __init__(self,*,nPh=1000,fluence_mode=False,dtype='float32',
                 nr=50,nz=20,dr=0.1,dz=0.1,):

        super().__init__(nPh = nPh,fluence_mode =fluence_mode, model = PlateModel(),
                         dtype='float32',nr=nr,nz=nz,dr=dr,dz=dz)
        self.led_params = {
            'h':2.5,
            'nx':5,
            'ny':5,
            'dx':1.2,
            'dy':0.298,
        }
        self.led_keys = list(self.led_params.keys())

    def set_led_params(self,*initial_data, **kwargs):
        set_params(self.led_params,self.led_keys,*initial_data, **kwargs)


    def _set_led(self):
        l = self.model.voxel_space
        a= []
        for i in range(self.led_params['nx']):
            for j in range(self.led_params['ny']):
                a.append([i,j,0])
        a = np.array(a).astype('float32').T
        a[0]*=self.led_params['dx'];a[1]*=self.led_params['dy']
        a[0]-=a[0].max()/2;a[1]-=a[1].max()/2
        b = a.copy()
        b[2] = self.led_params['h']
        b = b/np.linalg.norm(b,axis=0)
        ni = self.model.n[-1]
        nt = self.model.n[0]
        b[:2] *= nt/ni
        b[2] = (1-(1-b[2]**2)*(ni/nt)**2)**0.5
        n_agrid = int(self.nPh/(self.led_params['nx']*self.led_params['ny']))
        self.nPh = n_agrid*self.led_params['nx']*self.led_params['ny']

        gp = np.zeros((2,self.nPh))
        gv = np.zeros((3,self.nPh))
        for i in range(self.led_params['nx']*self.led_params['ny']):
            for j in range(3):
                if j !=2:
                    gp[j,n_agrid*i:n_agrid*(i+1)] = a[j,i]
                    gv[j,n_agrid*i:n_agrid*(i+1)] = b[j,i]
                else:
                    gv[j,n_agrid*i:n_agrid*(i+1)] = b[j,i]

        #各アドレスに振り分ける
        pp = (gp/l).astype("int16")
        ind = np.where(gb<0)
        pp[ind[0].tolist(),ind[1].tolist()] = \
            pp[ind[0].tolist(),ind[1].tolist()]-1
        pa = gp - pp*l -l/2
        ind = np.where((np.abs(pa)>=l/2))
        pa[ind[0].tolist(),ind[1].tolist()] = \
            np.sign(pa[ind[0].tolist(),ind[1].tolist()])*(l/2)

        self.add[:2] = self.add[:2] + pp
        self.p[:2] = pa.astype(self.dtype)
        self.v = gv

class VoxelDicomModel(BaseVoxelMonteCarlo):

    def __init__(
        self,*,nPh=1000,dtype='float32',
        nr=50,nz=20,dr=0.1,dz=0.1, fluence_mode=False,
        model_type = 'binary',w_beam = 0,beam_angle=0,
        initial_refrect_by_angle = False,
        wavelength = 850,beam_posision = 10,
        lens_curvature_radius = 51.68,grass_type = 'N-BK7',
    ):

        self.model_type = model_type
        model = self._model_select(self.model_type)
        self.model = model

        super().__init__(
            nPh = nPh,fluence_mode =fluence_mode,model = model,
            dtype='float32',nr=nr,nz=nz,dr=dr,dz=dz,
            beam_angle=beam_angle,
            initial_refrect_by_angle = initial_refrect_by_angle,
            wavelength = wavelength,
            beam_posision = beam_posision,
            lens_curvature_radius = lens_curvature_radius,
            grass_type = grass_type,

        )

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
        if initial_data == () and kwargs == {}:
            pass
        else:
            self.model.set_params(*initial_data, **kwargs)
        try:
            self.model.build(self.array_dicom,self.ConstPixelSpacing[0])
            del self.array_dicom
            gc.collect()
        except:
            warnings.warn('New voxel_model was not built')

    def set_params(self,*initial_data, **kwargs):
        self.model.set_params(*initial_data, **kwargs)

    def _model_select(self,model_type):
        if model_type == 'binary':
            model = DicomBinaryModel()
        elif model_type == 'linear':
            model = DicomLinearModel()
        return model

    def set_monte_params(self,*,nPh,dtype='float32',
                 nr=50,nz=20,dr=0.1,dz=0.1, fluence_mode=False,
                 model_type = 'binary',w_beam = 0):
        self.dtype = dtype
        self.nPh = nPh
        self.w_beam = w_beam
        self.fluence = fluence_mode
        self.fluence_mode = fluence_mode

        if self.fluence:
            if fluence_mode == '2D':
                self.fluence = Fluence2D(nr=nr,nz=nz,dr=dr,dz=dz)
            elif fluence_mode == '3D':
                self.fluence = Fluence3D(nr=nr,nz=nz,dr=dr,dz=dz)
        if model_type != 'keep':
            params = self.model.params
            self.model_type = model_type
            model = self._model_select(self.model_type)
            self.model = model
            self.model.set_params(params)

    def _calc_info(self,coment=''):
        calc_info = {
            'Date':datetime.datetime.now().isoformat(),
            'coment':coment,
            'number_of_photons':self.nPh,
            'calc_dtype':self.dtype,
            'model':{
                'model_type':self.model_type,
                'model_name':self.model.model_name,
                'model_params':self.model.params,
                'model_shape':self.model.model_shape,
                'model_dtype':self.model.dtype.name,
            },
            'dicom':{
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
                'w_beam':self.w_beam,
                'beam_angle':self.beam_angle,
                'initial_refrection':self.initial_refrect_by_angle,
                'beam_mode':'TEM00',
                'fluence_mode':self.fluence_mode,
            }
        }
        if self.beam_angle_mode:
            calc_info['wavelength'] = self.wavelength
            calc_info['beam_posision'] = self.beam_posision
            calc_info['lens_curvature_radius'] = self.lens_curvature_radius
            calc_info['grass_type'] = self.grass_type
        return calc_info

    def save_result(self,fname,coment=''):
        start_ = time.time()

        res = self.get_result()
        save_name = fname+"_LID.pkl.bz2"
        with bz2.open(save_name, 'wb') as fp:
            fp.write(pickle.dumps(res))
        print("Monte Carlo results saved in ")
        print("-> %s" %(save_name))
        print('')
        info = self._calc_info(coment)
        save_name = fname+"_info.json"
        with open(save_name, 'w') as fp:
            json.dump(info,fp,indent=4)
        print("Calculation conditions are saved in")
        print("-> %s" %(save_name))
        print('')
        if self.fluence_mode:
            fe = self.fluence.getArz()
            save_name = fname+'_fluence'+self.fluence_mode+".pkl.bz2"
            with bz2.open(save_name, 'wb') as fp:
                fp.write(pickle.dumps(fe))
            print("Internal Fluence saved in ")
            print("-> %s" %(save_name))
            print('')

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
        self.trimd_size =ConstPixelDims
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
                    ax[1,1].hist(image[xx,:,].flatten(),bins=50, color='c')
                elif hist_type == 'XZ':
                    ax[1,1].set_title('Histogram of X-Z pic pixel values')
                    ax[1,1].hist(image[:,yy,].flatten(),bins=50, color='c')
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

    def set_trim_pixel(self,*, right = 0, left = 0,
                  upper ,lower =0,
                  top = 0,bottom = 0,):
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
                self.threshold = int(threshold)
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
                              xx = int(self.array_dicom.shape[1]/2),
                              yy = int(self.array_dicom.shape[0]/2),
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

    def check_threshold(self,threshold=37,cmap = 'gray',zz = 0,graph_type = 'XY'):
        image = reConstArray(self.array_dicom,threshold)
        self.display_cross_section(image = image,zz = zz,
                              xx = int(image.shape[0]/2),
                              yy = int(image.shape[1]/2),
                              cmap = cmap,graph_type = graph_type)
        self.threshold = int(threshold*2**8)

    def set_threshold(self,*,threshold=37,cmap = 'gray',zz = 0, graph_type = 'XY'):
        self.array_dicom = reConstArray(self.array_dicom,threshold)
        self.display_cross_section(zz = zz,
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
