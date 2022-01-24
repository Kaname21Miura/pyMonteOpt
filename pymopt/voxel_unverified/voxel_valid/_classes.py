#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 16:56:05 2021
@author: Kaname Miura
"""
from ._cukernel import vmc_kernel

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
'VoxelPlateModel','VoxelTuringModel'
]
from numba import njit
@njit(fastmath=True)
def theta(g,rand):
    th = 0
    if g != 0.:
        th = (1 + g**2 - ((1 - g**2) / (1 - g + 2 * g * rand))** 2) / (2 * g)
        if th < -1:
            th = -1
    else:
        th = 2 * rand - 1
    return th

@njit(fastmath=True)
def _movement(
    add, p,v, w, ma, ms, n, g,
    voxel_model, l,
    nPh, end_point
    ):

    ma = ma.astype(np.float32)
    ms = ms.astype(np.float32)
    n = n.astype(np.float32)
    g = g.astype(np.float32)
    nPh = np.int32(nPh)
    out_index =np.arange(nPh)

    for idx in range(nPh):
        add_ = np.array([0,0,0]).astype(np.int32)
        v_ = np.array([0,0,0]).astype(np.float32)
        zero_vec = np.array([0,0,0]).astype(np.int32)
        one_vec = np.array([1,1,1]).astype(np.int32)

        wth = np.float32(0.0001)
        roulette_m = np.int32(10)
        index_ = voxel_model[add[0,idx], add[1,idx], add[2,idx]]
        index_next = np.int8(0)
        index_end = np.int8(end_point)

        fi = np.float32(0);
        cos_fi = np.float32(0); cos_th = np.float32(0)
        sin_fi = np.float32(0); sin_th = np.float32(0);

        valf = np.float32(0.); dbnum = np.int32(0); db = np.float32(1000.)

        ni = np.float32(n[index_])
        nt = np.float32(0)
        mt = np.float32(ma[index_] + ms[index_])
        st = np.float32(0)
        flag_tr = True
        flag_end = False

        st = np.float32(-np.log(np.random.rand()))
        while True:
            valf = 0.; dbnum = 0; db = 1000.
            for i in range(3):
                if (np.abs(v[i,idx]) > 0):
                    valf = (l / 2 - np.sign(v[i,idx]) * p[i,idx])/ np.abs(v[i,idx])
                    if (valf < db):
                        dbnum = i; db = valf
            if st>= db*mt:
                for i in range(3):
                    p[i,idx]+=v[i,idx]*db
                p[dbnum,idx] = l/2*np.sign(v[dbnum,idx])
                st-=db*mt
                for i in range(3):
                    add_[i] = add[i,idx]
                add_[dbnum] += np.sign(v[dbnum,idx])
                index_next = voxel_model[add_[0], add_[1], add_[2]]

                nt = n[index_next]
                flag_tr = True

                if (ni != nt):
                    ra = 0;at = 0
                    ai = np.arccos(abs(v[dbnum,idx]))
                    if (ni > nt)&(ai >= np.arcsin(nt/ni)):
                        ra = 1
                    else:
                        at = np.arcsin((ni/nt)*np.sin(ai))
                        if ai!=0:
                            ra = ((np.sin(ai - at) / np.sin(ai + at))**2 \
                            +(np.tan(ai - at) / np.tan(ai + at))**2)/2
                        else:
                            ra = 0
                    if ra < np.random.rand():
                        flag_tr = True
                        zero_vec[dbnum] = 1; one_vec[dbnum] = 0
                        for i in range(3):
                            v[i,idx] = one_vec[i]*v[i,idx]*ni/nt\
                            +zero_vec[i] * np.sign(v[i,idx]) * np.cos(at)
                            zero_vec[i] = 0; one_vec[i] = 1

                        valf = np.sqrt(v[0,idx]**2+v[1,idx]**2+v[2,idx]**2)
                        for i in range(3):
                            v[i,idx] /= valf
                    else:
                        flag_tr = False

                if flag_tr:
                    add[dbnum,idx] += np.sign(v[dbnum,idx])
                    p[dbnum,idx] *= -1
                    index_ = index_next
                    if index_ == index_end:
                        flag_end = True
                        out_index[idx]+=1
                        break
                    mt = ma[index_] + ms[index_]
                    ni = nt
                else:
                    v[dbnum,idx] *= -1
            else:
                for i in range(3):
                    p[i,idx]+=v[i,idx]*st/mt
                st = 0
                break

        if flag_end:
            continue

        w[idx] -= np.float32(w[idx]*ma[index_]/mt)

        if(w[idx]<=wth):
            if (1/roulette_m) < np.float32(np.random.rand()):
                for i in range(3):
                    p[i,idx] = 0
                    v[i,idx] = 0
                    add[i,idx] = 0
                w[idx] = 0
                out_index[idx]+=1
                continue
            else:
                w[idx]*=roulette_m

        cos_th = theta(g[index_], np.float32(np.random.rand()))
        sin_th = np.sqrt(1-cos_th**2)

        fi = 2 * 3.1415927*np.float32(np.random.rand())
        cos_fi = np.cos(fi)
        sin_fi = np.sin(fi)

        if 0.99999 < abs(v[2,idx]):
            v[0,idx] = sin_th * cos_fi
            v[1,idx] = sin_th * sin_fi
            v[2,idx] = np.sign(v[2,idx]) * cos_th

        else:
            valf = np.sqrt(1 - v[2,idx]**2)
            for i in range(3):
                v_[i] = v[i,idx]
            v[0,idx] = sin_th * (v_[0] * v_[2] * cos_fi - v_[1] * sin_fi) / valf + v_[0] * cos_th
            v[1,idx] = sin_th * (v_[1] * v_[2] * cos_fi + v_[0] * sin_fi) / valf + v_[1] * cos_th
            v[2,idx] = -sin_th * cos_fi * valf + v_[2] * cos_th

        valf = np.sqrt(v[0,idx]**2+v[1,idx]**2+v[2,idx]**2)

        for i in range(3):
            v[i,idx] /= valf

    return v,p,add,w,out_index

# =============================================================================
# Base solid model
# =============================================================================

class BaseVoxelMonteCarlo(MonteCalro,metaclass = ABCMeta):
    #@_deprecate_positional_args
    @abstractmethod
    def __init__(self,*,nPh,model,dtype='float32',
                 beam_type = 'TEM00',w_beam = 0,
                 beam_angle = 0,initial_refrect_by_angle = False,
                 first_layer_clear = False,
                 threadnum = 128,
                 ):
        super().__init__()

        def __check_list_name(name,name_list):
            if not(name in name_list):
                raise ValueError('%s is not a permitted for factor. Please choose from %s.'%(name,name_list))

        self.beam_type_list=['TEM00',False]
        __check_list_name(beam_type,self.beam_type_list)
        self.beam_type = beam_type

        self.dtype = dtype
        self.nPh = nPh
        self.w_beam = w_beam
        self.contnum = 0
        self.initial_refrect_by_angle = initial_refrect_by_angle
        self.beam_angle = beam_angle

        self.model = model
        self.first_layer_clear=first_layer_clear
        if threadnum > 1024:
            raise Exception('There are too many threads. threadnum < 1024.')
        else:
            self.threadnum = int(threadnum)

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
        super().start()

        self._end_process()
        print("###### End ######")
        self.getRdTtRate()
        calTime(time.time(), start_)
        #del func
        return self

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

        # nanデータを削除
        if np.isnan(v).any():
            print('Nan occurs in vector v')
            del_index = np.where(np.isnan(v)[0])[0]
            print("v",v[:,del_index])
            print("p",p[:,del_index])
            print("add",add[:,del_index])
            print("w",w[del_index])
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
    """
    def _border_act(self,v,p,add,s,db,mt):
        l = self.model.voxel_space
        A = self.create01Array(np.argmin(db,axis=0),m=3)
        B = ~A

        sigAv = A*np.sign(v).astype(np.float32)
        ds = np.min(db,axis=0).astype(np.float32)
        s -= ds*mt
        #p = l/2*A*np.sign(v)+B*(p+v*ds)
        p = (l/2*sigAv+B*(p+v*ds)).astype(np.float32)

        ni = self.model.getReflectiveIndex(add).astype(np.float32)
        nt = self.model.getReflectiveIndex(add+sigAv.astype("int16")).astype(np.float32)
        pb_index = np.where(ni!=nt)[0]

        if list(pb_index) != []:
            #透過判別
            #入射角の計算
            ai_ = np.abs(A[:,pb_index]*v[:,pb_index])
            ai = np.arccos(ai_[ai_!=0]).astype(np.float32)#Aから算出される0を排除して1次元配列にする\
            Ra = np.zeros_like(ai).astype(np.float32)

            #全反射するときのインデックスを取得
            sub_index = np.where((nt[pb_index]<ni[pb_index])&(ai>=np.arcsin(nt[pb_index]/ni[pb_index])))[0]

            #屈折角の計算
            ##(ai,at)=(0,0)の時は透過させ方向ベクトルは変化させない
            ###今の状態だと、subsub_indexもarcsinで計算しているため、しょうがないけどRuntimeWarningが出ます
            at = np.arcsin((ni[pb_index]/nt[pb_index])*np.sin(ai))
            Ra = (((np.sin(ai - at) / np.sin(ai + at))**2 \
            +(np.tan(ai - at) / np.tan(ai + at))**2)/2).astype(np.float32)
            #全反射はRaを強制的に0以下にし反射させる
            Ra[sub_index] = 1.
            Ra[np.where(ai==0)[0]]=0.
            rand_ = np.random.rand(ai.size).astype(np.float32)
            #透過と反射のindexを取得
            vl_index = pb_index[np.where(Ra>=rand_)[0]]#反射
            self.contnum+=vl_index.size
            sub_index = np.where(Ra<rand_)[0]#透過
            vt_index = pb_index[sub_index]#全体ベクトルvに対する透過のindex

            #反射するときのベクトルを更新　
            v[:,vl_index] *= (B[:,vl_index]*1-A[:,vl_index]*1).astype(np.float32)

            #透過するときのベクトルを更新
            if vt_index !=[]:
                v[:,vt_index] = ((ni[vt_index]/nt[vt_index])*(v[:,vt_index]*B[:,vt_index])\
                      +A[:,vt_index]*np.sign(v[:,vt_index])*np.cos(at[sub_index])).astype(np.float32)
            v[:,vt_index] /= np.linalg.norm(v[:,vt_index],axis=0).astype(np.float32)

            #境界を超えた時の位置更新,番地の更新
            pt_index = np.delete(np.arange(ni.size),vl_index)
            p[:,pt_index] *= (B[:,pt_index]*1-A[:,pt_index]*1).astype(np.float32)
            add[:,pt_index] += sigAv[:,pt_index].astype("int16")
        else:
            p = (p*(B*1-A*1)).astype(np.float32)
            add += sigAv.astype("int16")
        return v,p,add,s,mt
    """

    def _border_act(self,v,p,add,s,db,mt):
        l = self.model.voxel_space
        A = self.create01Array(np.argmin(db,axis=0),m=3)
        B = ~A.copy()

        sigAv = A*np.sign(v).astype("int16")
        ds = np.min(db,axis=0)
        s -= ds*mt
        p = l/2*A*np.sign(v)+B*(p+v*ds/mt)

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
                v[:,vt_index] = (ni[vt_index]/nt[vt_index])*v[:,vt_index]*B[:,vt_index]\
                      +A[:,vt_index]*np.sign(v[:,vt_index])*np.cos(at[sub_index])
            val = np.sqrt(v[0]**2+v[1]**2+v[2]**2)
            v = v/val
            #境界を超えた時の位置更新,番地の更新
            pt_index = np.delete(np.arange(ni.size),vl_index)
            p[:,pt_index] = p[:,pt_index]*(B[:,pt_index]*1-A[:,pt_index]*1)
            add[:,pt_index] += sigAv[:,pt_index]
        else:
            p = p*(B*1-A*1)
            add += sigAv

        return v,p,add,s,mt
    def _boundary_judgment(self,v,p,add):
        box_model = self.model.voxel_model
        l = np.float32(self.model.voxel_space)
        s = (-np.log(np.random.rand(self.w.size).astype(np.float32))).astype(np.float32)
        v_,p_,add_ = v,p,add
        index = np.arange(s.size)
        out_index = []
        while True:
            ###今の状態だと、ｖは最初Ｚ軸方向以外0なので、dbはしょうがないけどRuntimeWarningが出ます
            db = np.abs((l/2-p_*np.sign(v_))/np.abs(v_)).astype(np.float32)
            ma = self.model.getAbsorptionCoeff(add_)
            ms = self.model.getScatteringCoeff(add_)
            mt = (ma+ms).astype(np.float32)
            pb_index = np.where(s>=db.min(0)*mt)[0]
            pn_index = np.delete(np.arange(s.size),pb_index)
            if list(pb_index) != []:
                v[:,index[pn_index]] = v_[:,pn_index]
                p[:,index[pn_index]] = (p_[:,pn_index]\
                +s[pn_index]*v_[:,pn_index]/mt[pn_index]).astype(np.float32)

                add[:,index[pn_index]] = add_[:,pn_index]
                index = index[pb_index]

                v_,p_,add_,s,mt = self._border_act(
                    v_[:,pb_index],p_[:,pb_index],
                    add_[:,pb_index],s[pb_index],
                    db[:,pb_index],mt[pb_index])

                #外側に出る？
                out_index_ = np.where(
                    box_model[add_[0],add_[1],add_[2]] ==self.model.end_point
                )[0]
                if list(out_index_) != []:
                    v[:,index[out_index_]] = v_[:,out_index_]
                    p[:,index[out_index_]] = p_[:,out_index_]
                    add[:,index[out_index_]] = add_[:,out_index_]
                    out_index += list(index[out_index_])
                    v_ = np.delete(v_,out_index_, axis = 1)
                    p_ = np.delete(p_,out_index_, axis = 1)
                    add_ = np.delete(add_,out_index_, axis = 1)
                    s = np.delete(s,out_index_)
                    index = np.delete(index,out_index_)
            else:
                v[:,index] = v_
                p[:,index] = (p_+s*v_/mt).astype(np.float32)
                break
        return v,p,add,out_index

    def _w_update(self,w,ma,mt,p,add):
        dw = (w*ma/mt).astype(np.float32)
        return w-dw
    """
    def stepMovement(self):
        self.add = self.add.astype(np.int32)
        self.p = self.p.astype(np.float32)
        self.v = self.v.astype(np.float32)
        self.w = self.w.astype(np.float32)

        self.v,self.p,self.add,self.w,index = _movement(
            self.add, self.p,self.v, self.w,
            self.model.ma, self.model.ms, self.model.n, self.model.g,
            self.model.voxel_model, self.model.voxel_space,
            np.int32(self.w.size), np.int8(self.model.end_point)
        )
        sub_index = np.arange(index.size)
        index = np.where((index-sub_index) == 1)[0]
        self.p,self.v,self.w,self.add = self._border_out(self.p,self.v,self.w,self.add,index)
        return self.w.size
        """


    def stepMovement(self):
        self.v,self.p,self.add,index = self._boundary_judgment(self.v,self.p,self.add)

        self.p,self.v,self.w,self.add = self._border_out(self.p,self.v,self.w,self.add,index)

        ma = self.model.getAbsorptionCoeff(self.add).astype(np.float32)
        ms = self.model.getScatteringCoeff(self.add).astype(np.float32)
        self.w = self._w_update(self.w,ma,ma+ms,self.p,self.add)
        self.p,self.v,self.w,self.add = self._photon_vanishing(self.p,self.v,self.w,self.add)

        self.v = self.vectorUpdate(self.v,self.model.getAnisotropyCoeff(self.add))
        return self.w.size

    """
    def _end_process(self):#書き換え
        #index = np.where(~np.isnan(self.w))[0]
        self.v_result = self.v#[:,index]
        self.p_result = self.p#[:,index]
        self.add_result = self.add#[:,index]
        self.w_result = self.w#[index]"""

    def _end_process(self):
        print(self.contnum/self.nPh)
        """self.v_result = self.v_result[:,1:]
        self.p_result = self.p_result[:,1:]
        self.add_result = self.add_result[:,1:]
        self.w_result = self.w_result[1:]"""

    def _reset_results(self):
        self.v_result = np.empty((3,1)).astype(self.f_bit)
        self.p_result = np.empty((3,1)).astype(self.f_bit)
        self.add_result = np.empty((3,1)).astype('int32')
        self.w_result = np.empty(1).astype(self.f_bit)
        return self

    def get_voxel_model(self):
        return self.model.voxel_model

    def _generate_initial_coodinate(self,nPh,f = 'float32'):
        self._set_inital_add()
        self._set_beam_distribution()
        self._set_inital_vector()
        self._set_inital_w()


    def _set_inital_add(self):
        if self.beam_type == 'TEM00':
            self.add =  np.zeros((3, self.nPh),dtype = 'int32')
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
        else:
            print("ビームタイプが設定されていません")

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

    def set_monte_params(self,*,nPh,model, dtype='float32',w_beam = 0):
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
    def get_second_layer_addz(self):
        return np.where(self.voxel_model==self.voxel_model[1,1,1])[2][-1]+1

    def get_addIndex(self,add):
        return self.voxel_model[add[0],add[1],add[2]]

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
        self.model_shape = (3,3,3)

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

    def getModelSize(self):
        print("Memory area size for voxel storage: %0.3f Mbyte" % (self.voxel_model.nbytes*1e-6))

class TuringModel(VoxelModel):
    def __init__(self):
        self.model_name = 'TuringModel'
        self.dtype_f = np.float32
        self.dtype = np.int8
        self.ct_num=2
        self.subc_num=3
        self.skin_num=4
        self.end_point = 5
        self.params = {
            'xz_size':17.15,'voxel_space':0.0245,'dicom_path':False,'bv_tv':0.138,
            'symmetrization':False,'enclosure':False,
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

    def _make_voxel_model(self):
        #骨梁間隙を0,海綿骨を1、緻密骨を2、 皮下組織を3、皮膚を4、領域外を-1に設定する
        if self.params['dicom_path']:
            self.voxel_model = self._read_dicom()
        #　骨サイズに合わせてデータを削除
        #### 注意 ####
        #　xz_sizeはチューリングモデルよりも小さくなくてはならない
        num_pix = int((self.voxel_model.shape[0]-int(self.params['xz_size']/self.voxel_space))/2)
        self.voxel_model = self.voxel_model[num_pix:-num_pix,:,num_pix:-num_pix]

        def _cort_data_add(params,voxel_model,th_name,space,num,dtype):
            if params[th_name] !=0:
                num_pix = int(params[th_name]/space)
                # x軸
                voxel_model[:num_pix,:,:] = num
                voxel_model[-num_pix:,:,:] = num
                # y軸
                voxel_model[:,:num_pix,:] = num
                # z軸
                voxel_model[:,:,:num_pix] = num
                voxel_model[:,:,-num_pix:] = num
            return voxel_model

        def _soft_data_add(params,voxel_model,th_name,space,num,dtype):
            if params[th_name] !=0:
                num_pix = int(params[th_name]/space)
                ct = np.ones((
                voxel_model.shape[0],voxel_model.shape[1],num_pix),
                dtype = dtype)*num
                voxel_model = np.concatenate((ct,voxel_model),2)
                if params['symmetrization'] or params['enclosure']:
                    voxel_model = np.concatenate((voxel_model,ct),2)
                if params['enclosure']:
                    #y方向の追加
                    '''ct = np.ones((
                    voxel_model.shape[0],num_pix,voxel_model.shape[2]
                    ),dtype = dtype)*num
                    voxel_model = np.concatenate((voxel_model,ct),1)
                    voxel_model = np.concatenate((ct,voxel_model),1)'''
                    #x方向の追加
                    ct = np.ones((
                    num_pix,voxel_model.shape[1],voxel_model.shape[2]
                    ),dtype = dtype)*num
                    voxel_model = np.concatenate((voxel_model,ct),0)
                    voxel_model = np.concatenate((ct,voxel_model),0)
            return voxel_model

        self.voxel_model = _cort_data_add(
        self.params,self.voxel_model,'th_cortical',self.voxel_space,self.ct_num,self.dtype
        )
        self.voxel_model = _soft_data_add(
        self.params,self.voxel_model,'th_subcutaneus',self.voxel_space,self.subc_num,self.dtype
        )
        self.voxel_model = _soft_data_add(
        self.params,self.voxel_model,'th_dermis',self.voxel_space,self.skin_num,self.dtype
        )

        self.voxel_model[0,:,:] = self.end_point
        self.voxel_model[-1,:,:] = self.end_point
        self.voxel_model[:,0,:] = self.end_point
        self.voxel_model[:,-1,:] = self.end_point
        self.voxel_model[:,:,0] = self.end_point
        self.voxel_model[:,:,-1] = self.end_point

    def getModelSize(self):
        print("Memory area size for voxel storage: %0.3f Mbyte" % (self.voxel_model.nbytes*1e-6))

class TuringModel_cylinder(TuringModel):
    def __init__(self):
        self.model_name = 'TuringModel_cylinder'
        self.model_name = 'TuringModel_cylinder'
        self.dtype_f = np.float32
        self.dtype = np.int8
        self.ct_num=2
        self.subc_num=3
        self.skin_num=4
        self.air_num=5
        self.end_point = 6
        self.params = {
            'r_bone':9.17,'voxel_space':0.0245,'dicom_path':False,'bv_tv':0.138,
            'th_cortical':1.,'th_subcutaneus':2.6,'th_dermis':1.4,
            'n_space':1.,'n_trabecular':1.4,'n_cortical':1.4,'n_subcutaneus':1.4,'n_dermis':1.4,'n_air':1.,
            'ma_space':1e-8,'ma_trabecular':0.02374,'ma_cortical':0.02374,'ma_subcutaneus':0.011,'ma_dermis':0.037,'ma_air':1e-5,
            'ms_space':1e-8,'ms_trabecular':20.54,'ms_cortical':17.67,'ms_subcutaneus':20,'ms_dermis':20,'ms_air':1e-5,
            'g_space':0.90,'g_trabecular':0.90,'g_cortical':0.90,'g_subcutaneus':0.90,'g_dermis':.90,'g_air':.90,
            }
        self.keys = list(self.params.keys())

        self._make_model_params()
        self.bmd = self._get_bone_vbmd()
        self.voxel_space = self.params['voxel_space']
        self.voxel_model = np.zeros((3,3,3),dtype = self.dtype)
        self.model_shape = (3,3,3)

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

    def _make_voxel_model(self):
        #骨梁間隙を0,海綿骨を1、緻密骨を2、 皮下組織を3、皮膚を4、大気を５、領域外を-1に設定する
        if self.params['dicom_path']:
            self.voxel_model = self._read_dicom()
        #　骨サイズに合わせてデータを削除
        #### 注意 ####
        #　r_bone*2はチューリングモデルよりも小さくなくてはならない
        num_pix = int((self.voxel_model.shape[0]-int(self.params['r_bone']*2/self.voxel_space))/2)
        self.voxel_model = self.voxel_model[num_pix:-num_pix,:,num_pix:-num_pix]

        #軟組織分のVoxelを増やす。この時、全ては空気であると仮定する。
        num_pix = int((self.params['th_subcutaneus']+self.params['th_dermis'])/self.voxel_space)+1
        #Z軸方向
        ct = np.ones((
        self.voxel_model.shape[0],self.voxel_model.shape[1],num_pix),
        dtype = self.dtype)*self.air_num
        self.voxel_model = np.concatenate((ct,self.voxel_model),2)
        self.voxel_model = np.concatenate((self.voxel_model,ct),2)
        #x方向の追加
        ct = np.ones((
        num_pix,self.voxel_model.shape[1],self.voxel_model.shape[2]
        ),dtype = self.dtype)*self.air_num
        self.voxel_model = np.concatenate((self.voxel_model,ct),0)
        self.voxel_model = np.concatenate((ct,self.voxel_model),0)

        #中心からxzの半径方向距離を出す
        x_size = self.voxel_model.shape[0]
        z_size = self.voxel_model.shape[2]
        x_c = int((x_size)/2)
        z_c = int((z_size)/2)
        data_xnum = (np.tile(np.arange(x_size),(z_size,1)).T-x_c)*self.voxel_space
        data_znum = (np.tile(np.arange(z_size),(x_size,1))-z_c)*self.voxel_space
        r = np.sqrt(data_xnum**2+data_znum**2)

        #各組織の条件をつける
        #皮質骨
        r_ref_in = self.params['r_bone']-self.params['th_cortical']
        r_ref_out = self.params['r_bone']
        index = np.where((r>=r_ref_in)&(r<r_ref_out))
        self.voxel_model[index[0],:,index[1]]=self.ct_num
        num_pix = int((self.params['th_cortical'])/self.voxel_space)+1
        self.voxel_model[:,:num_pix,:]=self.ct_num
        #皮下組織
        r_ref_in = self.params['r_bone']
        r_ref_out = self.params['r_bone']+self.params['th_subcutaneus']
        index = np.where((r>=r_ref_in)&(r<r_ref_out))
        self.voxel_model[index[0],:,index[1]]=self.subc_num
        #真皮
        r_ref_in = self.params['r_bone']+self.params['th_subcutaneus']
        r_ref_out = self.params['r_bone']+self.params['th_subcutaneus']+self.params['th_dermis']
        index = np.where((r>=r_ref_in)&(r<r_ref_out))
        self.voxel_model[index[0],:,index[1]]=self.skin_num

        #境界を定義
        self.voxel_model[0,:,:] = self.end_point
        self.voxel_model[-1,:,:] = self.end_point
        self.voxel_model[:,0,:] = self.end_point
        self.voxel_model[:,-1,:] = self.end_point
        self.voxel_model[:,:,0] = self.end_point
        self.voxel_model[:,:,-1] = self.end_point
        self.voxel_model = self.voxel_model.astype(self.dtype)

# =============================================================================
# Public montecalro model
# =============================================================================

class VoxelPlateModel(BaseVoxelMonteCarlo):
    def __init__(
        self,*,nPh=1000,dtype='float32',
        beam_type = 'TEM00',w_beam = 0,
        beam_angle = 0,initial_refrect_by_angle = False,
        first_layer_clear = False,
        threadnum = 128,
    ):
        super().__init__(
            nPh = nPh, model = PlateModel(),dtype=dtype,
            w_beam=w_beam,beam_angle = beam_angle,beam_type = beam_type,
            initial_refrect_by_angle = initial_refrect_by_angle,
            first_layer_clear=first_layer_clear,
            threadnum = threadnum
        )

class VoxelTuringModel(BaseVoxelMonteCarlo):
    def __init__(
        self,*,nPh=1000,dtype='float32',
        beam_type = 'TEM00',w_beam = 0,
        beam_angle = 0,initial_refrect_by_angle = False,
        first_layer_clear = False,
        threadnum = 128,
        model_name = 'TuringModel'
        ):
        namelist = ['TuringModel','TuringModel_cylinder']
        if model_name==namelist[0]:
            model = TuringModel()
        elif model_name == namelist[1]:
            model = TuringModel_cylinder()
        else:
            print('Invalid name: ',model_name)

        super().__init__(
            nPh = nPh, model = model,dtype='float32',
            w_beam=w_beam,beam_angle = beam_angle,beam_type = beam_type,
            initial_refrect_by_angle = initial_refrect_by_angle,
            first_layer_clear=first_layer_clear,
            threadnum = threadnum
        )
        self.bone_model = False

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

    def get_model_fig(self,*,dpi=300,save_path = False,):
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
        if save_path:
            plt.savefig(
                save_path,
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
        }
        return calc_info
