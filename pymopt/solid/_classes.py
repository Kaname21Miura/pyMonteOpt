#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 11:28:02 2020

@author: kaname
"""

import numpy as np
import itertools
from abc import ABCMeta, abstractmethod

from ..montecalro import MonteCalro
from ..fluence import IntarnalFluence
from ..utils import _deprecate_positional_args


# =============================================================================
# Base solid model
# =============================================================================

class BaseSolidModel(MonteCalro,metaclass = ABCMeta):
    @abstractmethod
    @_deprecate_positional_args
    def __init__(self,*initial_data, **kwargs):
        super().__init__()
        self.keys = ['nPh','g','ma','ms','n','n_air','thickness',
                     'fluence','nr','nz','dr','dz',
                     'f_bit','vectorTh',
                     ]
        self.fluence = False
        self.nr=50;self.nz=20;self.dr=0.1;self.dz=0.1
        
        self.borderposit = np.array([0])
        self.thickness = np.array([0])
        self.n = np.array([1])
        self.n_air = 1
        self.ref_index = np.array([1,1])
        self.ms = np.array([1])
        self.ma = np.array([1])
        self.g = np.array([1])
        
        self.built(*initial_data, **kwargs)
        self.generateInisalCoodinate()
        if self.fluence :
            self.fluence = IntarnalFluence(nr=self.nr,nz=self.nz,dr=self.dr,dz=self.dz)
        
    def checkNumpyArray(self,val,key):
        if (not type(val) is np.ndarray)and(not type(val) is list):
            e_mess = key + ' must be list or ndarray'
            raise ValueError(e_mess)
            
    def checkPrams(self):
        check_values = np.array([self.g,self.ma,self.ms,self.n,self.thickness])
        check_keys = ['g','ma','ms','n','thickness']
        for val,key in zip(check_values,check_keys):
            self.checkNumpyArray(val,key)
            setattr(self,key,np.array(val).astype(self.f_bit))
        
    def built(self,*initial_data, **kwargs):
                
        for dictionary in initial_data:
            for key in dictionary:
                if not key in self.keys:
                    raise KeyError(key)
                setattr(self,key, dictionary[key])
        for key in kwargs:
            if not key in self.keys:
                raise KeyError(key)
            setattr(self, key, kwargs[key])
            
        self.setBorderPosit()
        self.checkPrams()
        self.setRefIndex()
        self.generateInisalCoodinate()


    def setRefIndex(self):
        border = np.append(self.n_air,self.n)
        border = np.append(border,self.n_air).astype(self.f_bit)
        setattr(self,'ref_index',border)
        
    def setBorderPosit(self):
        thick = [0]+self.thickness
        b = 0; b_list = []
        for i in  thick:
            b += i
            b_list.append(b)
        setattr(self,'borderposit',np.array(b_list).astype(self.f_bit))
        
    @abstractmethod
    def stepMovement(self):
        return self.w.size
    
    def generateInisalCoodinate(self):
        self.p = np.zeros((3,self.nPh),dtype = self.f_bit)
        self.v = np.zeros((3,self.nPh),dtype = self.f_bit)
        self.v[2] = 1
        self.w = self._initialWeight(np.full(self.nPh,1).astype(self.f_bit))
        
    def _initialWeight(self,w):
        Rsp = 0
        n1 = self.ref_index[0]
        n2 = self.ref_index[1]
        if n1 != n2:
            Rsp = ((n1-n2)/(n1+n2))**2
        return w-Rsp
    def saveResult(self,result):
        if result[0][0].tolist() != []:
            self.p_result = np.concatenate([self.p_result, result[0]],axis = 1)
            self.v_result = np.concatenate([self.v_result, result[1]],axis = 1)
            self.w_result = np.concatenate([self.w_result, result[2]])
            
    def endProcess(self):
        self.v_result = self.v_result[:,1:]
        self.p_result = self.p_result[:,1:]
        self.w_result = self.w_result[1:]
        
    def getResult(self):
        return {
            'p':self.p_result,
            'v':self.v_result,
            'w':self.w_result,
        }
    
    def getFluence(self):
        return {'Arz':self.fluence.getArz(),
                'r':self.fluence.getArrayR(),
                'z':self.fluence.getArrayZ(),
                }

# =============================================================================
# Public montecalro model
# =============================================================================
        
class SolidPlateModel(BaseSolidModel):
    @_deprecate_positional_args
    def __init__(self,*initial_data, **kwargs):
        super().__init__(*initial_data, **kwargs)
        
    # 境界への入射角を定義
    def getIncidentAngle(self,v):
        return np.arccos(abs(v[2]))
    
    # 境界上のベクトル更新
    def boundaryVectorUpdate(self,v,at,snell_rato,vl_index,vt_index):
        #反射するとき
        if list(vl_index) !=[]:
            v[2,vl_index] = -v[2,vl_index]
        #透過するときのベクトルを更新
        if list(vt_index)!=[]:
            v[0,vt_index] = snell_rato[vt_index]*v[0,vt_index]
            v[1,vt_index] = snell_rato[vt_index]*v[1,vt_index]
            v[2,vt_index] = np.sign(v[2,vt_index])*np.cos(at)
        return v
    
    # 境界判別と光子移動距離の更新
    def boundaryJudgment(self,p,v,s):
        ma,ms,g = self.getOpticalProperties(p,v)
        mt = ma+ms
        s = s/mt
        db = self.getDistanceBoundary(p,v)
        pb_index = np.where((s-db)>=0)[0]
        dl = s[pb_index]-db[pb_index]
        s[pb_index] = db[pb_index]
        return pb_index,s,dl,ma,ms,g

    
    def getBorderIndex(self,p,v):
        border = self.borderposit
        margin = 1e-8
        delta  = np.tile(border,(p[2].size,1))-np.tile(p[2],(border.size,1)).T
        index_zero = np.where(abs(delta)<margin)
        delta[index_zero[0],index_zero[1]] = 0
        delta = np.sign(delta).astype('int16')
        ind = np.array(np.where(delta==0))
        ind_shallow = np.array(np.where(ind[1]==0)[0])
        ind_deep = np.array(np.where(ind[1]==(border.size-1))[0])
        if list(ind[0])!= []:
            delta[ind[0],ind[1]] = -np.sign(v[2,ind[0]])
            delta[ind[0,ind_shallow],0] = -1
            delta[ind[0,ind_deep],border.size-1] = 1
        return np.where(delta[:,:-1] != delta[:,1:])[1]
    
    def getOpticalProperties(self,p,v):
        index = self.getBorderIndex(p,v)
        p_size = p[2].size
        A = self.create01Array(index,m=self.ma.size).T
        ma_out = (np.tile(self.ma,(p_size,1))*A).max(1)
        ms_out = (np.tile(self.ms,(p_size,1))*A).max(1)
        g_out = (np.tile(self.g,(p_size,1))*A).max(1)
        return ma_out,ms_out,g_out
    
    def getNextMt(self,p,v):
        index = self.getBorderIndex(p,v)
        index = np.sign(v[2])+index
        index_index = np.where((index >= 0) & (index < self.ma.size))[0]
        data = np.zeros((2,p[2].size))
        A = self.create01Array(index[index_index].astype('int16'),m=self.ma.size).T
        data[0,index_index] = (np.tile(self.ma,(index_index.size,1))*A).max(1)
        data[1,index_index] = (np.tile(self.ms,(index_index.size,1))*A).max(1)
        return data[0],data[1]
    
    def getDistanceBoundary(self,p,v):
        index_positive = np.where(v[2]>0)[0]
        index_negative = np.where(v[2]<0)[0]
        index_zero = np.where(v[2]==0)[0]
        index_border = self.getBorderIndex(p,v)
        border = self.borderposit
        tilearray = np.tile(border,(p[2].size,1))
        shallow = (tilearray*self.create01Array(index_border,m=border.size).T).max(1)
        deep = (tilearray*self.create01Array(index_border+1,m=border.size).T).max(1)
        S = np.zeros_like(p[2])
        S[index_negative] = (shallow[index_negative] - p[2,index_negative])/v[2,index_negative]
        S[index_positive] = (deep[index_positive] - p[2,index_positive])/v[2,index_positive]
        S[index_zero]=1000
        return abs(S)
    
    #屈折率ni,ntの取得
    def getNiNt(self,p,v):
        ind_negative = np.where(v[2]<0)[0]
        ind_positive = np.where(v[2]>0)[0]
        pz = p[2]
        n = self.ref_index
        border = self.borderposit
        index = self.getBorderIndex(p,v)+1
        
        n_array = np.tile(n,(index.size,1)).T
        ni = (n_array*self.create01Array(index,m=border.size+1)).max(0)
        
        nt = np.empty_like(pz)
        shallow = (n_array*self.create01Array(index-1,m=border.size+1)).max(0)
        deep = (n_array*self.create01Array(index+1,m=border.size+1)).max(0)
        nt[ind_positive] = deep[ind_positive]
        nt[ind_negative] = shallow[ind_negative]
        return ni,nt
    
    def limBorder(self,p): #境界面上のZ方向計算誤差をなくします。
        border = self.borderposit
        index = np.argmin(abs(np.tile(border,(p[2].size,1)).T-p[2]),axis=0)
        A = self.create01Array(index,m=border.size).T
        p[2] = (np.tile(border,(p[2].size,1))*A).max(1)
        return p
    
    def borderOut(self,p,v,w):
        margin = 1e-10
        border = self.borderposit
        del_index = np.where(((p[2]<=border[0]+margin)&(v[2]<0))|((p[2]>=border[-1]-margin)&(v[2]>0)))[0]
        result = list([p[:,del_index],v[:,del_index],w[del_index]])
        index = np.where((p[2]<=border[0]+margin)&(v[2]>0))[0]
        p[2,index] = border[0]
        index = np.where((p[2]>=border[-1]-margin)&(v[2]<0))[0]
        p[2,index] = border[-1]
        v = np.delete(v, del_index, axis = 1)
        p = np.delete(p, del_index, axis = 1)
        w = np.delete(w, del_index)
        return p,v,w,result
    
    def photonVanishing(self,w,p,v):
        #photn waight が0.0001以下
        del_index = np.where(w<=0.0001)[0]
        if list(del_index) != []:
            w[del_index] = self.russianRoulette(w[del_index])
            del_index = np.where(w==0)[0]
            v = np.delete(v, del_index, axis = 1)
            p = np.delete(p, del_index, axis = 1)
            w = np.delete(w, del_index)
        return p,v,w
    
    def updateOnBoundary(self,v,p,f="float32"):
        ## ここでは、透過判別や反射、屈折の計算を行います。
        #境界を超える前後の屈折率を取得  
        ni,nt = self.getNiNt(p,v)
        #境界前後で屈折率が変化する場合のindex
        pb_index = np.where(ni!=nt)[0]
        
        if list(pb_index) != []:
            #透過判別
            #入射角の計算
            ai = self.getIncidentAngle(v[:,pb_index])
            #全反射するときのインデックスを取得
            sub_index = np.where((ai>np.arcsin(nt[pb_index]/ni[pb_index])))[0]
            #屈折角の計算
            ##(ai,at)=(0,0)の時は透過させ方向ベクトルは変化させない
            at = np.arcsin(ni[pb_index]*np.sin(ai)/nt[pb_index])
            Ra = np.random.rand(ai.size)\
            -((np.sin(ai-at)/np.sin(ai+at))**2+(np.tan(ai-at)/np.tan(ai+at))**2)/2
            #全反射はRaを強制的に0以下にし反射させる
            Ra[sub_index] = -1
            #透過と反射のindexを取得
            vl_index = pb_index[np.where(Ra<=0)[0]]#反射
            sub_index = np.where(Ra>0)[0]
            vt_index = pb_index[sub_index]#透過
            at = at[sub_index]
            #ベクトルを更新
            v = self.boundaryVectorUpdate(v,at,ni/nt,vl_index,vt_index)
        return v

    def RTInterface(self,p,v,w,dl,mt):
        ## 境界を超えた時の対処
        pp = p.copy();vv = v.copy()
        index = np.arange(p[2].size)
        index_remain = []
        ma_list = [];ms_list = [];g_list = []
        while 1:
            pp = self.limBorder(pp)
            vv = self.updateOnBoundary(vv,pp)
            n_ma,n_ms = self.getNextMt(pp,vv)
            ind = np.where(n_ma == 0)[0]
            p[:,index[ind]] = pp[:,ind]
            v[:,index[ind]] = vv[:,ind]
            
            pp = np.delete(pp,ind,axis=1)
            vv = np.delete(vv,ind,axis=1)
            index = np.delete(index,ind)
            
            ind = np.where(n_ma != 0)[0]
            mt = mt[ind]
            pb_index,l,dl,ma,ms,g = self.boundaryJudgment(pp,vv,dl[ind]*mt)
            pn_index = np.delete(np.arange(index.size),pb_index)
            pp = self.positionUpdate(pp,vv,l)
            p[:,index[pn_index]] = pp[:,pn_index]
            v[:,index[pn_index]] = vv[:,pn_index]

            index_remain.append(index[pn_index])
            ma_list.append(ma[pn_index])
            ms_list.append(ms[pn_index])
            g_list.append(g[pn_index])
            mt = ma[pb_index]+ma[pb_index]
            
            pp = np.delete(pp,pn_index,axis=1)
            vv = np.delete(vv,pn_index,axis=1)
            index = np.delete(index,pn_index)
            if list(index) == []:
                break
        index = np.array(list(itertools.chain.from_iterable(index_remain)))
        if list(index)!=[]:
            ma = np.array(list(itertools.chain.from_iterable(ma_list)))
            ms = np.array(list(itertools.chain.from_iterable(ms_list)))
            g = np.array(list(itertools.chain.from_iterable(g_list)))        
            mt = ma+ms        
            v[:,index] = self.vectorUpdate(v[:,index],g)
            w[index] = self.wUpdate(w[index],ma,mt,1,p[:,index])
        return p, v, w
    #光の動きを制御する
    def stepMovement(self):
        ## 光子の位置並びにベクトルを更新します。
        #１stepで光子が進む予定の距離を定義
        s = self.stepLength(self.p[0].size)
        #境界を超える光子のindexを取得
        pb_index,l,dl,ma,ms,g = self.boundaryJudgment(self.p,self.v,s)
        mt = ma + ms
        #光子位置を更新
        self.p = self.positionUpdate(self.p,self.v,l)
        #print(mt)
        if list(pb_index) != []:
            #境界上光子のベクトルの更新
            self.p[:,pb_index],self.v[:,pb_index],self.w[pb_index] = self.RTInterface(
                self.p[:,pb_index],self.v[:,pb_index],self.w[pb_index],dl,mt[pb_index])
            
            #境界を超えない光子の位置,ベクトルを更新
            pn_index = np.delete(np.arange(self.p[2].size),pb_index)
            self.v[:,pn_index] = self.vectorUpdate(self.v[:,pn_index],g[pn_index])
            self.w[pn_index] = self.wUpdate(self.w[pn_index],ma[pn_index],mt[pn_index],1,self.p[:,pn_index])
            
        else:#全ての光子が境界に乗っていない時
            self.v= self.vectorUpdate(self.v,g)
            self.w = self.wUpdate(self.w,ma,mt,1,self.p)
            
        self.p,self.v,self.w,result=self.borderOut(self.p,self.v,self.w)
        self.p,self.v,self.w = self.photonVanishing(self.w,self.p,self.v)
        self.saveResult(result)
        return self.w.size