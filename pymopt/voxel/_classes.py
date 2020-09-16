#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 17:12:05 2020

@author: kaname
"""


import numpy as np
from ..montecalro import MonteCalro
from abc import ABCMeta, abstractmethod
from ..utils.validation import _deprecate_positional_args
from ..fluence import IntarnalFluence
import gc
__all__ = ['VoxelPlateModel']

# =============================================================================
# Base solid model
# =============================================================================
        
class BaseVoxelMonteCalro(MonteCalro,metaclass = ABCMeta):
    #@_deprecate_positional_args
    @abstractmethod
    def __init__(self,*,nPh,model,fluence = False,f_bit='float32'):
        super().__init__()
        self.f_bit = f_bit
        self.nPh = nPh
        self.model = model
        self.fluence = fluence
        self.generateInisalCoodinate(self.nPh)
        
    def generateInisalCoodinate(self,nPh,f = 'float32'):
        center_add_xy = int(self.model.voxel_model.shape[0]/2)
        self.add =  np.full((3, nPh),center_add_xy).astype("int16")
        self.add[2] = 1
        self.p = np.zeros((3,nPh)).astype(f)
        self.p[2] = -self.model.voxel_space/2
        self.v = np.zeros((3,nPh)).astype(f)
        self.v[2] = 1
        self.w = np.ones(nPh).astype(f)
        self.w = self.initialWeight(self.w)
        
    def getResult(self):
        encoded_position = self.encooder(self.p_result,self.add_result)
        return{
            'p':encoded_position,
            'v':self.v_result,
            'w':self.w_result,
        }
        
    def encooder(self,p,add):
        space = self.model.voxel_space
        center_add_xy = int(self.model.voxel_model.shape[0]/2)
        encoded_position = p.copy()
        encoded_position[0] = space*(add[0]-center_add_xy)+p[0]
        encoded_position[1] = space*(add[1]-center_add_xy)+p[1]
        encoded_position[2] = np.round(space*(add[2]-1)+p[2]+space/2,6)
        return encoded_position
        
    def wUpdate(self,w,ma,mt,p,add):
        dw = w*ma/mt
        if self.fluence != False:
            encoded_position = self.encooder(p,add)
            self.fluence.saveFluesnce(encoded_position,dw)
        return w-dw
    
    def endProcess(self):
        self.v_result = self.v_result[:,1:]
        self.p_result = self.p_result[:,1:]
        self.add_result = self.add_result[:,1:]
        self.w_result = self.w_result[1:]
    
    def initialWeight(self,w):
        Rsp = 0
        n1 = self.model.n[0]
        n2 = self.model.n[1]
        if n1 != n2:
            Rsp = ((n1-n2)/(n1+n2))**2
        return w-Rsp
    
    @_deprecate_positional_args   
    def built(self,**kwargs):
        self.model.built(**kwargs)
        self.generateInisalCoodinate(self.nPh)
    
    def photonVanishing(self,p,v,w,add):
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
    
    def borderOut(self,p,v,w,add,index):
        self.v_result = np.concatenate([self.v_result, v[:,index]],axis = 1)
        self.p_result = np.concatenate([self.p_result, p[:,index]],axis = 1)
        self.add_result = np.concatenate([self.add_result, add[:,index]],axis = 1)
        self.w_result = np.concatenate([self.w_result,w[index]])

        p = np.delete(p,index, axis = 1)
        v = np.delete(v,index, axis = 1)
        add = np.delete(add,index, axis = 1)
        
        w = np.delete(w,index)
        return p,v,w,add
    
    def borderAct(self,v,p,add,s,db):
        l = self.model.voxel_space
        A = self.create01Array(np.argmin(db,axis=0),m=3)
        B = ~A
        ni = self.model.getReflectiveIndex(add)
        sigAv = A*np.sign(v).astype("int16")
        ds = np.min(db,axis=0)
        s -= ds
        
        p = l/2*A*np.sign(v)+B*(p+v*ds)
        
        nt = self.model.getReflectiveIndex((add+sigAv))
        pb_index = np.where(ni!=nt)[0]
        
        if list(pb_index) != []:
            #透過判別
            #入射角の計算
            ai = np.abs(A[:,pb_index]*v[:,pb_index])
            ai = np.arccos(ai[ai!=0])#Aから算出される0を排除して1次元配列にする
            #全反射するときのインデックスを取得
            sub_index = np.where(ai>=np.arcsin(nt[pb_index]/ni[pb_index]))[0]
            #屈折角の計算
            ##(ai,at)=(0,0)の時は透過させ方向ベクトルは変化させない
            at = np.arcsin((ni[pb_index]/nt[pb_index])*np.sin(ai))
            Ra = np.random.rand(ai.size)-0.5*(np.add((np.sin(ai-at)/np.sin(ai+at))**2,
                                                     (np.tan(ai-at)/np.tan(ai+at))**2))
            #全反射はRaを強制的に0以下にし反射させる
            Ra[sub_index] = -1
            
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
    
    def RTInterface(self,v,p,add):
        box_model = self.model.voxel_model
        l = self.model.voxel_space
        s = -np.log(np.random.rand(self.w.size))
        v_,p_,add_,s_ = v,p,add,s
        index = np.arange(s.size)
        out_index = []
        while True:
            db = (l/2-p_*np.sign(v_))/np.abs(v_)
            ma = self.model.getAbsorptionCoeff(add_)
            ms = self.model.getScatteringCoeff(add_)
            mt = ma+ms
            s_ = s_/mt
            pb_index = np.where(s_-db.min(0)>=0)[0]
            pn_index = np.delete(np.arange(s_.size),pb_index)
            if list(pb_index) != []:
                v[:,index[pn_index]] = v_[:,pn_index]
                p[:,index[pn_index]] = p_[:,pn_index]\
                +s_[pn_index]*v_[:,pn_index]
                
                add[:,index[pn_index]] = add_[:,pn_index]
                index = index[pb_index]
                v_,p_,add_,s_ = self.borderAct(v_[:,pb_index],
                                               p_[:,pb_index],
                                               add_[:,pb_index],
                                               s_[pb_index],
                                               db[:,pb_index])
                s_ = s_*mt[pb_index]
                out_index_ = np.where(box_model[add_[0],add_[1],add_[2]] < 0)[0]
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
        self.v,self.p,self.add,index = self.RTInterface(self.v,self.p,self.add)

        self.p,self.v,self.w,self.add = self.borderOut(self.p,self.v,self.w,self.add,index)

        G = self.model.getAnisotropyCoeff(self.add)
        self.v = self.vectorUpdate(self.v,G)
        ma = self.model.getAbsorptionCoeff(self.add)
        ms = self.model.getScatteringCoeff(self.add)
        self.w = self.wUpdate(self.w,ma,ma+ms,self.p,self.add)

        self.p,self.v,self.w,self.add = self.photonVanishing(self.p,self.v,self.w,self.add)

        return self.w.size
    
    
# =============================================================================
# Modeling class
# =============================================================================
        
class VoxelModel:
    def built(self):
        pass
    def getAbsorptionCoeff(self,add):
        x,y,z = add
        index = self.voxel_model[x,y,z]
        return self.ma[index]
    def getScatteringCoeff(self,add):
        x,y,z = add
        index = self.voxel_model[x,y,z]
        return self.ms[index]
    def getAnisotropyCoeff(self,add):
        x,y,z = add
        index = self.voxel_model[x,y,z]
        return self.g[index]
    def getReflectiveIndex(self,add):
        x,y,z = add
        index = self.voxel_model[x,y,z]+1
        return self.n[index]
    
class PlateModel(VoxelModel):
    @_deprecate_positional_args
    def __init__(
        self,*,thickness=[0.2,] ,xy_size=0.1 ,voxel_space = 0.1,
        ma=[1,],ms=[100,],g=[0.9,],n=[1.37,],n_air=1,f = 'float32'):
        self.n =np.array([n_air]+n).astype(f)
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
    def built(self,*,thickness,xy_size,voxel_space,ma,ms,g,n,n_air,f = 'float32'):
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
        
    def getModelSize(self):
        print("Memory area size for voxel storage: %d Mbyte" % (self.voxel_model.nbytes*1e-6))
        
# =============================================================================
# Public montecalro model
# =============================================================================
class VoxelPlateModel(BaseVoxelMonteCalro):
    def __init__(self,*,nPh,fluence=False,
                 nr=50,nz=20,dr=0.1,dz=0.1):
        super().__init__(nPh = nPh,fluence =fluence, model = PlateModel())
        if self.fluence:
            self.fluence = IntarnalFluence(nr=nr,nz=nz,dr=dr,dz=dz)
            
    def getFluence(self):
        return {'Arz':self.fluence.getArz(),
                'r':self.fluence.getArrayR(),
                'z':self.fluence.getArrayZ(),
                }