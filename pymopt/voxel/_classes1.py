#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 17:12:05 2020

@author: kaname
"""


import numpy as np
import 
from abc import ABCMeta, abstractmethod
from ..utils.validation import _deprecate_positional_args
from ..fluence import IntarnalFluence
import gc
__all__ = ['VoxelPlateModel']

# =============================================================================
# Base solid model
# =============================================================================

class BaseVoxelMonteCarlo(metaclass = ABCMeta):
    #@_deprecate_positional_args
    @abstractmethod
    def __init__(self,*,nPh,model,fluence = False,f_bit='float32'):
        super().__init__()
        self.f_bit = f_bit
        self.nPh = nPh
        self.model = model
        self.fluence = fluence
        self.generateInisalCoodinate(self.nPh)
    
    def getVoxelModel(self):
        return self.model.voxel_model
        
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
        pass

    
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
    
    
# =============================================================================
# Modeling class
# =============================================================================
        
class VoxelModel:
    def build(self):
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
    def build(self,*,thickness,xy_size,voxel_space,ma,ms,g,n,n_air,f = 'float32'):
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