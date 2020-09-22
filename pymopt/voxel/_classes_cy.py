#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 17:12:05 2020

@author: kaname
"""


import numpy as np
import time
from abc import ABCMeta, abstractmethod
#from ..utils.validation import _deprecate_positional_args

from ._voxel_montecarlo import VoxelPlateMonteCarlo, Fluence
from ._classes import PlateModel

import gc

__all__ = ['VoxelPlateModelCy']

# =============================================================================
# Base model
# =============================================================================

class BaseVoxelMonteCarloCy(metaclass = ABCMeta):
    #@_deprecate_positional_args
    @abstractmethod
    def __init__(self,*,nPh,model,fluence = False,f_bit='float32'):
        super().__init__()
        self.f_bit = f_bit
        self.nPh = nPh
        self.model = model
        self.fluence = fluence
        self.generateInisalCoodinate(self.nPh)
        
         
    def build(self,**kwargs):
        pass
        
    def start(self):
        pass
    
    def getResult(self):
        pass
        
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
        

    
    def initialWeight(self,w):
        Rsp = 0
        n1 = self.model.n[0]
        n2 = self.model.n[1]
        if n1 != n2:
            Rsp = ((n1-n2)/(n1+n2))**2
        return w-Rsp
    
    def calTime(self, end, start):
        elapsed_time = end - start
        q, mod = divmod(elapsed_time, 60)
        if q < 60:
            print('Calculation time: %d minutes %0.3f seconds.' % (q, mod))
        else:
            q2, mod2 = divmod(q, 60)
            print('Calculation time: %d h %0.3f minutes.' % (q2, mod2))
    
    def getRdTtRate(self,v_result,w_result):
        Tt_index = np.where(v_result[2]>0)[0]
        Rd_index = np.where(v_result[2]<0)[0]
        self.Rdw = w_result[Rd_index].sum()/self.nPh
        self.Ttw = w_result[Tt_index].sum()/self.nPh
        print('######')
        print('Mean Rd %0.6f'% self.Rdw)
        print('Mean Tt %0.6f'% self.Ttw)
        print()
        
    def getRdTtValues(self):
        return {
            'Rd':self.Rdw,
            'Tt':self.Ttw,
        }
# =============================================================================
# Public montecalro model
# =============================================================================
class VoxelPlateModelCy(BaseVoxelMonteCarloCy):
    def __init__(self,*,nPh,fluence=False,
                 nr=50,nz=20,dr=0.1,dz=0.1):
        super().__init__(nPh = nPh,fluence =fluence, model = PlateModel())
        if self.fluence:
            self.fluence = Fluence(nr,nz,dr,dz)
        
            
    #@_deprecate_positional_args   
    def build(self,**kwargs):
        if self.fluence:
            self.monte = VoxelPlateMonteCarlo(self.nPh,1)
            self.setFluence(self.fluence)
        else:
            self.monte = VoxelPlateMonteCarlo(self.nPh,0)
        self.model = PlateModel()
        self.model.build(**kwargs)
        self.monte.setParams(
            self.model.ma,self.model.ms,
            self.model.g,self.model.n)
        self.monte.setModel(
            self.model.voxel_model,
            self.model.voxel_space)

        self.generateInisalCoodinate(self.nPh)
        self.monte.setInitialCoordinate(self.p,
                                   self.v,
                                   self.add.astype(int),
                                   self.w)
        del self.model
        gc.collect()
        
    def start(self):
        start_ = time.time()
        self.monte.start()
        print("")
        print("###### Finish ######")
        self.calTime(time.time(), start_)
        rez = self.monte.getResult()
        self.getRdTtRate(rez['v'],rez['w'])
        
        
    def getResult(self):
        return self.monte.getResult()
        
    def getFluence(self):
        return {'Arz':self.fluence.getArz(),
                'r':self.fluence.getArrayR(),
                'z':self.fluence.getArrayZ(),
                }
    def getParams(self):
        return self.model.getParams()
    
    
    