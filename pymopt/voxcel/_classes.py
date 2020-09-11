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
__all__ = ['VoxcelPlateModel']

class BaseVoxcelModel(MonteCalro,metaclass = ABCMeta):
    @abstractmethod
    @_deprecate_positional_args
    def __init__(
            self,*,
            nPh,
            voxcel_space,
            size,
            n_air=1.,
            f_bit = 'float32',
            ):
        super().__init__()
        self.nPh = nPh
        self.voxcel_space = voxcel_space
        self.size = size
        self.voxcel_model = np.zeros(self.size).astype('int16')
        self.n_air = n_air
        self.f_bit = f_bit
        
        self.add = np.zeros((3,1)).astype(int)
        
    def getAbsorptionCoeff(self,add):
        index = self.voxcel_model[add[0],add[1],add[2]]
        return self.ma[index]
    
    def getScatteringCoeff(self,add):
        index = self.voxcel_model[add[0],add[1],add[2]]
        return self.ms[index]
    def getAnisotropyCoeff(self,add):
        index = self.voxcel_model[add[0],add[1],add[2]]
        return self.g[index]
    def getReflectiveIndex(self,add):
        index = self.voxcel_model[add[0],add[1],add[2]]+1
        return self.ref_index[index]
    
        
class VoxcelPlateModel(BaseVoxcelModel):
    @_deprecate_positional_args
    def __init__(self,*,nPh,):
        super().__init__()
        