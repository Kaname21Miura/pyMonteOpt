#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 17:45:17 2020

@author: kaname
"""


import numpy as np
cimport numpy as np
np.import_array()

from numpy import float32 as DTYPE
from numpy import float64 as DOUBLE


cdef class VoxelModel:

    cdef float _getAbsorptionCoeff(self, int x, int y, int z):
        cdef int index = self.voxel_model[x][y][z]
        return self.ma[index]
    
    cdef float _getScatteringCoeff(self,add):
        cdef int index = self.voxel_model[x][y][z]
        return self.ms[index]
    
    cdef float _getAnisotropyCoeff(self,add):
        cdef int index = self.voxel_model[x][y][z]
        return self.g[index]
    
    cdef float _getReflectiveIndex(self,add):
        cdef int index = self.voxel_model[x][y][z]+1
        return self.n[index]
    
cdef class VoxelMonteCarlo:
    def __cinit__(self):
        cdef np.ndarray[np.float32_t,ndim=3] self.voxel_model
        
    cpdef setmodel(self,np.ndarray[np.float32_t,ndim=3] voxel_model):
        self.voxel_model = voxel_model
    
    
    
cdef class VoxelPlateMonteCarlo(VoxelMonteCarlo):
    def __cinit__(self,np.ndarray[SIZE_t, ndim=3] model,):
        