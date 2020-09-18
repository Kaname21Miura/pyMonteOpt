#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 17:45:17 2020

@author: kaname
"""


import numpy as np
cimport numpy as np
np.import_array()

from libc.math cimport sqrt
from libcpp cimport bool

# =============================================================================
# Fluence
# =============================================================================
    
cdef class Fluence:
    def __cinit__(self,int nr,int nz,float dr,float dz):
        self.r = np.array([(i)*dr for i in range(nr+1)])
        self.z = np.array([(i)*dz for i in range(nz+1)])
        self.Arz = np.zeros((nr,nz),dtype = 'float32')
        self.nr = nr
        self.nz = nz
        self.dr = dr
        self.dz = dz
        
    cdef saveFluence(self,float w,
                     float x,float y,float zz):
        cdef float rr = sqrt(x**2 + y**2)
        cdef bool flag, val_bool
        
        cdef int num_r, num_z 
        cdef float[:] r = self.r
        cdef float[:] z = self.z
        cdef int nz, nr
        nz = self.nz; nr = self.nr
        
        flag = True; val_bool = False
        for i in range(nr):
            if i == nr-1:
                flag = False
                break
            val_bool = (rr >= r[i])and(rr < r[i+1])
            if val_bool:
                num_r = i
                break
        else:
            continue
        if flag:
            for i in range(nz):
                if i == nz-1:
                    flag = False
                    break
                val_bool = (zz >= z[i])and(zz < z[i+1])
                if val_bool:
                    num_z = i
                    break
            else:
                continue
            if flag:
                self.Arz[num_r][num_z] += w