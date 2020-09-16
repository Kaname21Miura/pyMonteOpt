#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 21:47:07 2020

@author: kaname
"""
#スライスは使わない方が良いらしい

import cython
import numpy as np
cimport numpy as np

from numpy import float32 as DTYPE

cdef class IntarnalFluence:
    def __cinit__(self,SIZE_t nr,SIZE_t nz,DTYPE_t dr,DTYPE_t dz):
        self.nr = nr
        self.nz = nz
        self.dr = dr
        self.dz = dz
        self.coodinateData()
        
    cdef coodinateData(self):
        self.r = np.array([(i)*self.dr for i in range(self.nr+1)])
        self.z = np.array([(i)*self.dz for i in range(self.nz+1)])
        self.Arz = np.zeros((self.nr,self.nz),dtype = DTYPE)
        
    cpdef saveFluesnce(self,np.ndarray[ndim=3] p,np.ndarray[ndim=1] w):
        cdef np.ndarray[ndim=1, dtype=DTYPE] rr,zz
        rr = np.sqrt(p[0]**2+p[1]**2)
        zz = p[2]
        cdef np.ndarray[ndim=1, dtype=SIZE_t] index_in,index_z,index_r
        index_in = np.where((zz<=self.z[-1])&(rr<=self.r[-1]))[0]
        index_z = self.getIndex(self.z,zz[index_in])
        index_r = self.getIndex(self.r,rr[index_in])
        self.sumationW(w[index_in],index_r,index_z)
        
    cdef sumationW(self,np.ndarray[ndim=1] w,np.ndarray indexr,np.ndarray indexz):
        cdef np.ndarray unique_id = (indexr*self.nz+indexz)
        cdef np.ndarray AA = np.array([indexr,indexz]).T
        cdef np.ndarray uni_unique_id = np.unique(unique_id)
        
        for i in uni_unique_id:
            index = np.where(unique_id==i)[0]
            sum_w = w[index].sum()
            inxez_Arz = AA[index][0]
            self.Arz[inxez_Arz[0],inxez_Arz[1]] \
                = self.Arz[inxez_Arz[0],inxez_Arz[1]] + sum_w
    
    cdef getIndex(self,standard,posi):
        ff = float
        c = np.sign(np.tile(posi,(standard.size,1)).astype(ff).T-standard)
        index = np.where(c==0)
        ind = np.where(index[1] == 0)
        c[index[0],index[1]] = -1
        c[index[0][ind[0]],0] = 1
        return np.where(c[:,:-1] != c[:,1:])[1]

    def getArrayZ(self):
        return np.array([(i+0.5)*self.dz for i in range(self.nz)])
    
    def getArrayR(self):
        return np.array([(i+0.5)*self.dr for i in range(self.nr)])