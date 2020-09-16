#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 10 10:30:04 2020

@author: kaname
"""
import numpy as np
from numba import njit
from ..utils import _deprecate_positional_args


class IntarnalFluence(object):
    @_deprecate_positional_args
    def __init__(self,*,nr,nz,dr,dz):
        self.r = np.array([(i)*dr for i in range(nr+1)])
        self.z = np.array([(i)*dz for i in range(nz+1)])
        self.Arz = np.zeros((nr,nz),dtype = 'float32')
        self.nr = nr
        self.nz = nz
        self.dr = dr
        self.dz = dz
        print("Memory area size for fluence storage: %d Mbyte" % (self.Arz.nbytes*1e-6))
        
    def getArz(self):
        return self.Arz
    
    def getArrayZ(self):
        return np.array([(i+0.5)*self.dz for i in range(self.nz)])
    
    def getArrayR(self):
        return np.array([(i+0.5)*self.dr for i in range(self.nr)])

    def saveFluesnce(self,p,w):
        rr = np.sqrt(p[0]**2+p[1]**2)
        zz = p[2]
        prz = np.array([rr,zz]).astype('float32')
        prz = prz[:,np.argsort(prz[1,:])]
        self.Arz = self._inputArz(prz,w,self.r,self.z,self.Arz)
        
    @staticmethod
    @njit('f4[:,:](f4[:,:],f4[:],f8[:],f8[:],f4[:,:])')
    def _inputArz(prz,w,r,z,Arz):
        count = 0
        nr,nz = r.size,z.size
        for wi in w:
            flag = True
            num_r = 0
            num_z = 0
            val = 0
            for i in r:
                if val==nr-1:
                    flag = False
                    break
                if prz[0][count]>=i and prz[0][count]<r[val+1]:
                    num_r = val
                    break
                val+=1
            else:
                continue
            if flag:
                val = 0
                for i in z:
                    if val==nz-1:
                        flag = False
                        break
                    if prz[1][count]>=i and prz[1][count]<z[val+1]:
                        num_z = val
                        break
                    val+=1
                else:
                    continue
                if flag:
                    Arz[num_r][num_z]+=wi
                    count+=1
        return Arz