"""
Created on Thu Sep 10 20:43:52 2020

@author: kaname
"""

import cython
import numpy as np
cimport numpy as np


# =============================================================================
# Types and constants
# =============================================================================

from numpy import float32 as DTYPE
from numpy import float64 as DOUBLE

# =============================================================================
# SolidBuilder
# =============================================================================
cdef class SolidBuilder:
    cpdef build(self):
        pass
    
cdef class SolidPlateBuilder(SolidBuilder):
    def __cinit__ (self, DTYPE ms, DTYPE ma, DTYPE g, DTYPE n):
        self.ms = ms
        self.ma = ma
        self.g = g
        self.n = n

cdef class MonteCore:
    
    def vectorUpdate(self,v,G):
        index = np.where(G==0.0)[0]
        cosTh = np.empty_like(G)
        if list(index) != []:
            rand_num = np.random.rand(G.size).astype(self.f_bit)
            cosTh[index] = 2*rand_num[index]-1
            index = np.where(G!=0)[0]
            if list(index) != []:
                cosTh[index] = ((1+G[index]**2\
                                 -((1-G[index]**2)/(1-G[index]+2*G[index]*rand_num[index]))**2)/(2*G[index]))
        else:
            cosTh = (1+G**2-((1-G**2)/(1-G+2*G*np.random.rand(G.size).astype(self.f_bit)))**2)/(2*G)
        sinTh = np.sqrt(1-cosTh**2)
        
        #cos(fai)とsin(fai)と求める
        Fi = 2*np.pi*np.random.rand(G.size).astype(self.f_bit)
        cosFi = np.cos(Fi)
        sinFi = np.sin(Fi)
        
        #Zが１かそれ以外で分離
        th = self.vectorTh
        v1_index = np.where(np.abs(v[2])<=th)[0]
        v2_index = np.where(np.abs(v[2])>th)[0]
        
        #Z方向ベクトルが0.99999以下
        v1 = v[:,v1_index]
        cosTh1 = cosTh[v1_index]; sinTh1 = sinTh[v1_index]
        cosFi1 = cosFi[v1_index]; sinFi1 = sinFi[v1_index]
        B = np.sqrt(1-v1[2]**2)
        A = np.array([
            sinTh1*(v1[0]*v1[2]*cosFi1-v1[1]*sinFi1)/B,
            sinTh1*(v1[1]*v1[2]*cosFi1+v1[0]*sinFi1)/B,
            -sinTh1*cosFi1*B,
        ])
        v[:,v1_index] = A+v1*cosTh1
            
        #Z方向ベクトルが0.99999以上
        v2 = v[:,v2_index]
        cosTh2 = cosTh[v2_index]; sinTh2 = sinTh[v2_index]
        cosFi2 = cosFi[v2_index]; sinFi2 = sinFi[v2_index]
        v[:,v2_index] = np.array([
            sinTh2*cosFi2,
            sinTh2*sinFi2,
            np.sign(v2[2])*cosTh2,
        ],dtype=self.f_bit)
        v = v/np.linalg.norm(v,axis=0)
        return v

    def positionUpdate(self,p,v,L):
        return p+v*L

    #光子の１ステップにおけるエネルギーの損失を計算
    def wUpdate(self,w,ma,mt,rato,p):
        dw = w*rato*ma/mt
        if self.fluence != False:
            self.fluence.saveFluesnce(p,dw)
        return w-dw
            
    def russianRoulette(self,w):
        ## 確率的に光子を生き返らせます。
        m = 10
        ra = np.random.rand(w.size).astype(self.f_bit)
        index = np.where(ra>(1/m))[0].tolist()
        w[index] = 0
        index = np.where(ra<=(1/m))[0].tolist()
        w[index] = w[index]*m
        return w

    #光子の移動距離, uniformly distributed over the interval (0,1)
    def stepLength(self,size):
        return -np.log(np.random.rand(size)).astype(self.f_bit)
    
    #任意の位置(indexの行)が１でそれ以外は0の行列を作る
    def create01Array(self,index,m=3):
        n = index.size
        array_0_1 = np.zeros(m*n,dtype = bool)
        array_0_1[index+m*np.arange(n)] = 1
        return array_0_1.reshape(n,m).T

cdef class SolidPlate(MonteCore):
    