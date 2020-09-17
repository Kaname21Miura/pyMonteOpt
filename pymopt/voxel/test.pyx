
"""
Created on Thu Sep 17 20:12:59 2020

@author: kaname
"""


#import numpy as np
cimport numpy as np
np.import_array()

from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport sin, cos,copysign, sqrt
from libcpp cimport bool

cdef class BaceVoxelMonteCalro:
    def __cinit__(self,int nPh, float[:] p, float[:] v, int[:] add):
        self.vectorTh = 0.99999
        self.nPh = nPh
        self.p = p
        self.v = v
        self.add = add
        
    def build(self, object model):
        self.model = model
        
    cdef float _getAbsorptionCoeff(self, int x, int y, int z):
        cdef int index = self.model.voxel_model[x][y][z]
        return self.model.ma[index]
    
    cdef float _getScatteringCoeff(self, int x, int y, int z):
        cdef int index = self.model.voxel_model[x][y][z]
        return self.model.ms[index]
    
    cdef float _getAnisotropyCoeff(self, int x, int y, int z):
        cdef int index = self.model.voxel_model[x][y][z]
        return self.model.g[index]
    
    cdef float _getReflectiveIndex(self, int x, int y, int z):
        cdef int index = self.model.voxel_model[x][y][z]+1
        return self.model.n[index]
        
    cdef void _monteCycle(self,start_):
        print("### Start ####")
        cdef int n_Photon = self.nPh
        cdef int counter =  int(n_Photon/10)
        # Let's MonteCalro!
        for i in range(n_Photon):
            self._a_photon_movement()
            ###################### 途中
            
            
            
            if i%counter==0:
                print("Progress: %s [％]"%int(i*100/n_Photon))
        #return 0
    
    cdef void _a_photon_movement(self):
        cdef float px,py,pz
        cdef float vx,vy,vz
        cdef int adx,ady,adz
        cdef bool flag = True
        px = self.p[0]; py = self.p[1]; pz = self.p[2]
        vx = self.v[0]; vy = self.v[1]; vz = self.v[2]
        adx = self.add[0]; ady = self.add[1]; adz = self.add[2]
        
        while True:
            if flag:
                s = self.random_uniform()
                
            else:
                break
        #return 0
    
    cdef float random_uniform(self):
        cdef float r = rand()
        return r / RAND_MAX
    
    cdef float vectorUpdate(self,float vx,float vy,float vz,float g):
        cdef float randnum1, randnum2
        randnum1 = self.random_uniform()
        randnum2 = self.random_uniform()
        cdef float cosTh,sinTh, cosFi, sinFi
        cdef float th = self.vectorTh
        if g == 0:
            cosTh = 2*randnum1-1
        else:
            cosTh = (1+g**2-((1-g**2)/(1-g+2*g*randnum1**2)))/(2*g)
        
        sinTh = (1-cosTh**2)**0.5
    
        #cos(fai)とsin(fai)と求める
        cdef float Fi = 2*3.141592*randnum2
        cosFi = cos(Fi)
        sinFi = sin(Fi)
    
        #Zが１かそれ以外で分離
        if vz <= th:
            vx = sinTh*(vx*vz*cosFi-vy*sinFi)/sqrt(1-vz**2) + vx*cosTh
            vy = sinTh*(vy*vz*cosFi+vx*sinFi)/sqrt(1-vz**2) + vy*cosTh
            vz = -sinTh*cosFi*sqrt(1-vz**2) + vz*cosTh
            
        else:#Z方向ベクトルが0.99999以上
            vx = sinTh*cosFi
            vy = sinTh*sinFi
            vz = copysign(cosTh,vz)
        cdef float distance
        distance = sqrt(vx**2 + vy**2 + vz**2)
        vx = vx/distance
        vy = vy/distance
        vz = vz/distance
        return vx,vy,vz