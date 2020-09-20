
import numpy as np
cimport numpy as np
np.import_array()

from libc.math cimport sqrt

ctypedef np.float32_t DTYPE_t

# =============================================================================
# Fluence
# =============================================================================
    
cdef class Fluence:
    cdef int nr,nz
    cdef float dr,dz
    cdef np.ndarray r,z
    cdef np.ndarray Arz
    
    def __cinit__(self,int nr,int nz,float dr,float dz):

        self.nr = nr
        self.nz = nz
        self.dr = dr
        self.dz = dz
    
        self.r = np.array([(i)*dr for i in range(nr+1)]).astype('float32')
        self.z = np.array([(i)*dz for i in range(nz+1)]).astype('float32')
        self.Arz = np.zeros((nr,nz),dtype = 'float32')
        
    cpdef getArz(self):
        return self.Arz
    
    cpdef test(self,float w,
                     float x,float y,float zz):
        self.saveFluence(w,x,y,zz)
    
    cdef saveFluence(self,float w,
                     float x,float y,float zz):
        cdef float rr = sqrt(x**2 + y**2)
        cdef int flag, val_bool
        
        cdef int num_r, num_z 
        cdef np.ndarray[ndim=1, dtype=DTYPE_t] r = self.r
        cdef np.ndarray[ndim=1, dtype=DTYPE_t] z = self.z
        cdef int nz, nr
        nz = self.nz; nr = self.nr
        
        flag = 1
        
        for i in range(nr):
            if i == nr-1:
                flag = 0
                break
            
            if (rr >= r[i])and(rr < r[i+1]):
                num_r = i
                break
        if flag:
            for i in range(nz):
                if i == nz-1:
                    flag = 0
                    break
                if (zz >= z[i])and(zz < z[i+1]):
                    num_z = i
                    break
                
            if flag:
                self.Arz[num_r][num_z] += w