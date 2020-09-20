
"""
Created on Thu Sep 17 20:12:59 2020

@author: kaname
"""

import numpy as np
cimport numpy as np
np.import_array()

from libc.stdlib cimport rand, RAND_MAX
from libc.math cimport sin, cos, tan, acos, asin, copysign, sqrt,log,fabs
from libcpp cimport bool

cdef float NAN = np.nan
        
# =============================================================================
# BaseVoxelMonteCalro
# =============================================================================    
cdef class BaseVoxelMonteCalro:
    # 基本的に任意のモデルに継承して使います。
    # start() で計算が開始します。
    # 計算開始前に setInitialCoordinate(), setInitialPhotonWeight で初期値を渡してください。
    # その他のモデル固有のパラメーターは setParams() で定義しモデルに渡します。
    
    # fluence_mode が Ture の時は、 Fluence のオブジェクトを生成し、setFluence() で渡してください
    # 結果は、 getResult で取得します。 getResult は dict型で、p,v,wで定義されています。
    # 内部で消失した光子は、np.nanで表現しています。
    
    def __cinit__(self,int nPh, bool fluence_mode, float vectorTh, float wTh):
        self.vectorTh = vectorTh
        self.wTh = wTh
        self.nPh = nPh
        self.fluence_mode = fluence_mode
        self.russian_m = 10
        
        self.p_result = np.zeros((3,nPh),dtype = "float32")
        self.v_result = np.zeros((3,nPh),dtype = "float32")
        self.w_result = np.zeros((3,nPh),dtype = "float32")
        
    def setInitialCoordinate(self,float[::] p, float[::] v, int[::] add):
        self.p = p
        self.v = v
        self.add = add
        
    def setInitialPhotonWeight(self,float[:] w):
        self.w = w
        
    def setModel(self,np.ndarray[np.float32_t,ndim=3] voxel_model,
                 int[:] xy_size, float voxel_space):
        self.voxel_model = voxel_model
        self.xy_size = xy_size
        self.voxel_space = voxel_space
    
        
    def setFluence(self, Fluence _fluence):
        self.fluence = _fluence
        
    def setParams(self):
        pass
    
    def getResult(self):
        return {
            'p':self.p_result,
            'v':self.v_result,
            'w':self.w_result,
            }
    
    cpdef start(self):
        print("")
        print("###### Start ######")
        print("")
        
        self._monteCycle()
        
        print("")
        print("###### Finish ######")
            
    cdef void _monteCycle(self):
        cdef int n_Photon = self.nPh
        cdef int counter =  int(n_Photon/10)
        # Let's MonteCalro!
        for i in range(n_Photon):
            self._a_photon_movement(i)
            if i%counter==0:
                print("Progress: %s [％]"%int(i*100/n_Photon))
    
    cdef void _a_photon_movement(self,int p_id):
        cdef float px,py,pz
        cdef float vx,vy,vz
        cdef int adx,ady,adz
        cdef float w
        px = self.p[0][p_id]; py = self.p[1][p_id]; pz = self.p[2][p_id]
        vx = self.v[0][p_id]; vy = self.v[1][p_id]; vz = self.v[2][p_id]
        adx = self.add[0][p_id]; ady = self.add[1][p_id]; adz = self.add[2][p_id]
        w = self.w[p_id]
        
        cdef float ma, ms, mt, ni, nt
        cdef float ai, at, Ra
        cdef float dby,dbx,dbz,db_min,l
        l = self.model.voxel_space
        
        cdef int index,val_i,val_xi, val_yi, val_zi
        cdef float val_f,val_xf, val_yf, val_zf
        
        cdef int flag_1,flag_2
        
        flag_1 = 1
        while flag_1:
            s = self.random_uniform()
            s = -log(s)
            
            flag_2 = 1
            while flag_2:
                ma = self._getAbsorptionCoeff(adx,ady,adz)
                ms = self._getScatteringCoeff(adx,ady,adz)
                mt = ma + ms
                s = s/mt
                
                dbx = (l/2-copysign(px,vx))/fabs(vx)
                dby = (l/2-copysign(py,vy))/fabs(vy)
                dbz = (l/2-copysign(pz,vz))/fabs(vz)
                if dbz < dbx and dbz < dby:
                    db_min = dbz
                    index = 2
                elif dby < dbx and dby < dbz:
                    db_min = dby
                    index = 1
                elif dbx < dby and dbx < dbz:
                    db_min = dbx
                    index = 0
                val_f = db_min-s
                
                if val_f <= 0:
                    s -= db_min
                    ni = self._getReflectiveIndex(adx,ady,adz)
                    val_xi, val_yi, val_zi = self._get_next_add(
                        index,adx,ady,adz,vx,vy,vz)
                    nt = self._getReflectiveIndex(val_xi,val_yi,val_zi)
                    
                    if ni != nt:
                        ai = self._get_index_val(index,vx,vy,vz)
                        ai = fabs(ai)
                        ai = acos(ai)
                        val_f = asin(nt/ni)
                        if ai < val_f:
                            val_f = self.random_uniform()
                            at = asin(ni*sin(ai)/nt)
                            Ra = val_f-0.5*((sin(ai-at)/sin(ai+at))**2\
                                            + (tan(ai-at)/tan(ai+at))**2)
                                
                        else:
                            Ra = -1
                            
                        if Ra <= 0:
                            val_xi,val_yi,val_zi = self._create01val(index,-1,1)
                            vx *= val_xi
                            vy *= val_yi
                            vz *= val_zi
                            
                        else:
                            adx = val_xi; ady = val_yi; adz = val_zi
                            
                            val_xi,val_yi,val_zi = self._create01val(index,0,1)
                            vx = vx*val_xi
                            vy = vx*val_yi
                            vz = vx*val_zi
                            
                            val_xi,val_yi,val_zi = self._create01val(index,1,0)
                            val_f = cos(ai)
                            vx += val_xi*copysign(val_f,vx)
                            vy += val_yi*copysign(val_f,vy)
                            vz += val_zi*copysign(val_f,vz)
                            
                            val_f = ni/nt
                            vx *= val_f
                            vy *= val_f
                            vz *= val_f
                            
                            val_xi,val_yi,val_zi = self._create01val(index,-1,1)
                            px *= val_xi
                            py *= val_yi
                            pz *= val_zi
                            
                    else:
                        adx = val_xi; ady = val_yi; adz = val_zi
                        
                        val_xi,val_yi,val_zi = self._create01val(index,-1,1)
                        px *= val_xi
                        py *= val_yi
                        pz *= val_zi
                        
                    s *= mt
                    val_i = self._get_model_val_atAdd(adx,ady,adz)
                    if val_i < 0:
                        self._save_photon(p_id,px,py,pz,adx,ady,adz,vx,vy,vz,w)
                        flag_1 = 0
                        break
                    
                else:
                    px = px + vx*s
                    py = py + vy*s
                    pz = pz + vz*s
                    
                    g = self._getAnisotropyCoeff(adx,ady,adz)
                    vx,vy,vz = self.vectorUpdate( vx, vy, vz, g)
                    w = self._wUpdate(w,ma,mt,px,py,pz,adx,ady,adz)
                    if w <= self.wTh:
                        w = self._russianRoulette(w)
                        if w == 0.:
                            self._save_vanish_photon(p_id)
                            flag_1 = 0
                            break
                    flag_2 = 0
                    
    cdef int _get_model_val_atAdd(self, int x, int y, int z):
        cdef int val = self.voxel_model[x][y][z]
        return val
    
    cdef void _save_photon(self,int p_id, 
                           float px, float py, float pz,
                           int adx,int ady,int adz,
                           float vx, float vy, float vz, float w):
        px,py,pz = self._encooder(px,py,pz,adx,ady,adz)
        
        self.p_result[0][p_id] = px
        self.p_result[1][p_id] = py
        self.p_result[2][p_id] = pz
        
        self.v_result[0][p_id] = vx
        self.v_result[1][p_id] = vy
        self.v_result[2][p_id] = vz
        
        self.w_result[p_id] = w
    
    cdef _save_vanish_photon(self,int p_id):
        self.w_result[p_id] = NAN
        for i in range(3):
            self.p_result[i][p_id] = NAN
            self.v_result[i][p_id] = NAN
    
    cdef float _wUpdate(self,float w, float ma, float mt,
                        float px, float py, float pz, 
                        int adx, int ady, int adz):
        cdef float dw
        dw = w*ma/mt
        w -= dw
        if self.fluence_mode:
            px,py,pz = self._encooder(px,py,pz,adx,ady,adz)
            self.fluence.saveFluence(dw,px,py,pz)
        return w
    
    cdef float _russianRoulette(self,float w):
        ## 確率的に光子を生き返らせます。
        cdef int m = self.russian_m
        cdef float randnum = self.random_uniform()
        if randnum > 1/m:
            w = 0.
        else:
            w = w*m
        return w
    
    cdef float _encooder(self,
                         float px, float py, float pz, 
                         int adx, int ady, int adz):
        cdef float space = self.voxcel_space
        cdef int center_add_x,center_add_y
        center_add_x = int((self.xy_size[0]+2)/2)
        center_add_y = int((self.xy_size[1]+2)/2)
        
        cdef float  ex, ey, ez
        ex = space*(adx-center_add_x)+px
        ey = space*(ady-center_add_y)+py
        ez = space*(adz-1)+pz+space/2
        return ex, ey, ez
        
        
    cdef float _get_index_val(self,int index ,float x, float y, float z):
        cdef float val
        if index == 0:
            val = x
        elif index == 1:
            val = y
        elif index == 2:
            val = z
        return val
                    
    cdef int _get_next_add(self,int index, int x, int y, int z, 
                           float vx, float vy, float vz):
        if index == 0:
            vx = copysign(1,vx)
            x = x + int(vx)
        elif index == 1:
            vy = copysign(1,vy)
            y = y + int(vy)
        elif index == 2:
            vz = copysign(1,vz)
            z = z + int(vz)
        return x,y,z
    
    cdef int _create01val(self,int index, int index_val,int base_val):
        cdef float x,y,z
        x = base_val; y = base_val; z = base_val
        if index == 0:
            x = index_val
        elif index == 1:
            y = index_val
        elif index == 2:
            z = index_val
        return x,y,z
                
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
    
    cdef float _getAbsorptionCoeff(self, int x, int y, int z):
        pass
    
    cdef float _getScatteringCoeff(self, int x, int y, int z):
        pass
    
    cdef float _getAnisotropyCoeff(self, int x, int y, int z):
        pass
    
    cdef float _getReflectiveIndex(self, int x, int y, int z):
        pass
    
# =============================================================================
# Public montecalro class
# =============================================================================
cdef class VoxelPlateMonteCalr(BaseVoxelMonteCalro):
    
    def setParams(self,float[:] ma, float[:] ms, float[:] g, float[:] n):
        self.ma = ma
        self.ms = ms
        self.g = g
        self.n = n
    
    cdef float _getAbsorptionCoeff(self, int x, int y, int z):
        cdef int index = self.voxel_model[x][y][z]
        return self.ma[index]
    
    cdef float _getScatteringCoeff(self, int x, int y, int z):
        cdef int index = self.voxel_model[x][y][z]
        return self.ms[index]
    
    cdef float _getAnisotropyCoeff(self, int x, int y, int z):
        cdef int index = self.voxel_model[x][y][z]
        return self.g[index]
    
    cdef float _getReflectiveIndex(self, int x, int y, int z):
        cdef int index = self.voxel_model[x][y][z]+1
        return self.n[index]
    
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