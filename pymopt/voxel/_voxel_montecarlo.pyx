
"""
Created on Thu Sep 17 20:12:59 2020

@author: kaname
"""

import numpy as np
cimport numpy as np
np.import_array()
from cython.parallel import prange
from libc.stdlib cimport rand, RAND_MAX

from libc.math cimport sin, cos, tan, acos, asin, copysign, sqrt,log,fabs

#from libcpp cimport bool
cdef float NAN = np.nan
ctypedef np.float32_t DTYPE_t

# =============================================================================
# BaseVoxelMonteCarlo
# =============================================================================
cdef class BaseVoxelMonteCarlo:
    # 基本的に任意のモデルに継承して使います。
    # start() で計算が開始します。
    # 計算開始前に setInitialCoordinate(), setInitialPhotonWeight で初期値を渡してください。
    # その他のモデル固有のパラメーターは setParams() で定義しモデルに渡します。

    # fluence_mode が Ture の時は、 Fluence のオブジェクトを生成し、setFluence() で渡してください
    # 結果は、 getResult で取得します。 getResult は dict型で、p,v,wで定義されています。
    # 内部で消失した光子は、np.nanで表現しています。
    cdef:
        float vectorTh,wTh,voxel_space,russian_m
        int nPh,fluence_mode
        np.ndarray p_result,v_result,w_result
        np.ndarray p,v,w,add,voxel_model
        np.ndarray xy_size
        int logger1,logger2
        np.ndarray logger_int,logger_float

    def __cinit__(self,int nPh, int fluence_mode):
        self.vectorTh = 0.9999
        self.wTh = 0.0001
        self.nPh = nPh
        self.fluence_mode = fluence_mode
        self.russian_m = 10.
        self.logger1 = 0
        self.logger2 = 0
        self.logger_int = np.zeros(nPh,dtype = int)
        self.logger_float = np.zeros(nPh,dtype = 'float32')

        self.p_result = np.zeros((3,nPh),dtype = "float32")
        self.v_result = np.zeros((3,nPh),dtype = "float32")
        self.w_result = np.zeros(nPh,dtype = "float32")

        self.voxel_model = np.zeros((10,10,10),dtype = "int8")


    def set_nofphoton(self,int nPh):
        self.nPh = nPh
        self.p_result = np.zeros((3,nPh),dtype = "float32")
        self.v_result = np.zeros((3,nPh),dtype = "float32")
        self.w_result = np.zeros(nPh,dtype = "float32")

    def get_logger(self):
        return {
            'logger1':self.logger1,
            'logger2':self.logger2,
            'logger_int':self.logger_int,
            'logger_float':self.logger_float,
            }

    def setInitialCoordinate(self,np.ndarray[ndim = 2, dtype=DTYPE_t] p,
                             np.ndarray[ndim = 2, dtype=DTYPE_t] v,
                             np.ndarray[ndim = 2, dtype=np.int_t] add,
                             np.ndarray[ndim = 1, dtype=DTYPE_t] w):
        self.p = p
        self.v = v
        self.add = add
        self.w = w

    def setModel(self,np.ndarray[ndim = 3, dtype=np.int8_t] voxel_model,
                 float voxel_space):
        self.voxel_model = voxel_model
        self.xy_size = np.array([voxel_model.shape[0],
                                 voxel_model.shape[1]]).astype(int)
        self.voxel_space = voxel_space

    cdef Fluence fluence

    def setFluence(self,Fluence fluence):
        self.fluence = fluence

    def getFluenceResult(self):
        return self.fluence.getArz

    def setParams(self):
        pass

    def getResult(self):
        return {
            'p':self.p_result,
            'v':self.v_result,
            'w':self.w_result,
            }


    cpdef start(self):
        self._monteCycle()

    cdef void _monteCycle(self):
        cdef int n_Photon = self.nPh
        cdef int counter =  int(n_Photon/10)
        # Let's MonteCarlo!
        for i in range(n_Photon):
            self._a_photon_movement(i)
            #if i%counter==0:
                #print("Progress: %s [％]"%int(i*100/float(n_Photon)))

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
        cdef float m = self.russian_m
        cdef float randnum = self.random_uniform()
        if randnum > 1./m:
            w = 0.
        else:
            w = w*m
        return w

    cdef _encooder(self,float px, float py, float pz,
                         int adx, int ady, int adz):
        cdef float space = self.voxel_space
        cdef int center_add_x,center_add_y
        center_add_x = int((self.xy_size[0]+2)/2)
        center_add_y = int((self.xy_size[1]+2)/2)

        cdef float  ex, ey, ez
        ex = space*(adx-center_add_x)+px
        ey = space*(ady-center_add_y)+py
        ez = space*(adz-1)+pz+space/2.
        return ex, ey, ez


    cdef float _get_index_val(
        self,int index ,float x, float y, float z
        ):
        cdef float val = 0.
        if index == 0:
            val = x
        elif index == 1:
            val = y
        elif index == 2:
            val = z
        return val

    cdef (int,int,int) _get_next_add(
        self,int index,int x, int y, int z,
        float vx, float vy, float vz
        ):
        if index == 0:
            x += int(copysign(1,vx))
        elif index == 1:
            y += int(copysign(1,vy))
        elif index == 2:
            z += int(copysign(1,vz))
        return x,y,z

    cdef (float,float,float) _create01val(
        self,int index, float index_val,float base_val
        ):
        cdef float x_,y_,z_
        x_ = base_val; y_ = base_val; z_ = base_val
        if index == 0:
            x_ = index_val
        elif index == 1:
            y_ = index_val
        elif index == 2:
            z_ = index_val
        return x_, y_, z_


    cdef float random_uniform(self):
        cdef float random = float(rand())
        cdef float randmax = float(RAND_MAX)
        return random/randmax

    """cdef np.float32_t random_uniform(self):
        cdef np.float32_t r = np.random.rand()
        return r"""

    cdef (float,float,float) vectorUpdate(
        self,float vx,float vy,float vz,float g
        ):
        cdef:
            float randnum1, randnum2
            float cosTh,sinTh, cosFi, sinFi,val_f,Fi
            float th = 0.99999
            float distance

        randnum1 = self.random_uniform()
        randnum2 = self.random_uniform()
        if g == 0.:
            cosTh = 2*randnum1-1
        else:
            cosTh = (1+g**2-((1-g**2)/(1-g+2*g*randnum1))**2)/(2*g)

        sinTh = sqrt(1-cosTh**2)

        Fi = 2*3.141592*randnum2
        cosFi = cos(Fi)
        sinFi = sin(Fi)

        if fabs(vz) <= th:
            val_f = sqrt(1.-vz**2)

            vx = sinTh*(vx*vz*cosFi-vy*sinFi)/val_f + vx*cosTh
            vy = sinTh*(vy*vz*cosFi+vx*sinFi)/val_f + vy*cosTh
            vz = -sinTh*cosFi*val_f + vz*cosTh

        else:#Z方向ベクトルが0.99999以上
            vx = sinTh*cosFi
            vy = sinTh*sinFi
            vz = cosTh*copysign(1,vz)

        #distance = (vx**2 + vy**2 + vz**2)**0.5
        #vx /= distance;vy /= distance;vz /= distance
        return vx,vy,vz


    cdef float _distance_to_boundary(self,float x, float v, float l):
        cdef float db
        if v == 0:
            db = 1000.
        else:
            db = (l/2.-x*copysign(1,v))/fabs(v)
        return db

    cdef void _a_photon_movement(self,int p_id):
        cdef:
            float px,py,pz
            float vx,vy,vz
            int adx,ady,adz
            float w

            float ma, ms, mt, ni, nt
            float ai, at, Ra
            float dby,dbx,dbz,db_min,l
            int index,val_i,val_xi, val_yi, val_zi
            float val_f,val_xf, val_yf, val_zf

            int flag_1,flag_2

        px = self.p[0][p_id]; py = self.p[1][p_id]; pz = self.p[2][p_id]
        vx = self.v[0][p_id]; vy = self.v[1][p_id]; vz = self.v[2][p_id]
        adx = self.add[0][p_id]; ady = self.add[1][p_id]; adz = self.add[2][p_id]
        w = self.w[p_id]

        l = self.voxel_space
        with nogil:
          flag_1 = 1
          while flag_1:
              s = -log(self.random_uniform())
              flag_2 = 1
              while flag_2:
                  ma = self._getAbsorptionCoeff(adx,ady,adz)
                  ms = self._getScatteringCoeff(adx,ady,adz)
                  mt = ma + ms
                  s /= mt
                  dbx = self._distance_to_boundary(px,vx,l)
                  dby = self._distance_to_boundary(py,vy,l)
                  dbz = self._distance_to_boundary(pz,vz,l)

                  if dbz < dbx and dbz < dby:
                      db_min = dbz
                      index = 2
                  elif dby < dbx and dby < dbz:
                      db_min = dby
                      index = 1
                  elif dbx < dby and dbx < dbz:
                      db_min = dbx
                      index = 0
                  val_f = s-db_min

                  if val_f >= 0:
                      px,py,pz = self._p_movement_to_bouder(
                          index,px,py,pz,vx,vy,vz,db_min,l)
                      s -= db_min

                      ni = self._getReflectiveIndex(adx,ady,adz)
                      val_xi, val_yi, val_zi = self._get_next_add(
                          index,adx,ady,adz,vx,vy,vz)
                      nt = self._getReflectiveIndex(val_xi,val_yi,val_zi)

                      if ni != nt:
                          ai = fabs(self._get_index_val(index,vx,vy,vz))
                          ai = acos(ai)
                          val_f = asin(nt/ni)
                          if ai < val_f:
                              val_f = self.random_uniform()
                              self.logger_float[p_id] = val_f
                              at = asin(sin(ai)*(ni/nt))
                              Ra = val_f - 0.5*((sin(ai-at)/sin(ai+at))**2\
                                                +(tan(ai-at)/tan(ai+at))**2)
                          else:
                              Ra = -1

                          if Ra <= 0: #Internally reflect
                              val_xf,val_yf,val_zf = self._create01val(index,-1.,1.)
                              vx *= val_xf; vy *= val_yf; vz *= val_zf

                          else: #Transmit
                              adx = val_xi; ady = val_yi; adz = val_zi

                              val_f = cos(ai)
                              vx,vy,vz = self.transmit_v(index,vx,vy,vz,ni/nt,val_f)

                              val_xf,val_yf,val_zf = self._create01val(index,-1.,1.)
                              px *= val_xf; py *= val_yf; pz *= val_zf

                      else:
                          adx = val_xi; ady = val_yi; adz = val_zi

                          val_xf,val_yf,val_zf= self._create01val(index,-1.,1.)
                          px *= val_xf; py *= val_yf; pz *= val_zf

                      s *= mt
                      val_i = self.voxel_model[adx][ady][adz]
                      if val_i < 0:
                          self._save_photon(p_id,px,py,pz,adx,ady,adz,vx,vy,vz,w)
                          flag_1 = 0
                          break
                  else:
                      px += vx*s; py += vy*s; pz += vz*s
                      flag_2 = 0
                      break

              g = self._getAnisotropyCoeff(adx,ady,adz)
              vx,vy,vz = self.vectorUpdate( vx, vy, vz, g)
              w = self._wUpdate(w,ma,mt,px,py,pz,adx,ady,adz)
              if w <= 0.0001:
                  w = self._russianRoulette(w)
                  if w == 0.:
                      self._save_vanish_photon(p_id)
                      flag_1 = 0
                      break

    cdef (float,float,float) transmit_v(
        self,int index, float vx,float vy, float vz,
        float ni_by_nt,float cos_ai
        ):
        if index == 0:
            vx = cos_ai*copysign(1,vx)
            vy *= ni_by_nt; vz *= ni_by_nt
        elif index ==1:
            vy = cos_ai*copysign(1,vy)
            vx *= ni_by_nt; vz *= ni_by_nt
        elif index == 2:
            vz = cos_ai*copysign(1,vz)
            vx *= ni_by_nt; vy *= ni_by_nt
        return vx,vy,vz

    cdef (float,float,float) _p_movement_to_bouder(
        self,int index, float px, float py, float pz,
        float vx, float vy, float vz, float db_min, float l
        ):
        if index == 0:
            px = l*copysign(1,vx)/2.
            py += vy*db_min; pz += vz*db_min
        elif index ==1:
            py = l*copysign(1,vy)/2.
            px += vx*db_min; pz += vz*db_min
        elif index == 2:
            pz = l*copysign(1,vz)/2.
            px += vx*db_min; py += vy*db_min
        return px,py,pz


    cdef float _getAbsorptionCoeff(self, int x, int y, int z):
        pass

    cdef float _getScatteringCoeff(self, int x, int y, int z):
        pass

    cdef float _getAnisotropyCoeff(self, int x, int y, int z):
        pass

    cdef float _getReflectiveIndex(self, int x, int y, int z):
        pass


# =============================================================================
# Public montecarlo class
# =============================================================================
cdef class VoxelPlateMonteCarlo(BaseVoxelMonteCarlo):
    cdef:
        np.ndarray ma,ms,g,n

    def setParams(self,
                  np.ndarray[ndim = 1, dtype=DTYPE_t] ma,
                  np.ndarray[ndim = 1, dtype=DTYPE_t] ms,
                  np.ndarray[ndim = 1, dtype=DTYPE_t] g,
                  np.ndarray[ndim = 1, dtype=DTYPE_t] n):
        self.ma = ma
        self.ms = ms
        self.g = g
        self.n = n
    def getParams(self):
        return {
            "ma":self.ma,
            "ms":self.ms,
            "g":self.g,
            "n":self.n,
            }

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
        cdef int index = int(self.voxel_model[x][y][z])
        index += 1
        return self.n[index]


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
