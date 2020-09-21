
cdef float random_uniform():
    cdef float random = float(rand())
    cdef float randmax = float(RAND_MAX)
    return random/randmax
cdef void _a_photon_movement(int p_id):
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
    l = self.voxel_space
    
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