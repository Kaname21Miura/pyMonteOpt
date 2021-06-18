
## *** All parameters should be defined in millimeters ***

import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pa
from tqdm import tqdm
import gc
from ..utils.utilities import calTime,set_params

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import bz2,pickle,json
__all__ = ['OBD']
# =============================================================================
# Optical parts
# =============================================================================

#### スリットの親クラス ####
class Slit(object):
    def __init__(self,outerD,slitD,width,thickness,position):
        self.outerD = outerD/2
        self.slitD = slitD
        self.width = width
        self.thickness = thickness
        self.position = position
        self.d_out = self.slitD/2 + self.width/2
        self.d_in = self.slitD/2 - self.width/2
        self.position = position
        self.front_z = position
        self.back_z = position - thickness

    #光子の衝突位置
    def hittingPotision(self,p,v,posit):
        a = (posit-p[2])/v[2]
        return p + a*v

    #slitで弾かれる光子を削除
    def delBySlit(self,p,v,w):
        pp = np.sqrt(p[0]**2+p[1]**2)
        index_ = np.where((pp<self.d_out)&(pp>self.d_in))[0].tolist()
        #index_ = np.where(pp>self.d_in)[0].tolist()
        p = p[:,index_]
        v = v[:,index_]
        w = w[index_]
        return p,v,w

    #Slit全体の動きを定義
    def opticalAnalysis(self,p,v,w):
        p = self.hittingPotision(p,v,self.front_z)
        p,v,w = self.delBySlit(p,v,w)
        p = self.hittingPotision(p,v,self.back_z)
        p,v,w = self.delBySlit(p,v,w)
        return p,v,w

#### レンズの親クラス ####
class Lens (object):
    def __init__(self,outerD,ct,r,n,position,typ):
        self.outerD = outerD/2
        self.ct = ct
        self.r = r
        self.n = n
        self.position = position
        self.typ = typ
        if self.typ == "Outward":
            self.center = self.position - (self.ct - self.r)
        elif self.typ == "Inward":
            self.center = self.position + (self.ct - self.r)
        else:
            print("レンズの向きが入力されていないか間違っています")

    #レンズ平面での屈折と反射と位置の変更
    def updaterAtPlano(self,p,v,w):#######
        v = self.normVector(v)
        p = self.hittingPointPlano(p,v)
        p,v,w = self.deleteOutOfLens(p,v,w)
        nn = self.orthVectorPlano(p)
        if self.typ == "Inward":#レンズ2
            p,v,w = self.intLensToAir(p,v,w,nn)
        elif self.typ =="Outward":#レンズ1
            p,v,w = self.intAirToLens(p,v,w,nn)
        return p,self.normVector(v),w

    #球面レンズ曲面での屈折、反射、光子位置の変更
    def updaterAtConvex(self,p,v,w):
        v = self.normVector(v)
        p = self.hittingPointConvex(p,v)
        p,v,w = self.deleteOutOfLens(p,v,w)
        nn = self.orthVectorConvex(p)
        if self.typ == "Inward":#レンズ2
            p,v,w = self.intAirToLens(p,v,w,nn)
        elif self.typ == "Outward":#レンズ1
            p,v,w = self.intLensToAir(p,v,w,nn)
        return p,self.normVector(v),w

    #レンズ外に飛び出した光子の削除
    def deleteOutOfLens(self,p,v,w):
        pp = np.sqrt(p[0]**2 + p[1]**2)
        index_ = np.where(pp<=self.outerD)[0].tolist()
        return p[:,index_],v[:,index_],w[index_]

    #レンズ平面部の当たる光子の位置
    def hittingPointPlano(self,p,v):
        a = (self.position-p[2])/v[2]
        return p+a*v

    #レンズ球面部の当たる光子の位置
    def hittingPointConvex(self,p,v): ##########
        pp = p[2]
        pp = pp-self.center
        A = v[0]**2 + v[1]**2 + v[2]**2
        B = 2*(p[0]*v[0]+p[1]*v[1]+pp*v[2])
        C = p[0]**2 + p[1]**2 + pp**2-self.r**2
        if self.typ == "Inward":#レンズ2
            t = (-B-np.sqrt(B**2-4*A*C))/(2*A)
        elif self.typ == "Outward":#レンズ1
            t = (-B+np.sqrt(B**2-4*A*C))/(2*A)
        p = p + np.multiply(v,t)
        return p

    #空気→レンズの順での光入射
    def intAirToLens(self,p,v,w,nn):
        cos_ai = self.cosAi(nn,v)
        nt = self.n
        ni = 1
        a = ni/nt
        Ra = np.zeros_like(cos_ai)
        ai = np.arccos(cos_ai)
        cos_at = self.cosAt(a,cos_ai)
        at = np.arccos(cos_at)
        Ra = self.Rai(ai,at)
        index_ = np.where(Ra>0)[0].tolist()
        p = p[:,index_]
        w = w[index_]
        g = cos_at[index_]-a*cos_ai[index_]
        v = a*v[:,index_]+g*nn[:,index_]
        return p,v,w

    #レンズ→空気の順での光入射
    def intLensToAir(self,p,v,w,nn):
        cos_ai = self.cosAi(nn,v)
        nt = 1
        ni = self.n
        a = ni/nt
        Ra = np.zeros_like(cos_ai)
        ai = np.arccos(cos_ai)
        index_ = np.where(ai >= np.arcsin(nt/ni))[0].tolist()
        Ra[index_] = -1
        index_ = np.delete(np.arange(Ra.shape[0]),index_)
        cos_at = self.cosAt(a,cos_ai)
        at = np.arccos(cos_at)
        Ra[index_] = self.Rai(ai[index_],at[index_])
        index_ = np.where(Ra>0)[0].tolist()
        p = p[:,index_]
        w = w[index_]
        g = cos_at[index_]-a*cos_ai[index_]
        v = a*v[:,index_]+g*nn[:,index_]
        return p,v,w

    def normVector(self,v):
        return v/np.sqrt(v[0]**2+v[1]**2+v[2]**2)

    #レンズ平面部の法線ベクトルの取得
    def orthVectorPlano(self,p):
        nn = np.zeros_like(p)
        nn[2] = -1
        return self.normVector(nn)

    #レンズ球面部の法線ベクトルの取得
    def orthVectorConvex(self,p):
        d = np.sqrt(self.r**2-np.add(p[0]**2,p[1]**2))
        dx = np.divide(-p[0],d)
        dy = np.divide(-p[1],d)

        if self.typ == "Inward":#レンズ2
            nn = np.array([dx,dy,-np.ones(p[2].size)])
        elif self.typ == "Outward":#レンズ1
            nn = np.array([-dx,-dy,-np.ones(p[2].size)])
        return self.normVector(nn)

    #フルネルの公式から得られる反射率
    def Rai(self,ai,at):
        Ra = np.random.rand(ai.size)-0.5*(np.add((np.sin(ai-at)/np.sin(ai+at))**2,
                                                 (np.tan(ai-at)/np.tan(ai+at))**2))
        return Ra

    def cosAt(self,snel,cos):
        return np.sign(cos)*np.sqrt(1-(snel**2)*(1-cos**2))

    def cosAi(self,nn,v):
        return (nn[0]*v[0]+nn[1]*v[1]+nn[2]*v[2])


##### 曲面が負の方向を向いているレンズ #####
class Lens1(Lens):
    def __init__(self,outerD,ct,r,n,position):
        super().__init__(outerD,ct,r,n,position,typ = "Outward")

    def opticalAnalysis(self,p,v,w):
        p,v,w = self.updaterAtPlano(p,v,w)
        p,v,w = self.updaterAtConvex(p,v,w)
        return p,v,w

##### 曲面が正の方向を向いたレンズ #####
class Lens2(Lens):
    def __init__(self,outerD,ct,r,n,position):
        super().__init__(outerD,ct,r,n,position,typ = "Inward")

    def opticalAnalysis(self,p,v,w):
        p,v,w = self.updaterAtConvex(p,v,w)
        p,v,w = self.updaterAtPlano(p,v,w)
        return p,v,w


#### フォトダイオードのクラス ####
class Photodiode(object):
    def __init__(self,d,position):
        self.d = d
        self.r = d/2
        self.position = position
        self.count = 0
        self.record_w = 0

    #衝突位置の計算
    def hittingPotision(self,p,v):
        a = (self.position-p[2])/v[2]
        return p + a*v

    #Pdで観測される光子の層エネルギー量
    def catcherInThePhotodiode(self,p,v,w):
        p = self.hittingPotision(p,v)
        pp = np.sqrt(p[0]**2+p[1]**2)
        index_ = np.where(pp<self.r)[0].tolist()
        return np.sum(w[index_]),p


# =============================================================================
# OBD class
# =============================================================================
class OBD:
    def __init__(self):
        #self.dtype = 'float64'
        self.params ={
            'start':-10,'end':65,'split':1,
            'outerD_1':50,'efl_1':100,'bfl_1':93.41,
            'ct_1':10,'et_1':3.553,'r_1':51.68,'n_1':1.517,
            'outerD_2' : 50,'efl_2' : 50,'bfl_2' : 43.28,
            'ct_2':12,'et_2':3.01,'r_2':39.24,'n_2':1.758,
            'slit_outerD':50,'slit_D':20,'slit_width':2,'slit_thickness':5,
            'd_pd':3,
            'distance_2slits':37,'pd_poit_correction':0,
        }
        self.keys_params  = list(self.params.keys())
        self.data = {'p':0,'v':0,'w':0,'nPh':1000}
        self.keys_data = list(self.data.keys())
        self.nPh = 1000

    def set_params(self,*initial_data, **kwargs):
        set_params(self.params,self.keys_params,*initial_data, **kwargs)

    def set_monte_data(self,*initial_data, **kwargs):
        set_params(self.data,self.keys_data,*initial_data, **kwargs)
        self.nPh = self.data['nPh']

    def open_pklbz2_file(self,path):
        with bz2.open(path, 'rb') as fp:
            data = pickle.loads(fp.read())
        return data

    def open_jason_file(self,path):
        with open(path, 'r') as fp:
            json_load = json.load(fp)
        return json_load

    def load_file(self,path):
        self.data = self.open_pklbz2_file(path)
        self.nPh = self.data['nPh']

    def start(self,show_graph = False):
        res = self.opticalAnalysisMeth()
        if show_graph:
            plt.figure(figsize=(8,6),dpi=90)
            plt.plot(res[0],np.log10(res[1]/self.nPh),"-",c = "k")
            plt.xlabel("$Z\ [mm]$")
            plt.ylabel("$log_{10}(I/I_0)$")
            plt.show()
        self.result = pa.DataFrame(res,index=["Z","int"]).T
        self.result["log(int)"] = np.log10(res[1]/self.nPh)

    def get_result(self):
        return self.result

    def save_result(self,path):
        fname_save = path+"_opt.csv"
        self.result.to_csv(fname_save, index=False)

    #光学系を定義
    def opticalUnit(self,Z,p,v,w):
        z_lens1 = -self.params['bfl_1'] + Z
        z_lens2 = z_lens1 - self.params['ct_1']\
        -self.params['slit_thickness']*2-self.params['ct_2']-self.params['distance_2slits']
        z_slit1 = z_lens1 - self.params['ct_1']
        z_slit2 = z_lens2 + self.params['ct_2']+ self.params['slit_thickness']
        z_pd = z_lens2 - self.params['bfl_2']-self.params['pd_poit_correction']

        #レンズとスリット、フォトダイオードのオブジェクトをそれぞれ生成
        #outerD,ct,r,n,position
        lens_1 = Lens1(
            self.params['outerD_1'],self.params['ct_1'],
            self.params['r_1'],self.params['n_1'],z_lens1
            )
        lens_2 = Lens2(
            self.params['outerD_2'],self.params['ct_2'],
            self.params['r_2'],self.params['n_2'],z_lens2
            )
        slit_1 = Slit(
            self.params['slit_outerD'],self.params['slit_D'],
            self.params['slit_width'],self.params['slit_thickness'],z_slit1
            )
        slit_2 = Slit(
            self.params['slit_outerD'],self.params['slit_D'],
            self.params['slit_width'],self.params['slit_thickness'],z_slit2
            )
        pd = Photodiode(self.params['d_pd'],z_pd)
        #解析
        p,v,w = lens_1.opticalAnalysis(p,v,w)
        p,v,w = slit_1.opticalAnalysis(p,v,w)
        p,v,w = slit_2.opticalAnalysis(p,v,w)
        p,v,w = lens_2.opticalAnalysis(p,v,w)
        intdist,p = pd.catcherInThePhotodiode(p,v,w)
        return intdist

    #光学系の挙動を定義
    def opticalAnalysisMeth(self):
        start_ = time.time()
        step = np.arange(
            self.params['start'],
            self.params['end'],
            self.params['split'])
        rd_index = np.where(self.data['v'][2]<0)[0]
        p = self.data['p'][:,rd_index]
        p[2] = 0
        v = self.data['v'][:,rd_index]
        w = self.data['w'][rd_index]
        intdist = np.empty_like(step)
        for (i,Z) in enumerate(tqdm(step)):
            intdist[i] = self.opticalUnit(Z,p,v,w)
        calTime(time.time(), start_)
        return step,intdist
