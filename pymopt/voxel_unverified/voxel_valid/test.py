from pymopt.voxelslow import VoxelPlateModel as VPM_cpu_slow
from scipy import stats
import numpy as np
import pandas as pa
import time

import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style("ticks", {'grid.linestyle': '--'})
import warnings
warnings.filterwarnings('ignore')

def radialDistance(p,w,nn,dr):
    alpha = np.array([(i)*dr for i in range(nn+1)])
    da = np.array([2*np.pi*(i+0.5)*dr**2 for i in range(nn)])
    r = np.sqrt(p[0]**2+p[1]**2)
    Rdr = []
    for i in range(nn):
        index = np.where((alpha[i]<r)&(alpha[i+1]>=r))[0]
        Rdr.append(w[index].sum())
    Rdr = np.array(Rdr)/(da*nPh)
    return alpha[:-1],Rdr

def radialDistance(p,w,nn,dr,nPh):
    alpha = np.array([(i)*dr for i in range(nn+1)])
    da = np.array([2*np.pi*(i+0.5)*dr**2 for i in range(nn)])
    r = np.sqrt(p[0]**2+p[1]**2)
    Rdr = []
    for i in range(nn):
        index = np.where((alpha[i]<r)&(alpha[i+1]>=r))[0]
        Rdr.append(w[index].sum())
    Rdr = np.array(Rdr)/(da*nPh)
    return alpha[:-1],Rdr

def AngularyResolved(v,w,nn):
    da = np.pi/(2*nn)
    alpha = np.array([(i+0.5)*da for i in range(nn+1)])
    alpha2 = np.array([(i)*da for i in range(nn+1)])
    do = 4*np.pi*np.sin(alpha)*np.sin(da/2)
    at = np.arccos(np.sign(v[2])*(v[2]))
    Rda = []
    for i in range(nn):
        index = np.where((alpha2[i]<at)&(alpha2[i+1]>=at))[0]
        Rda.append(w[index].sum())
    Rda = np.array(Rda)/(do[:-1]*nPh)
    return alpha[:-1],Rda

def linedistance(p,w,nn,dr,nPh,y_range=5):
    alpha = np.array([(i)*dr for i in range(nn+1)])
    da = np.ones(nn)*dr*y_range*2
    ind = np.where((np.abs(p[1])<y_range))[0]
    p = p[:,ind].copy()
    r = p[2]
    Rdr = []
    for i in range(nn):
        index = np.where((alpha[i]<r)&(alpha[i+1]>=r))[0]
        Rdr.append(w[index].sum())
    Rdr = np.array(Rdr)/(da*nPh)
    return alpha[:-1],Rdr
nPh = 100000
nn = 200
xy_size = 20
dr = xy_size/(2*nn)

nn_ = 200
dr_ = xy_size/(nn_)
params = {
    'thickness':[xy_size],
    'ms':[0.9],
    'ma':[0.02374],
    'g':[0.9],
    'n':[1.],
    'n_air':1.,
    'x_size':xy_size,'y_size':xy_size,
    'voxel_space':20
}
model = VPM_cpu_slow(nPh = nPh,threadnum = 128)
model.build(**params)
model = model.start()

rez_cpu_slow = model.get_result()
margin = 1e-8
ind = np.where((rez_cpu_slow["v"][2]<0)&(rez_cpu_slow["p"][2]<margin))[0]
alphaRd,Rd_cpu_slow = radialDistance(rez_cpu_slow["p"][:,ind],rez_cpu_slow["w"][ind],nn,dr,rez_cpu_slow["nPh"])
ind = np.where(rez_cpu_slow["v"][2]>0&(rez_cpu_slow["p"][2]>params["thickness"][0]-margin))[0]
alphaTt,Tt_cpu_slow = radialDistance(rez_cpu_slow["p"][:,ind],rez_cpu_slow["w"][ind],nn,dr,rez_cpu_slow["nPh"])
ind = np.where((rez_cpu_slow["v"][0]<0))[0]
alpha_ssyz,Ssyz_cpu_slow = linedistance(rez_cpu_slow["p"][:,ind],rez_cpu_slow["w"][ind],nn_,dr_,rez_cpu_slow["nPh"],y_range=2.5)
