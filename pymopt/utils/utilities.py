
import json
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import os


def calTime(end, start):
    elapsed_time = end - start
    q, mod = divmod(elapsed_time, 60)
    if q < 60:
        print('Calculation time: %d minutes %0.3f seconds.' % (q, mod))
    else:
        q2, mod2 = divmod(q, 60)
        print('Calculation time: %d h %0.3f minutes.' % (q2, mod2))

def check_folder(folder_dir):
    if not os.path.exists(folder_dir):
        os.makedirs(folder_dir)

def set_params(data,keys,*initial_data, **kwargs):
    for dictionary in initial_data:
        for key in dictionary:
            if not key in keys:
                raise KeyError(key)
            data[key] = dictionary[key]

    for key in kwargs:
        if not key in keys:
            raise KeyError(key)
        data[key] = kwargs[key]

class ToJsonEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)

def correlationLine(x,y):
    x = np.array(x).flatten()
    y = np.array(y).flatten()
    #X = np.linspace(x.min(), x.max(),int((x.max()-x.min())/10))

    #相関
    slope, intercept, r_value, _, _  = stats.linregress(x,y)
    r, p = stats.pearsonr(x,y)
    print(stats.spearmanr(x,y))
    p_str = ""
    if p >= 0.05:
        p_str = "p = " + str(round(p,3))
    elif p < 0.05 and p >= 0.01:
        p_str = "p < 5%"
    elif p < 0.01 and p >=0.005:
        p_str = "p < 1%"
    elif p < 0.005 and p >= 0.001:
        p_str = "p < 0.5%"
    elif p < 0.001 and 0.0001:
        p_str = "p < 0.1%"
    else:
        p_str = "p < 0.01%"

    label_ = "r = "+str(round(r_value,3)) + ", " + p_str
    print(label_)
    #print("p = %s"%p)
    ysub = np.poly1d(np.polyfit(x,y,1))(x)
    xx = [x.min(),x.max()]
    yy = [ysub.min(),ysub.max()]
    if r < 0:
        yy = [ysub.max(),ysub.min()]
    plt.plot(xx,yy,"--",color="0.2",label = label_)
