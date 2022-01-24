import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

__all__ = [
'correlationLine',
]

def correlationLine(x,y,score = 'r'):
    x = np.array(x).flatten()
    y = np.array(y).flatten()

    #相関
    if score =='r':
        slope, intercept, r_value, _, _  = stats.linregress(x,y)
        r, p = stats.pearsonr(x,y)
        print(stats.spearmanr(x,y))
        label_ = "r = "+str(round(r_value,3))
        print(label_)
        #print("p = %s"%p)

    if score == 'r2':
        r_value = metrics.r2_score(x,y)
        label_ = "$r^2$ = "+str(round(r_value,3))
        print('pearsonr:',stats.pearsonr(x,y))
    ysub = np.poly1d(np.polyfit(x,y,1))(x)
    xx = [x.min(),x.max()]
    yy = [ysub.min(),ysub.max()]
    if r_value < 0:
        yy = [ysub.max(),ysub.min()]
    plt.plot(xx,yy,"--",color="0.2",label = label_)
