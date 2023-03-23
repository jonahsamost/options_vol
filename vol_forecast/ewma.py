import sys
from datetime import datetime,date
from timeit import default_timer as timer

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np

class EWMA():
    # def __init__(self, symbol, db, start=None, end=None):
    def __init__(self, df , lam=.94):
        out = 0 
        cnt_needed = 0
        for i in range(1000): 
            out += (1-lam)*(lam**i) 
            if out > .99: 
                cnt_needed = i
                break 

        self.inited = False
        for cname,cdata in df.iteritems(): 
            if cname=='date' or cname=='rets':
                continue
            if len(cdata.dropna()) < cnt_needed:
                return 

        self.inited = True
        self.df = df
        self.lam = lam 
        self.cnt = cnt_needed

    def get_vol(self):

        weights = []
        for i in range(self.cnt):
            weights.append((1-self.lam) * (self.lam ** i))
        weights = np.array(weights[::-1]) # most weighted for most recent returns/vols

        term2 = sum(self.df.rets[len(self.df.rets) - self.cnt :] * (1 - weights))

        out = pd.DataFrame()
        for cname,cdata in df.iteritems(): 
            if cname=='date' or cname=='rets':
                continue
            vols = cdata.dropna()
            vols = vols[len(vols) - self.cnt :]
            cur_vol = math.sqrt(sum(weights * (vols**2)) + term2)
            out[cname] = cur_vol

        return out 


'''
In [184]: for ll in range(1,100,1): 
     ...:     l = ll/100. 
     ...:     out = 0 
     ...:     for i in range(1000): 
     ...:         out += (1-l)*(l**i) 
     ...:         if out > .99: 
     ...:             if i >= 30: 
     ...:                 print(i,l) 
     ...:             break 
     ...:              
     ...:                                                                                                                                                                                                          
30 0.86
33 0.87
36 0.88
39 0.89
43 0.9
48 0.91
55 0.92
63 0.93
74 0.94
89 0.95
112 0.96
151 0.97
227 0.98
458 0.99
''' 
