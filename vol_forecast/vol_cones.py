import json
import pandas as pd
import sys
from datetime import datetime,date, timedelta
import numpy as np
import math
import matplotlib.pyplot as plt
from scipy.stats import percentileofscore as pos

import logging
logger = logging.getLogger('VOL')

class VolCone():
    # returns are log single day returns 
    # vc = VolCone(df['1day'].dropna())
    def __init__(self, returns):
        self.df = returns
        self.cones = None
        '''
        to get vol that corresponds to quantile .1
            self.quantile[20].quantile(.1)
        to get quantile that corresponds to vol .5 
            pos(self.quantile[20], .5)

        '''
        self.quantile = None # dic to be used for determining quantiles (  )

    def vol_for_quantile(quant=20, perc):
        return self.quantile[quant].quantile(perc)

    def quantile_for_vol(quant=20, vol):
        return pos(self.quantile[quant], vol)

    def create(self):
        logger.info('')
        windows = [10,15,20,25,30,40,50,60,70,80,90,100] # trading days
        dic = {}
        quant = {}
        for win in windows:
            result = self.df.rolling(
                window=win,
                center=False
            ).std() * math.sqrt(252)
            result = result.dropna()
            _mean = result.mean()
            _std = result.std()
            dic[win] = {'min':result.min(), 'mean':_mean, 'max':result.max(),
                'stdup1':_mean+_std,'stddown1':_mean-_std,
                'stdup2':_mean+2*_std,'stddown2':_mean-2*_std}
            # allows you to call self.quantile[x].quantile(y)
            quant[win] = result

        self.quantile = quant 
        self.cones = dic 

    def plot(self):

        maxs  = []
        means = []
        mins  = []
        stdup1 = []
        stdup2 = []
        stddown1 = []
        stddown2 = []
        for k,v in self.cones.items():
            mins.append(v['min'])
            means.append(v['mean'])
            maxs.append(v['max'])
            stdup1.append(v['stdup1'])
            stdup2.append(v['stdup2'])
            stddown1.append(v['stddown1'])
            stddown2.append(v['stddown2'])


        x_axis = list(self.cones.keys())
        plt.plot(x_axis , mins, '-rD', label='mins') # red
        plt.plot(x_axis , means, '-bD', label='means') # blue
        plt.plot(x_axis , maxs, '-gD', label='maxs') # green 

        plt.plot(x_axis , stdup1, '--m', linewidth=1, label='1std_dev up') # green 
        plt.plot(x_axis , stdup2, '--k', linewidth=1, label='2std_dev up') # green 
        plt.plot(x_axis , stddown1, '--m', linewidth=1, label='1std_dev down') # green 
        plt.plot(x_axis , stddown2, '--k', linewidth=1, label='2std_dev down') # green 

        plt.xticks(x_axis)
        plt.legend()
        plt.show()


