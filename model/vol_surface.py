from data_feed import tiingo, Time
from vol_models.YangZhang import get_estimator
import pandas as pd
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline as uspline

from model.structures import *
from data_feed.tdameritrade import *

class Model:
    
    '''
    self.dfs : dataframe 
    '''
    def __init__(self, dfs):
        self.dfs = dfs

    def chart_chains(self):
        dates = []
        for df in self.dfs:
            _dte = df.iloc[0].dte
            _desc = df.iloc[0].desc
            _spot = df.iloc[0].spot

            zdf = df.drop([dte,desc,spot],axis=1)
            pcs = [x for _, x in zdf.groupby(kind)] # put or call
            c = pcs[0] if pcs[0].iloc[0].kind==ContractType.CALL.value else pcs[1]
            p = pcs[1] if c is pcs[0] else pcs[0]
            c.drop([strike],axis=1,inplace=True)

            c.reset_index(drop=True, inplace=True) 
            p.reset_index(drop=True, inplace=True)  

            chart = pd.concat([c,p],axis=1)
            chart[dte] = _dte
            dt = datetime.strptime(' '.join(_desc.split()[1:4]), '%b %d %Y')
            chart[desc] = datetime.strftime(dt,'%Y-%m-%d')

            below = chart[chart[strike] <= _spot]
            above = chart[chart[strike] > _spot]
            line = pd.DataFrame([['-'] * below.shape[1]], columns=below.columns)
            line[strike] = _spot
            
            chain = pd.concat([below,line,above])

            dates.append(chain)

        return dates


    def fit_skew_surface(self):
        plt.cla()
        plt.clf()

        colormap = plt.cm.nipy_spectral
        colors = [colormap(i) for i in np.linspace(0, 1,len(self.dfs))]
        fig = plt.figure(1)
        fig.gca().set_prop_cycle('color', colors)

        ax = plt.gca()

        for opt in self.dfs:
            desc = opt.desc.iloc[0]
            exp = ' '.join(desc.split()[1:4])
            exp += ' Weekly' if desc.find('Weekly') != -1 else ''

            spot = opt.spot.iloc[0]
            calls = opt[(opt.strike > spot) & (opt.kind == 'CALL')]
            puts = opt[(opt.strike < spot) & (opt.kind == 'PUT')]
            skew = pd.concat([puts,calls])
            
            strikes = skew.strike
            vols    = skew.vol

            # interpolate spline through points
            s = uspline(strikes,vols,s=3)
            xs = np.linspace(min(strikes),max(strikes),10000)
            ys = s(xs)

            plt.xticks(strikes)
            color = next(ax._get_lines.prop_cycler)['color']
            plt.plot(xs,ys,color=color)
            plt.plot(strikes,vols,'x',label=f'{exp}', color=color)

        plt.legend(loc='upper right')
        plt.show()

'''
opts = win.gui_lay.td.opts
mod = Model(opts)
chart = mod.chart_chains()
mod.fit_skew_surface()
'''
