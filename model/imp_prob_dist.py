import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline as uspline

from model.structures import *
import opt_plot.bs_opt_plot as bs
import data_feed.tdameritrade as tda

from scipy.signal import savgol_filter as sf

import logging
logger = logging.getLogger('VOL')

'''
https://reasonabledeviations.com/2020/10/01/option-implied-pdfs/
'''

def get_option_implied_prob_dist(opt):  
    grps = opt.groupby(by=kind)
    prob_dic = {}
    for i,grp in grps:
        spot = grp.iloc[0].spot
        if grp.iloc[0].kind.lower() =='put':
            df = grp[grp.strike <= spot + 5]
        else:
            df = grp[grp.strike >= spot - 5]

        butter_list = [df[i:i+3] for i in range(len(df) - 2)]
        for bl in butter_list:
            k0 = bl.iloc[0].strike ; k1 = bl.iloc[1].strike ;
            k2 = bl.iloc[2].strike
            if abs(k0 - k1) == abs(k1 - k2):
                # assume buy at ask, sell at bid
                prob = bl.iloc[0].ask - 2 * bl.iloc[1].bid + bl.iloc[2].ask
                if k1 in prob_dic:
                    cur_prob = prob_dic[k1]
                    nxt_prob = prob / abs(k1 - k0)
                    prob_dic[k1] = (cur_prob + nxt_prob) / 2
                else:
                    prob_dic[k1] = prob / abs(k1 - k0)

    return pd.DataFrame(sorted(prob_dic.items()), columns=['strike','prob'])

def plot_option_implied_prob_dist(df):
    plt.cla() ; plt.clf() ;
    s = uspline(df.strike,df.prob, s = .04)
    xs = np.linspace(min(df.strike),max(df.strike),1000)
    plt.plot(xs, s(xs), color='red')

    plt.plot(df.strike, df.prob, linestyle='', marker='o')

    df_len = len(df.strike)
    win = df_len if df_len % 2 == 1 else df_len - 1
    xhat,yhat = sf((df.strike,df.prob), df_len , 4)
    plt.plot(xhat,yhat, color='blue')

    plt.xlabel('strike')
    plt.ylabel('implied prob')
    plt.show()

def get_opt_from_sym(sym):

    td = tda.TD([sym.upper()])
    td.run()
    opt = td.chains[0]

    df = get_option_implied_prob_dist(opt)
    plot_option_implied_prob_dist(df)

