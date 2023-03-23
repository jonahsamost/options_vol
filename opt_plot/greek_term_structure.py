from opt_plot.bs_opt_plot import *
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# super simple just to see what it looks like
def plot_greek_term_structure(opttype=LC, vol=.25):

    out_delta = {}
    out_theta = {}
    out_vega = {}
    out_gamma = {}
    s = 100 ; k = 100 ; r = 0 ; vol = vol ; 
    ot = opttype
    for i in range(1,100):
        out_delta[i] = delta(s,k,r,vol,i/365,ot)
        out_theta[i] = theta(s,k,r,vol,i/365,ot)
        out_vega[i]  = vega(s,k,r,vol,i/365)
        out_gamma[i] = gamma(s,k,r,vol,i/365)

    df_delta = pd.DataFrame(out_delta.items(), columns=['day', 'delta'])
    df_theta = pd.DataFrame(out_theta.items(), columns=['day', 'theta'])
    df_vega = pd.DataFrame(out_vega.items(), columns=['day', 'vega'])
    df_gamma = pd.DataFrame(out_gamma.items(), columns=['day', 'gamma'])

    plt.cla() ; plt.clf() ;
    plt.plot(df_delta.day, df_delta.delta)
    plt.title('delta')
    plt.show()

    plt.cla() ; plt.clf() ;
    plt.plot(df_vega.day, df_vega.vega)
    plt.title('vega')
    plt.show()

    plt.cla() ; plt.clf() ;
    plt.plot(df_theta.day, df_theta.theta)
    plt.title('theta')
    plt.show()

    plt.cla() ; plt.clf() ;
    plt.plot(df_gamma.day, df_gamma.gamma)
    plt.title('gamma')
    plt.show()
