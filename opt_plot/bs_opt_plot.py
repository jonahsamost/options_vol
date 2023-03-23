from scipy.stats import norm 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math
import sys

import model.structures as mod_struc

'''
https://clinthoward.github.io/portfolio/2017/04/16/BlackScholesGreeks/ -- 
https://leventozturk.com/engineering/Black_Scholes/
 S: underlying stock price 
 K: Option strike price 
 r: risk free rate 
 D: dividend value 
 vol: Volatility 
 T: time to expiry (assumed that we're measuring from t=0 to T) 
'''

#TODO add 'Black' and 'BSM' greeks too from vollib 

CALL = 'call'
PUT  = 'put'
LONG = 'long'
SHORT = 'short'

LC = 'Long Call'
SC = 'Short Call'
LP = 'Long Put'
SP = 'Short Put'

TRADING_DAYS = 252.0
IMPLIED_VOL_DAYS = 252.0
YEAR_DAYS = 365.

def otype_shortcut(otype):
    if otype==LC:
        return 'LC'
    elif otype==SC:
        return 'SC'
    elif otype==LP:
        return 'LP'
    elif otype==SP:
        return 'SP'
    return ''



def d1_calc(S, K, r, vol, t):
    a = (np.log(S/K) + (r + 0.5 * vol**2)*t)
    b = (vol*np.sqrt(t))
    return a/b

def d2_calc(d1, vol, t):
    return d1 - vol * np.sqrt(t)


### FIRST ORDER GREEKS ###

def delta(S, K, r, vol, t, otype):
    d1 = d1_calc(S, K, r, vol, t)
    if(otype in [LC,SC,CALL]):  return norm.cdf(d1)
    elif(otype in [LP,SP,PUT]): return -norm.cdf(-d1)

def vega(S, K, r, vol, t):
    d1 = d1_calc(S, K, r, vol, t)
    return S * norm.pdf(d1) * np.sqrt(t) 

def theta(S, K, r, vol, t, otype):
    d1 = d1_calc(S, K, r, vol, t)
    d2 = d2_calc(d1 , vol, t)

    if(otype in [LC,SC,CALL]):  
        _ = -(S*vol*norm.pdf(d1) / (2*np.sqrt(t))) - r*K*np.exp(-r*t)*norm.cdf(d2) 
    elif(otype in [LP,SP,PUT]):
        _ = -(S*vol*norm.pdf(d1) / (2*np.sqrt(t))) + r*K*np.exp(-r*t)*norm.cdf(-d2)
    return _ / TRADING_DAYS

def rho(S, K, r, vol, t, otype):
    d1 = d1_calc(S, K, r, vol, t)
    d2 = d2_calc(d1 , vol, t)
    
    if(otype in [LC,SC,CALL]): return t*K*np.exp(-r*t)*norm.cdf(d2)
    elif(otype in [LP,SP,PUT]): return -t*K*np.exp(-r*t)*norm.cdf(-d2)

### SECOND ORDER GREEKS ###

def gamma(S, K, r, vol, t):
    d1 = d1_calc(S, K, r, vol, t)
    return norm.pdf(d1) / (S * vol * np.sqrt(t))

def vanna(S, K, r, vol, t):
    d1 = d1_calc(S, K, r, vol, t)
    d2 = d2_calc(d1 , vol, t)
    return norm.pdf(d1) * d2 / vol

def charm(S, K, r, vol, t):
    d1 = d1_calc(S, K, r, vol, t)
    d2 = d2_calc(d1 , vol, t)
    return -norm.pdf(d1) * ((2*r*t - d2*vol*np.sqrt(t)) / (2*t*vol*np.sqrt(t)))

def vomma(S, K, r, vol, t):
    v = vega(S,K,r,vol,t)
    d1 = d1_calc(S, K, r, vol, t)
    d2 = d2_calc(d1 , vol, t)
    return v * d1 * d2 / vol 


######## STRATEGY #########

def BS_call(S, K, r, vol, t):
    d1 = d1_calc(S, K, r, vol, t)
    d2 = d2_calc(d1 , vol, t)
    p = S * norm.cdf(d1) - K * np.exp(-r * t) * norm.cdf(d2)
    return p 

def BS_put(S, K, r, vol, t):
    d1 = d1_calc(S, K, r, vol, t)
    d2 = d2_calc(d1 , vol, t)
    return K * np.exp(-r * t) * norm.cdf(-d2) - S * norm.cdf(-d1) 


####### FIND IMPLIED VOL ######

'''
sometimes this just returns 0 cause it blows up ? 
'''
def find_vol(cur_price, S, K, T, r, opttype, *args):
    MAX_ITERATIONS = 200
    PRECISION = 1.0e-5
    sigma = 0.5
    for i in range(0, MAX_ITERATIONS):
        if math.isnan(sigma): 
            return 
        if opttype.lower() == CALL.lower(): 
            price = BS_call(S, K, r, sigma, T)
        else:                               
            price = BS_put(S, K, r, sigma, T)
        diff = cur_price - price  # our root
        if (abs(diff) < PRECISION):
            return sigma
        veg = vega(S, K, r, sigma, T) 
        if veg == 0: return 0 
        sigma = sigma + (diff/veg) # f(x) / f'(x)
    return sigma # value wasn't found, return best guess so far

def find_bid_ask_vols(opt):
    bid_vol = [] ; ask_vol = []
    for i,r in opt.iterrows():
        if r.bid:
            bv = find_vol(r.bid, r.spot, r.strike, r.dte/252., 0,r.kind)
        if r.ask:
            av = find_vol(r.ask, r.spot, r.strike, r.dte/252., 0,r.kind)

        ask_vol.append(av if av else 0)
        bid_vol.append(bv if bv else 0)
        av = bv = 0 

    opt['BidVol'] = bid_vol
    opt['AskVol'] = ask_vol


def vxo_atm_vol(first, second):
    '''
    vxo atm vol calculation
    https://rewconsulting.files.wordpress.com/2012/09/jd93.pdf
    '''
    spot = first.iloc[0].spot

    n = near_call_under_s = first[(first.strike < first.spot) & (first.kind == CALL.upper())].iloc[-1]
    near_call_under_s_vol = find_vol((n.bid + n.ask) / 2, n.spot, n.strike, n.dte / IMPLIED_VOL_DAYS, 0 , CALL)
    n = near_put_under_s  = first[(first.strike < first.spot) & (first.kind == PUT.upper())].iloc[-1]
    near_put_under_s_vol  = find_vol((n.bid + n.ask) / 2, n.spot, n.strike, n.dte / IMPLIED_VOL_DAYS, 0 , PUT)
    vol_near_under        = (near_call_under_s_vol + near_put_under_s_vol) / 2  # sigma1 x1

    n = near_call_over_s  = first[(first.strike > first.spot) & (first.kind == CALL.upper())].iloc[0]
    near_call_over_s_vol  = find_vol((n.bid + n.ask) / 2, n.spot, n.strike, n.dte / IMPLIED_VOL_DAYS, 0 , CALL)
    n = near_put_over_s   = first[(first.strike > first.spot) & (first.kind == PUT.upper())].iloc[0]
    near_put_over_s_vol   = find_vol((n.bid + n.ask) / 2, n.spot, n.strike, n.dte / IMPLIED_VOL_DAYS, 0 , PUT)
    vol_near_over         = (near_call_over_s_vol + near_put_over_s_vol) / 2    # sigma1 x2

    strike_above = near_call_over_s.strike ; strike_below = near_call_under_s.strike 
    sigma1  = vol_near_under * ((strike_above - spot) / (strike_above - strike_below))
    sigma1 += vol_near_over * ((spot - strike_below) / (strike_above - strike_below))

    n = far_call_under_s  = second[(second.strike < second.spot) & (second.kind == CALL.upper())].iloc[-1]
    far_call_under_s_vol  = find_vol((n.bid + n.ask) / 2, n.spot, n.strike, n.dte / IMPLIED_VOL_DAYS, 0 , CALL)
    n = far_put_under_s   = second[(second.strike < second.spot) & (second.kind == PUT.upper())].iloc[-1]
    far_put_under_s_vol   = find_vol((n.bid + n.ask) / 2, n.spot, n.strike, n.dte / IMPLIED_VOL_DAYS, 0 , PUT) 
    vol_far_under         = (far_call_under_s_vol + far_put_under_s_vol) / 2    # sigma2 x1

    n = far_call_over_s   = second[(second.strike > second.spot) & (second.kind == CALL.upper())].iloc[0]
    far_call_over_s_vol   = find_vol((n.bid + n.ask) / 2, n.spot, n.strike, n.dte / IMPLIED_VOL_DAYS, 0 , CALL)
    n = far_put_over_s    = second[(second.strike > second.spot) & (second.kind == PUT.upper())].iloc[0]
    far_put_over_s_vol    = find_vol((n.bid + n.ask) / 2, n.spot, n.strike, n.dte / IMPLIED_VOL_DAYS, 0 , PUT)
    vol_far_over          = (far_call_over_s_vol + far_put_over_s_vol) / 2       # sigma2 x2

    strike_above = far_call_over_s.strike ; strike_below = far_call_under_s.strike 
    sigma2  = vol_far_under * ((strike_above - spot) / (strike_above - strike_below))
    sigma2 += vol_far_over * ((spot - strike_below) / (strike_above - strike_below))

    first_dte = first.dte.iloc[0] ; sec_dte = second.dte.iloc[0]
    vxo = sigma1 * (sec_dte - 22) / (sec_dte - first_dte) # the vxo paper says 22
    vxo += sigma2 * (22 - first_dte) / (sec_dte - first_dte)
    return vxo

def get_option_atm_vol(opt):
    spot = opt.iloc[0].spot
    n = near_call_under_s = opt[(opt.strike < opt.spot) & (opt.kind == CALL.upper())].iloc[-1]
    near_call_under_s_vol = find_vol((n.bid + n.ask) / 2, n.spot, n.strike, n.dte / IMPLIED_VOL_DAYS, 0 , CALL)
    n = near_put_under_s  = opt[(opt.strike < opt.spot) & (opt.kind == PUT.upper())].iloc[-1]
    near_put_under_s_vol  = find_vol((n.bid + n.ask) / 2, n.spot, n.strike, n.dte / IMPLIED_VOL_DAYS, 0 , PUT)
    vol_near_under        = (near_call_under_s_vol + near_put_under_s_vol) / 2  # sigma1 x1

    n = near_call_over_s  = opt[(opt.strike > opt.spot) & (opt.kind == CALL.upper())].iloc[0]
    near_call_over_s_vol  = find_vol((n.bid + n.ask) / 2, n.spot, n.strike, n.dte / IMPLIED_VOL_DAYS, 0 , CALL)
    n = near_put_over_s   = opt[(opt.strike > opt.spot) & (opt.kind == PUT.upper())].iloc[0]
    near_put_over_s_vol   = find_vol((n.bid + n.ask) / 2, n.spot, n.strike, n.dte / IMPLIED_VOL_DAYS, 0 , PUT)
    vol_near_over         = (near_call_over_s_vol + near_put_over_s_vol) / 2    # sigma1 x2

    strike_above = near_call_over_s.strike ; strike_below = near_call_under_s.strike 
    sigma1  = vol_near_under * ((strike_above - spot) / (strike_above - strike_below))
    sigma1 += vol_near_over * ((spot - strike_below) / (strike_above - strike_below))
    return sigma1 

def get_premium_with_vol(option, atm_vol):
    prems = []
    def do_find_price(opt):
        if opt.kind.lower() == CALL:
            p = BS_call(opt.spot, opt.strike, 0 , atm_vol, opt.dte / IMPLIED_VOL_DAYS)
        else:
            p = BS_put(opt.spot, opt.strike, 0 , atm_vol, opt.dte / IMPLIED_VOL_DAYS)
        prem = (opt.bid + opt.ask) / 2 - p 
        prems.append(prem)

    option.apply(do_find_price, axis=1)
    option['prems'] = prems
    option['atmvol'] = atm_vol 

def get_greeks(option):
    deltas = []; gammas = [] ; vegas = [] ; thetas = [] ; 
    for i,r in option.iterrows():
        vol = (r.BidVol + r.AskVol) / 2
        try: d = delta(r.spot, r.strike, 0, vol, r.dte/252., r.kind.lower())
        except: d = 0 

        try: g = gamma(r.spot, r.strike, 0, vol, r.dte/252.)
        except: g = 0 

        try: v = vega(r.spot, r.strike, 0, vol, r.dte/252.)
        except: v = 0 

        try: t = theta(r.spot, r.strike, 0, vol, r.dte/252., r.kind.lower())
        except: t = 0 

        deltas.append(d)
        gammas.append(g)
        vegas.append(v)
        thetas.append(t)

    option['del'] = deltas
    option['gam'] = gammas
    option['veg'] = vegas
    option['the'] = thetas


def near_atm_opts(opt):
    calls = opt[(opt.kind == 'CALL') & (opt.strike + 3 >= opt.spot)]     
    puts  = opt[(opt.kind == 'PUT') & (opt.strike - 3 <= opt.spot)] 
    return calls, puts

'''
def get_atm_vol(optchain):
    kinds = [x for x in optchain.groupby(mod_struc.kind)]
    assert len(kinds)==2 , 'Optchain has not two kinds'

    # https://stackoverflow.com/questions/61289020/fast-implied-volatility-calculation-in-python
    atm_sum = 0 
    for desc,kind in kinds:
        last_strike = -1
        pre = post = None
        for i,r in kind.iterrows():
            if r.strike >= r.spot:
                post = r
                break
            pre = r 

        rate = .001
        
        pre_bid = find_vol(pre.bid, pre.spot, pre.strike, pre.dte / TRADING_DAYS, rate, desc)
        pre_ask = find_vol(pre.ask, pre.spot, pre.strike, pre.dte / TRADING_DAYS, rate, desc)

        if pre_bid and pre_ask: pre_mid = (pre_bid + pre_ask) / 2.0
        elif pre_bid and not pre_ask: pre_mid = pre_bid
        elif not pre_bid and pre_ask: pre_mid = pre_ask
        else: 
            print("OPT CHAIN FAIL")
            return 0 

        post_bid = find_vol(post.bid, post.spot, post.strike, post.dte / TRADING_DAYS, rate, desc)
        post_ask = find_vol(post.ask, post.spot, post.strike, post.dte / TRADING_DAYS, rate, desc)

        if post_bid and post_ask: post_mid = (post_bid + post_ask) / 2.0
        elif post_bid and not post_ask: post_mid = post_bid
        elif not post_bid and post_ask: post_mid = post_ask
        else: 
            print("OPT CHAIN FAIL")
            return 0 

        k_diff = post.strike - pre.strike
        w1 = 1 - (post.strike - post.spot) / k_diff # post
        w2 = 1 - w1  # pre
        
        atm_sum += post_mid * w1 + pre_mid * w2
 
    return atm_sum / 2.0
'''


class Opts():

    # vol=decimal, time=days, rate=decimal
    def __init__(self, opttype, strike, vol, time, rate, price , limits=20):
        self.opttype = opttype
        assert self.opttype in [LC,SC,LP,SP], 'Incorrect option type'
        self.k = strike
        self.orig_vol = vol 
        self.v = vol
        if (self.v > 1): print("WARNING!!!! vol above 100%!!!!")
        self.t = time / YEAR_DAYS # dte graphically (will change)
        self.dte = time      # original dte (will not change)
        self.r = rate
        self.start = strike - limits
        self.end   = strike + limits
        self.range = np.arange(self.start, self.end) 
        self.price = price 
        self.expired = False

        self.f = 1 
        if self.opttype in [SC,LP]: self.f = -1
        self.func = BS_call if self.opttype in [LC,SC] else BS_put
    
    def set_greeks(self):
        if self.expired:
            self.term_strat = self.strat = self.delta = self.vega  = self.theta = self.gamma = self.vanna = self.charm = self.vomma = [0] * self.range
        else:
            z = -1 * self.f * self.price 
            self.term_strat = [z + self.f * self.func(x, self.k, self.r, self.v, .00001) for x in self.range]
            self.strat = [z + self.f * self.func(x, self.k, self.r, self.v, self.t) for x in self.range]
            self.delta = [self.f * delta(x, self.k, self.r, self.v, self.t, self.opttype) for x in self.range]
            self.vega  = [self.f * .01 * vega(x, self.k, self.r, self.v, self.t) for x in self.range]
            self.vomma = [self.f * .01 * vomma(x, self.k, self.r, self.v, self.t) for x in self.range]
            self.theta = [self.f * theta(x, self.k, self.r, self.v, self.t, self.opttype) for x in self.range]
            self.gamma = [self.f * gamma(x, self.k, self.r, self.v, self.t) for x in self.range]
            self.vanna = [self.f * vanna(x, self.k, self.r, self.v, self.t) for x in self.range]
            self.charm = [self.f * charm(x, self.k, self.r, self.v, self.t) for x in self.range]

    def perturb(self,v=None,t=None):
        if not v and not t: return 

        vnew = v if v else self.v
        tnew = t if t else self.t

        return [self.f * self.func(x, self.k, self.r, vnew, tnew) for x in self.range]

    def set_range(self,start,end):
        self.start = start
        self.end   = end 
        self.range = np.arange(self.start,self.end)

class OptPlot():

    def __init__(self, fig, axs):
        self.fig = fig
        self.axs = axs
        self.strat = []
        self.snapshot_taken = False
        self.snapshot_dic = {}

    def add_option(self,opt):
        assert type(opt)==Opts, 'Opt is wrong type '
        self.strat.append(opt)
        self.regraph()

    def remove_option(self,opt):
        if opt not in self.strat: return 
        self.strat.remove(opt)
        self.snapshot_taken = False
        self.regraph()

    def snapshot(self):
        if self.snapshot_taken:
            return 

        start = min([x.start for x in self.strat])
        end   = max([x.end for x in self.strat])
        S = np.arange(start,end)

        term_options = list(map(sum, zip(*[x.term_strat for x in self.strat])))
        options      = list(map(sum, zip(*[x.strat for x in self.strat])))

        deltas = list(map(sum, zip(*[x.delta for x in self.strat])))
        vegas  = list(map(sum, zip(*[x.vega for x in self.strat])))
        thetas = list(map(sum, zip(*[x.theta for x in self.strat])))

        gammas = list(map(sum, zip(*[x.gamma for x in self.strat])))
        vannas = list(map(sum, zip(*[x.vanna for x in self.strat])))
        charms = list(map(sum, zip(*[x.charm for x in self.strat])))
        vommas = list(map(sum, zip(*[x.vomma for x in self.strat])))

        lines = []
        lines.append({'line':term_options,'name':'terminal','coords':(0,0)})
        lines.append({'line':options,     'name':'strategy','coords':(1,0)})
        lines.append({'line':deltas,      'name':'delta'   ,'coords':(0,1)})
        lines.append({'line':vommas,      'name':'vomma'   ,'coords':(2,1)})
        lines.append({'line':vegas,       'name':'vega'    ,'coords':(2,0)})
        lines.append({'line':thetas,      'name':'theta'   ,'coords':(1,1)})
        lines.append({'line':gammas,      'name':'gamma'   ,'coords':(0,2)})
        lines.append({'line':vannas,      'name':'vanna'   ,'coords':(2,2)})
        lines.append({'line':charms,      'name':'charm'   ,'coords':(1,2)})

        self.snapshot_dic = {
            'start':start,
            'end':end,
            'S':  S,
            'lines': lines
            }

        self.snapshot_taken = True

    def snapshot_draw(self):
        S = self.snapshot_dic['S']
        start = self.snapshot_dic['start']
        end = self.snapshot_dic['end']
        lines = self.snapshot_dic['lines']
        for line in lines:
            coords = line['coords']
            tl = self.axs[coords[0],coords[1]]
            tl.plot(S,line['line'])

    def param_change(self,v_diff,dte_diff, atm_vol_dte):
        if not self.strat: return 
        for st in self.strat: 
            # vol weighting prob wrong...its an estimate anyway
            st.v = st.orig_vol * v_diff 
            assert st.v > 0, 'new vol is negative'

            print(f'{st.orig_vol=}')
            print(f'{st.v=}')
            if (dte_diff > st.dte): 
                st.t = 0 
                st.expired = True 
                print("EXPIRED")
            else:              
                st.expired = False
                st.t = (st.dte - dte_diff) / YEAR_DAYS
        self.regraph() 

    def graph_by_type(self,line,name,coords, S, start, end, spot):
        tl = self.axs[coords[0],coords[1]]
        x_major_axis = np.arange(start,end,10)
        x_minor_axis = np.arange(start,end,1)
        tl.set_xticks(x_major_axis)
        tl.set_xticks(x_minor_axis,minor=True)
        mi,ma=min(line),max(line)
        tic = int((ma-mi)/5)
        tl.plot(S,line, '--')
        tl.set(ylabel='PnL')
        tl.set_title(name)
        tl.grid(which='minor', alpha=0.2)
        tl.grid(which='major', alpha=0.5)

    def regraph(self):
        strat_cnt = len(self.strat)
        for i in range(3):
            for j in range(3):
                self.axs[i,j].clear()
        if not strat_cnt:
            return

        spot = self.strat[0].price 
        start = min([x.start for x in self.strat])
        end   = max([x.end for x in self.strat])
        S = np.arange(start,end)
        for s in self.strat: 
            s.set_range(start,end)
            s.set_greeks()

        options      = list(map(sum, zip(*[x.strat for x in self.strat])))
        term_options = list(map(sum, zip(*[x.term_strat for x in self.strat])))

        deltas = list(map(sum, zip(*[x.delta for x in self.strat])))
        vegas  = list(map(sum, zip(*[x.vega for x in self.strat])))
        thetas = list(map(sum, zip(*[x.theta for x in self.strat])))

        gammas = list(map(sum, zip(*[x.gamma for x in self.strat])))
        vannas = list(map(sum, zip(*[x.vanna for x in self.strat])))
        charms = list(map(sum, zip(*[x.charm for x in self.strat])))
        vommas = list(map(sum, zip(*[x.vomma for x in self.strat])))

        self.graph_by_type(term_options,'terminal',(0,0),S,start,end,spot)
        self.graph_by_type(options,'strategy',(1,0),S,start,end,spot)
        self.graph_by_type(deltas,'delta',(0,1),S,start,end,spot)
        self.graph_by_type(vommas,'vomma',(2,1),S,start,end,spot)
        self.graph_by_type(vegas,'vega',(2,0),S,start,end,spot)
        self.graph_by_type(thetas,'theta',(1,1),S,start,end,spot)
        self.graph_by_type(gammas,'gamma',(0,2),S,start,end,spot)
        self.graph_by_type(vannas,'vanna',(2,2),S,start,end,spot)
        self.graph_by_type(charms,'charm',(1,2),S,start,end,spot)

        if self.snapshot_taken:
            self.snapshot_draw()

        self.fig.canvas.draw_idle()


# fig, axs = plt.subplots(3, 3)
# fig.tight_layout(pad=1.0)
# oplot = OptPlot(fig,axs)
# opt = Opts(LC,100,.25,3,.01)
# oplot.add_option(opt)

# S = np.arange(30, 160)
# 
# k=100
# r=.1
# v=.25
# t=3

# lcall  = [BS_call(x, 80, r, v, t) for x in S]
# scall  = [-BS_call(x, 110, r, v, t) for x in S]
# strat  = [lcall,scall]
# pos    = list(map(sum, zip(*strat)))
# 
# _delta = [delta(x, k, r, v, t,CALL) for x in S]
# # _vega  = [vega(x, k, r, v, t) / 100 for x in S]
# _vega_l  = [vega(x, 80, r, v, t) / 100 for x in S]
# _vega_s  = [-vega(x, 110, r, v, t) / 100 for x in S]
# _vegas  = [_vega_l,_vega_s]
# _vega    = list(map(sum, zip(*_vegas)))
# 
# _theta = [theta(x, k, r, v, t,CALL) for x in S]
# 
# _gamma  = [gamma(x, k, r, v, t) for x in S]
# _vanna  = [vanna(x, k, r, v, t) for x in S]
# _charm  = [charm(x, k, r, v, t) for x in S]
# 
# # fig, ax = plt.subplots(nrows=2, ncols=2, sharex=True, figsize=(40, 30))
# fig, axs = plt.subplots(3, 3)
# fig.tight_layout(pad=1.0)
# # fig.text(.5,.08,'Underlying')
# tl = axs[0,0]
# tl.plot(S,pos)
# tl.set(ylabel='PnL')
# tl.set_title('call spread')
# tl.grid()
# 
# tl = axs[1,0]
# tl.plot(S,_delta)
# tl.set_title('delta')
# tl.grid() 
# 
# tl = axs[1,1]
# tl.plot(S,_vega)
# tl.set_title('vega')
# tl.grid() 
# 
# tl = axs[1,2]
# tl.plot(S,_theta)
# tl.set_title('theta')
# tl.grid() 
# 
# tl = axs[2,0]
# tl.plot(S,_gamma)
# tl.set_title('gamma')
# tl.grid() 
# 
# tl = axs[2,1]
# tl.plot(S,_vanna)
# tl.set_title('vanna')
# tl.grid() 
# 
# tl = axs[2,2]
# tl.plot(S,_charm)
# tl.set_title('charm')
# tl.grid() 
# 
# plt.show() 


#vals_short_call = [-1 * BS_call(x, 30, 0.10, 0.), 2, 0) for x in S]

#vegaprof = [vega(x, 100, 0.1, 0.3, .5 , 0) for x in S]

#strat = [vals_long_call , vals_short_call]
#cumsum = list(map(sum, zip(*strat)))
# vals_put = [BS_put(x, 50, 0.10, 0.2, 2, 0) for x in S]
# plt.plot(S,vals_long_call, '--',label = "Long Call")
# plt.plot(S,vals_short_call, '--', label = "Short Call")
# plt.plot(S,cumsum, label = "Spread")
#plt.plot(S,vegaprof, label = "vega")
#plt.plot(S, vals_put, 'b', label  = "Put")
#plt.legend()
# plt.xlabel("Stock Price ($)")
# plt.ylabel("Option Price ($)")
#plt.show()



