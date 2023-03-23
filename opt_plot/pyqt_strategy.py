import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import model.structures as mod_struc

import opt_plot.bs_opt_plot as bs
import opt_plot.pyqt_graph as pq_graph
import opt_plot.pyqt_layout as pq_layout
import opt_plot.pyqt_options as pq_opts
import data_feed.tdameritrade as tda

import opt_plot.strats.put_butterfly as put_butterfly
import opt_plot.strats.call_butterfly as call_butterfly
import opt_plot.strats.put as put
import opt_plot.strats.call as call
import opt_plot.strats.put_spread as put_spread
import opt_plot.strats.call_spread as call_spread
import opt_plot.strats.collar as collar
import opt_plot.strats.straddle as straddle
import opt_plot.strats.iron_condor as iron_condor
import opt_plot.strats.calendar_call as calendar_call
import opt_plot.strats.calendar_put as calendar_put
import opt_plot.strats.call_ladder as call_ladder
import opt_plot.strats.put_ladder as put_ladder
import opt_plot.strats.call_ratio_spread as call_ratio_spread
import opt_plot.strats.put_ratio_spread as put_ratio_spread
import opt_plot.strats.call_condor as call_condor
import opt_plot.strats.put_condor as put_condor
import opt_plot.strats.double_diagonal as double_diagonal
import opt_plot.strats.strap as strap
import opt_plot.strats.strip as strip
import opt_plot.strats.risk_reversal as risk_reversal

gui = None

def setup_strategy(g):
    global gui
    gui = g

### STRATEGIES ###

def init_strategies():
    put.put(gui)
    put_spread.put_spread(gui)
    calendar_put.calendar_put(gui)
    call.call(gui)
    call_spread.call_spread(gui)
    calendar_call.calendar_call(gui)
    collar.collar(gui)
    straddle.straddle(gui)
    iron_condor.iron_condor(gui)
    put_butterfly.put_butterfly(gui)
    call_butterfly.call_butterfly(gui)
    call_ladder.call_ladder(gui)
    put_ladder.put_ladder(gui)
    call_ratio_spread.call_ratio_spread(gui)
    put_ratio_spread.put_ratio_spread(gui)
    call_condor.call_condor(gui)
    put_condor.put_condor(gui)
    double_diagonal.double_diagonal(gui)
    strap.strap(gui)
    strip.strip(gui)
    risk_reversal.risk_reversal(gui)

def strategy_chosen(key): 
    # get symbol and options chains
    sym = gui.sym_edit.text()
    if not sym: return 
    gui.td = tda.TD([sym.upper()])
    gui.td.run()
    gui.opts = gui.td.chains
    if not gui.opts:
        print(f"No option chain for {sym.upper()}")
        return 

    for opt in gui.opts:
        opt[mod_struc.atm_vol] = bs.get_option_atm_vol(opt)

    gui.cur_atm_vol          = gui.opts[0].iloc[0].atm_vol
    gui.cur_atm_vol_dte      = gui.opts[0].iloc[0].dte

    # closest expiration's atm vol ... changing vol in slider requires vega weighting
    atmvol = gui.opts[0].iloc[0].atm_vol * 100 * pq_layout.vol_step  
    gui.vol_slider.setValue(int(atmvol)) 

    spot = gui.opts[0].iloc[0].spot
    pq_layout.spot_slider_set_min_max(int(spot))
    gui.spot_slider.setValue(spot * pq_layout.spot_step)
    gui.cur_spot = spot

    pq_layout.delete_live_opt_box_entries()

    print(f"Sym chosen: {sym} with strategy: {key}")
    if key not in gui.strategies: return 
    gui.strategies[key]()

