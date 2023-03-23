import sys, math 
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

from opt_plot.bs_opt_plot import *
import opt_plot.pyqt_layout as pq_layout
import opt_plot.pyqt_graph as pq_graph
import opt_plot.pyqt_strategy as pq_strat

import data_feed.tdameritrade as tda

gui = None

def setup_options(g):
    global gui
    gui = g

### single_option_tenor_choices ###
# opt choices holds a list of actual options on a certain date

def get_opt_tenor_name(opt):
    desc = opt.iloc[0].desc.split()
    name = ' '.join(desc[1:4]) 
    if 'week' in opt.iloc[0].desc.lower():
        name += ' (Weekly)'
    return name

def get_name_from_opt_row(r):
    return f"{r.kind[0]} d:{round(r.delta,2)} k:{round(r.strike,2)} S:{round(r.spot,2)}"

def get_single_names_from_opt(opt):
    if opt is None: return 
    idx = opt.index[0]
    names = []
    for i,r in opt.iterrows():
        names.append(get_name_from_opt_row(r))
    return names

def get_single_opt_from_name(name, opt):
    for i,r in opt.iterrows():
        if name == get_name_from_opt_row(r):
            return r
    return None


def get_opt_from_opt_chain(desc):
    for i,o in enumerate(gui.opts):
        if desc == get_opt_tenor_name(o):
            return o
    return None


def add_option(opt, opt_name, is_long):
    k = opt.strike ; t = opt.dte ; r = 0  ; d = opt.delta ; 

    if opt_name.lower()[0] == 'c':
        otype = LC if is_long else SC
        price = opt.ask if is_long else opt.bid
        try: v = find_vol(price, opt.spot, k, t / 365. , 0 , CALL)
        except Exception as e:
            print(f'Exception {e}: using given vol')
            v = opt.vol
    elif opt_name.lower()[0] == 'p':
        otype = LP if is_long else SP
        price = opt.ask if is_long else opt.bid
        try: v = find_vol(price, opt.spot, k, t / 365. , 0 , PUT )
        except Exception as e:
            print(f'Exception {e}: using given vol')
            v = opt.vol

    gui.last = opt

    print(f'[+] {otype} {price=} vol={v} spot={opt.spot} dte={t}')

    _ = ' '.join(opt.desc.split()[1:4])
    exp = datetime.strftime(datetime.strptime(_,'%b %d %Y').date(), '%m/%d/%y')
    tag = f'{otype_shortcut(otype)} {round(d,2)}d {k}k {exp}'

    if (int(t) > gui.dte_slider.value()):
        gui.dte_slider.setValue(int(t))
        gui.max_dte_slider = int(t)

    opt = Opts(otype,k, v , t , r, price)
    gui.canvas.oplot.add_option(opt)
    gui.refresh_plot()

    add_opt_to_list(tag,opt)

def snapshot():
    gui.canvas.oplot.snapshot()

def add_opt_to_list(tag,opt):
    gui.opt_cnt += 1
    lbl = QLabel(tag) 
    # toggle = QPushButton('Toggle')
    remove      = QPushButton('Delete')
    remove.clicked.connect(del_option_callback)
    # toggle.clicked.connect(toggle_option_callback)
    
    hbox = QHBoxLayout()
    hbox.addWidget(lbl)
    # hbox.addWidget(toggle)
    hbox.addWidget(remove)

    if gui.opt_list.count() == 0:
        remove_all = QPushButton('Clear Options')
        remove_all.clicked.connect(del_all_options_callback)
        gui.opt_list.addWidget(remove_all)

    gui.opt_list.insertLayout(0,hbox)

    gui.opt_dic[hbox] = opt

def del_all_options_callback():
    while gui.opt_list.count():
        item = gui.opt_list.itemAt(0)
        print(item)
        if type(item.widget())==QPushButton:
            gui.opt_list.removeItem(item)
            item.widget().setParent(None)
        else:
            pq_layout.clearLayout(item)
            gui.opt_list.removeItem(item)
            item.setParent(None)
            gui.canvas.oplot.remove_option(gui.opt_dic[item])
            gui.refresh_plot()

def toggle_option_callback():
    print("NEED TO IMPLEMENT TOGGLE")

def del_option_callback():
    for i in range(gui.opt_list.count()):
        hbox = gui.opt_list.itemAt(i)
        for j in range(hbox.count()):
            if hbox.itemAt(j).widget() == gui.sender():
                for k in range(hbox.count()-1 , -1, -1):
                    cur_item = hbox.itemAt(k)
                    hbox.removeItem(cur_item)
                    if cur_item.widget(): 
                        cur_item.widget().setParent(None)

                gui.opt_list.removeItem(hbox)
                hbox.setParent(None)

                gui.canvas.oplot.remove_option(gui.opt_dic[hbox])
                gui.refresh_plot()
        
                del gui.opt_dic[hbox]

                return 


def change_vol_time():
    dte = gui.dte_slider.value() 
    try:
        if dte > gui.max_dte_slider:
            print("can only decrease dte, not increase")
            return 
    except: # means no options added yet
        return 

    vol = pq_layout.vol_slider_value() / pq_layout.vol_max 
    vol_diff = vol / gui.cur_atm_vol
    if vol_diff > .98 and vol_diff < 1.02:
        vol_diff = 1 
    dte_diff = gui.max_dte_slider - dte # only positive

    print(f'{vol=} {gui.cur_atm_vol=} {dte_diff=} {vol_diff=}')

    gui.canvas.oplot.param_change(vol_diff , dte_diff, gui.cur_atm_vol_dte )

