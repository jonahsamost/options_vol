import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import opt_plot.bs_opt_plot as bs
import opt_plot.pyqt_graph as pq_graph
import opt_plot.pyqt_options as pq_options
import opt_plot.pyqt_strategy as pq_strat

import data_feed.tdameritrade as tda

vol_step = 10.
vol_max  = 1000
spot_step = 100
gui = None

def setup_layout(g):
    global gui
    gui = g

#### LAYOUT #####

def create_sliders():
    ### spot slider below ###

    def spot_slider_create():
        slider = QSlider()
        slider.setOrientation(Qt.Horizontal)
        slider.setTickPosition(slider.TicksBothSides)

        slider.setTickInterval(5 * spot_step)
        slider.setSingleStep(1 * spot_step)
        slider.setMinimum(0)
        slider.setMaximum(100)

        slider.valueChanged.connect(spot_slider_val_change)
        return slider 

    def spot_label_create():
        label = QLabel()
        label.setText(f"spot: 0")
        return label

    def spot_slider_val_change():
        s = gui.spot_slider.value() / spot_step
        gui.spot_label.setText(f"spot: {float(s)}")

    ### vol slider below ###
    def vol_slider_create():
        slider = QSlider()
        slider.setOrientation(Qt.Horizontal)
        slider.setTickPosition(slider.TicksBothSides)

        slider.setTickInterval(50)
        slider.setSingleStep(vol_step)
        slider.setMinimum(0)
        slider.setMaximum(vol_max)

        slider.valueChanged.connect(vol_slider_val_change)
        return slider 

    def vol_label_create():
        label = QLabel()
        label.setText(f"ATM vol: 0")
        return label

    def vol_slider_val_change():
        '''
        vertically, when atm vol changes, change other vols in relation to atm vol change
            i.e. if atmvol 20-25 (25% inc), then wing 30vol goes to 30 * 1.25 = 37.5
        horizontally, do some shady shit with vega weighting to determine how much 
            atm vol for next month will change then do the same percentage change shit
        '''

        sz = vol_slider_value() / vol_step 
        gui.vol_label.setText(f"ATM vol: {sz}")

    ### DTE slider below ### 

    def dte_slider_create():
        slider = QSlider()
        slider.setOrientation(Qt.Horizontal)
        slider.setTickPosition(slider.TicksBothSides)

        slider.setTickInterval(5)
        slider.setSingleStep(1)
        slider.setMinimum(0)
        slider.setMaximum(70)

        # TODO 
        '''
        check if sticky strike/delta 
        if sticky strike do nothing
        if sticky delta ... 
        '''

        slider.valueChanged.connect(dte_slider_val_change)
        return slider 

    def dte_label_create():
        label = QLabel()
        label.setText(f"dte: 0")
        return label
    
    def dte_slider_val_change():
        sz = gui.dte_slider.value()
        gui.dte_label.setText(f"dte: {float(sz)}")

    gui.dte_slider = dte_slider_create()
    gui.dte_slider.setFixedWidth(300)
    gui.dte_label  = dte_label_create()

    gui.vol_slider = vol_slider_create()
    gui.vol_slider.setFixedWidth(300)
    gui.vol_label  = vol_label_create()

    gui.spot_slider = spot_slider_create()
    gui.spot_slider.setFixedWidth(300)
    gui.spot_label  = spot_label_create()

    gui.change_time_vol = QPushButton('Change vol/time')
    gui.change_time_vol.clicked.connect(pq_options.change_vol_time)

    gui.grid_dte_vol = QGridLayout()
    i = 0 
    gui.grid_dte_vol.addWidget(gui.dte_label,i,0,alignment=Qt.AlignLeft) ; i+= 1
    gui.grid_dte_vol.addWidget(gui.dte_slider,i,0,alignment=Qt.AlignLeft) ; i+= 1
    gui.grid_dte_vol.addWidget(gui.vol_label,i,0,alignment=Qt.AlignLeft); i+= 1
    gui.grid_dte_vol.addWidget(gui.vol_slider,i,0,alignment=Qt.AlignLeft); i+= 1
    # gui.grid_dte_vol.addWidget(gui.spot_label,i,0,alignment=Qt.AlignLeft); i+= 1
    # gui.grid_dte_vol.addWidget(gui.spot_slider,i,0,alignment=Qt.AlignLeft); i+= 1
    gui.grid_dte_vol.addWidget(gui.change_time_vol,i,0,alignment=Qt.AlignCenter); i+= 1


def create_opt_layout():
    gui.live_opts = QFormLayout()
    gui.sym_edit = QLineEdit()
    gui.sym_edit.setFixedWidth(70)

    # opt_chain keeps list of option tenors 
    gui.opt_chain = QComboBox()
    gui.opt_chain.addItems([])

    gui.strat_choices = QComboBox()
    gui.strat_choices.addItems(['Strategies'])
    # different ways used can choose
    # gui.strat_choices.view().pressed.connect(strategy_chosen)
    gui.strat_choices.activated[str].connect(pq_strat.strategy_chosen)

    # add strategies 
    gui.strategies = {}
    gui.strat_layouts = {}
    pq_strat.init_strategies()

    gui.live_opt_box = QHBoxLayout()
    gui.live_opt_box.addWidget(QLabel('Symbol'))
    gui.live_opt_box.addWidget(gui.sym_edit)
    # gui.live_opt_box.addWidget(gui.find_option)
    gui.live_opt_box.addWidget(gui.strat_choices)

    gui.live_opt_box_widg_cnt = gui.live_opt_box.count()

    gui.live_opts.addRow('', gui.live_opt_box)


def create_opt_list():
    gui.opt_list = QVBoxLayout() 
    gui.opt_cnt = 0

def master_layout_create():
    gui.outer = QGridLayout()
    gui.outer.addLayout(gui.live_opts,0,0,alignment=Qt.AlignLeft)
    gui.outer.addLayout(gui.opt_list,0,2,alignment=Qt.AlignLeft)
    gui.outer.addLayout(gui.grid_dte_vol,0,3,alignment=Qt.AlignLeft)
    gui.outer.addWidget(gui.canvas,1,0,4,4)

### helper methods below ###

def clearLayout(layout):
    if layout is not None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
            else:
                clearLayout(item.layout())

# used for dynamically updating strategy/single option choices
def delete_live_opt_box_entries():
    while gui.live_opt_box.count() > gui.live_opt_box_widg_cnt:
        item = gui.live_opt_box.takeAt(gui.live_opt_box_widg_cnt)
        widget = item.widget()
        if widget is not None:
            widget.deleteLater()
        else:
            clearLayout(item.layout())

# get if short or long is checked
def get_long_short_widget(s_widg, l_widg):
    if s_widg.isChecked() and l_widg.isChecked():
        print("only short or long")
        return False
    if not s_widg.isChecked() and not l_widg.isChecked():
        print("at least one short or long")
        return False

    return bs.SHORT if s_widg.isChecked() else bs.LONG


def make_long_short_widget():
    short_long = QVBoxLayout()
    short_widg = QCheckBox('short')
    long_widg  = QCheckBox('long')
    short_long.addWidget(short_widg)
    short_long.addWidget(long_widg)
    return (short_long, long_widg, short_widg)

def vol_slider_value():
    return gui.vol_slider.value() 

def spot_slider_set_min_max(spot):
    gui.spot_slider.setMinimum(max(0,spot_step*(spot-50)))
    gui.spot_slider.setMaximum(spot_step*(spot + 50))

