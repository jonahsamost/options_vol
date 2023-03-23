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
import opt_plot.pyqt_layout as pq_layout
import opt_plot.pyqt_options as pq_opts
import opt_plot.pyqt_strategy as pq_strat


class straddle:
    def __init__(self,g):
        self.gui = g
        name='straddle/strangle'
        self.place = 'Option'
        self.opt_list = None

        self.gui.strategies[name] = self._callback
        self.gui.strat_choices.addItems([name])


    def _add(self):
        ls = pq_layout.get_long_short_widget(self.short_widg, self.long_widg)
        if not ls: return 
        if ls == bs.LONG: legs = [bs.LONG, bs.LONG]
        else: legs = [bs.SHORT, bs.SHORT]
        cnt = 0 

        l1 = self.leg1.currentText()
        l2 = self.leg2.currentText()
        if l1==self.place or l2==self.place: return 

        l1opt = pq_opts.get_single_opt_from_name(l1,self.opt_list)
        pq_opts.add_option(l1opt, l1, legs[cnt]==bs.LONG)
        cnt += 1

        l2opt = pq_opts.get_single_opt_from_name(l2,self.opt_list)
        pq_opts.add_option(l2opt, l2, legs[cnt]==bs.LONG)
        pq_opts.snapshot()


    def _date_chosen(self,desc):
        self.opt_list = pq_opts.get_opt_from_opt_chain(desc)
        if self.opt_list is None: return 
        names = pq_opts.get_single_names_from_opt(self.opt_list)
        puts  = [x for x in names if x[0].lower() == 'p']
        calls = [x for x in names if x[0].lower() == 'c']
        self.leg1.clear() 
        self.leg2.clear() 
        self.leg1.addItems([self.place] + puts)
        self.leg2.addItems([self.place] + calls)

    def _callback(self):
        outer = QHBoxLayout()

        names = ['Dates']
        for opt in self.gui.opts:
            names.append(pq_opts.get_opt_tenor_name(opt))

        opt_chain = QComboBox()
        opt_chain.addItems(names)
        opt_chain.activated[str].connect(self._date_chosen)

        self.short_long, self.long_widg, self.short_widg = pq_layout.make_long_short_widget()

        self.leg1 = QComboBox()
        self.leg1.addItems([self.place])
        self.leg2 = QComboBox()
        self.leg2.addItems([self.place])

        go = QPushButton('Add!')
        go.clicked.connect(self._add)

        strat = QFormLayout()
        strat.addRow('opt1', self.leg1)
        strat.addRow('opt2', self.leg2)
        strat.addRow('', go)
        
        outer.addWidget(opt_chain)
        outer.addLayout(self.short_long)
        outer.addLayout(strat)

        self.gui.live_opt_box.addLayout(outer)

