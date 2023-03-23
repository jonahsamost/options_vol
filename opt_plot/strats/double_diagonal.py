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

# calendar strangle/straddle swap
class double_diagonal:

    def __init__(self,g):
        self.gui = g
        name='double diagonal'
        self.place = 'Option'
        self.opt_list = None

        self.gui.strategies[name] = self._callback
        self.gui.strat_choices.addItems([name])

    def _add(self):
        ls = pq_layout.get_long_short_widget(self.short_widg, self.long_widg)
        if not ls: return 
        if ls == bs.LONG: legs = [bs.LONG,bs.SHORT,bs.SHORT,bs.LONG]
        else: legs = [bs.SHORT,bs.LONG,bs.LONG,bs.SHORT]
        cnt = 0 

        l1 = self.leg1.currentText() 
        l2 = self.leg2.currentText() 
        l3 = self.leg3.currentText() 
        l4 = self.leg4.currentText() 
        if l1==self.place or l2==self.place or l3==self.place or l4==self.place: return 

        l1opt = pq_opts.get_single_opt_from_name(l1,self.further)
        pq_opts.add_option(l1opt, l1, legs[cnt]==bs.LONG)
        cnt += 1
        
        l2opt = pq_opts.get_single_opt_from_name(l2,self.closer)
        pq_opts.add_option(l2opt, l2, legs[cnt]==bs.LONG)
        cnt += 1

        l3opt = pq_opts.get_single_opt_from_name(l3,self.closer)
        pq_opts.add_option(l3opt, l3, legs[cnt]==bs.LONG)
        cnt += 1

        l4opt = pq_opts.get_single_opt_from_name(l4,self.further)
        pq_opts.add_option(l4opt, l4, legs[cnt]==bs.LONG)
        cnt += 1
        pq_opts.snapshot()

    def _date_chosen1(self,desc):
        self.closer = pq_opts.get_opt_from_opt_chain(desc)
        if self.closer is None: return 
        names = pq_opts.get_single_names_from_opt(self.closer)
        calls = [x for x in names if x[0].lower() == 'c']
        puts =  [x for x in names if x[0].lower() == 'p']

        self.leg2.clear() 
        self.leg3.clear() 
        self.leg2.addItems([self.place] + puts) 
        self.leg3.addItems([self.place] + calls) 

    def _date_chosen2(self,desc):
        self.further = pq_opts.get_opt_from_opt_chain(desc)
        if self.further is None: return 
        names = pq_opts.get_single_names_from_opt(self.further)
        calls = [x for x in names if x[0].lower() == 'c']
        puts = [x for x in names if x[0].lower() == 'p']

        self.leg1.clear() 
        self.leg4.clear() 
        self.leg1.addItems([self.place] + puts) 
        self.leg4.addItems([self.place] + calls) 

    def _callback(self):

        outer = QHBoxLayout()

        longer = ['Long DTE']
        shorter = ['Short DTE']
        for opt in self.gui.opts:
            longer.append(pq_opts.get_opt_tenor_name(opt))
            shorter.append(pq_opts.get_opt_tenor_name(opt))

        opt_chain1 = QComboBox()
        opt_chain1.addItems(shorter)
        opt_chain1.activated[str].connect(self._date_chosen1)

        opt_chain2 = QComboBox()
        opt_chain2.addItems(longer)
        opt_chain2.activated[str].connect(self._date_chosen2)

        self.short_long, self.long_widg, self.short_widg = pq_layout.make_long_short_widget()

        self.leg1 = QComboBox()
        self.leg1.addItems([self.place])
        self.leg4 = QComboBox()
        self.leg4.addItems([self.place])
        long_dte = QFormLayout()
        long_dte.addRow('put', self.leg1)
        long_dte.addRow('call', self.leg4)

        self.leg2 = QComboBox()
        self.leg2.addItems([self.place])
        self.leg3 = QComboBox()
        self.leg3.addItems([self.place])
        short_dte = QFormLayout()
        short_dte.addRow('put', self.leg2)
        short_dte.addRow('call', self.leg3)

        self.short_dte_box = QHBoxLayout() 
        self.short_dte_box.addWidget(opt_chain1)
        self.short_dte_box.addLayout(short_dte)

        self.far_dte_box = QHBoxLayout() 
        self.far_dte_box.addWidget(opt_chain2)
        self.far_dte_box.addLayout(long_dte)

        go = QPushButton('Add!')
        go.clicked.connect(self._add)

        strat = QFormLayout()
        strat.addRow('', self.short_dte_box)
        strat.addRow('', self.far_dte_box)
        strat.addRow('', go)

        outer.addLayout(self.short_long)
        outer.addLayout(strat)

        self.gui.live_opt_box.addLayout(outer)

