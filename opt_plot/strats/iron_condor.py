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


class iron_condor:

    def __init__(self,g):
        self.gui = g
        name='iron condor / iron butterfly'
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

        l1opt = pq_opts.get_single_opt_from_name(l1,self.opt_list)
        pq_opts.add_option(l1opt, l1, legs[cnt]==bs.LONG)
        cnt += 1
        
        l2opt = pq_opts.get_single_opt_from_name(l2,self.opt_list)
        pq_opts.add_option(l2opt, l2, legs[cnt]==bs.LONG)
        cnt += 1

        l3opt = pq_opts.get_single_opt_from_name(l3,self.opt_list)
        pq_opts.add_option(l3opt, l3, legs[cnt]==bs.LONG)
        cnt += 1

        l4opt = pq_opts.get_single_opt_from_name(l4,self.opt_list)
        pq_opts.add_option(l4opt, l4, legs[cnt]==bs.LONG)
        pq_opts.snapshot()

    def _date_chosen(self,desc):
        self.opt_list = pq_opts.get_opt_from_opt_chain(desc)
        if self.opt_list is None: return 

        names = pq_opts.get_single_names_from_opt(self.opt_list)
        calls = [x for x in names if x[0].lower() == 'c']
        puts  = [x for x in names if x[0].lower() == 'p']
        self.leg1.clear() 
        self.leg2.clear() 
        self.leg3.clear() 
        self.leg4.clear() 
        self.leg1.addItems([self.place] + puts) 
        self.leg2.addItems([self.place] + puts) 
        self.leg3.addItems([self.place] + calls) 
        self.leg4.addItems([self.place] + calls) 

    def _callback(self):

        outer = QHBoxLayout()

        names = ['Dates']
        for opt in self.gui.opts:
            names.append(pq_opts.get_opt_tenor_name(opt))

        opt_chain = QComboBox()
        opt_chain.addItems(names)
        # opt_chain.view().pressed.connect(_date_chosen)
        opt_chain.activated[str].connect(self._date_chosen)

        self.short_long, self.long_widg, self.short_widg = pq_layout.make_long_short_widget()

        self.leg1 = QComboBox()
        self.leg1.addItems([self.place])
        self.leg2 = QComboBox()
        self.leg2.addItems([self.place])
        self.leg3 = QComboBox()
        self.leg3.addItems([self.place])
        self.leg4 = QComboBox()
        self.leg4.addItems([self.place])

        go = QPushButton('Add!')
        go.clicked.connect(self._add)

        strat = QFormLayout()
        strat.addRow('opt', self.leg1)
        strat.addRow('opt', self.leg2)
        strat.addRow('opt', self.leg3)
        strat.addRow('opt', self.leg4)
        strat.addRow('', go)

        outer.addWidget(opt_chain)
        outer.addLayout(self.short_long)
        outer.addLayout(strat)

        self.gui.live_opt_box.addLayout(outer)

