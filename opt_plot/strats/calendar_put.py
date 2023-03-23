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


class calendar_put:

    def __init__(self,g):
        self.gui = g
        name='calendar put spread'
        self.place = 'Option'
        self.opt_list = None
        self.note = '(1) Sell put at strike A and (2) buy a put at strike A further DTE'

        self.gui.strategies[name] = self._callback
        self.gui.strat_choices.addItems([name])

    def _add(self):
        ls = pq_layout.get_long_short_widget(self.short_widg, self.long_widg)
        if not ls: return 
        if ls == bs.LONG: legs = [bs.SHORT,bs.LONG]
        else: legs = [bs.LONG,bs.SHORT]
        cnt = 0 

        l1 = self.leg1.currentText() 
        l2 = self.leg2.currentText() 
        if l1==self.place or l2==self.place: return 

        l1opt = pq_opts.get_single_opt_from_name(l1,self.l1_opt_list)
        pq_opts.add_option(l1opt, l1, legs[cnt]==bs.LONG)
        cnt += 1
        
        l2opt = pq_opts.get_single_opt_from_name(l2,self.l2_opt_list)
        pq_opts.add_option(l2opt, l2, legs[cnt]==bs.LONG)
        pq_opts.snapshot()

    def _date_chosen1(self,desc):
        self.l1_opt_list = pq_opts.get_opt_from_opt_chain(desc)
        if self.l1_opt_list is None: return 

        names = pq_opts.get_single_names_from_opt(self.l1_opt_list)
        names = [x for x in names if x[0].lower() == 'p']
        self.leg1.clear() 
        self.leg1.addItems([self.place] + names) 

    def _date_chosen2(self,desc):
        self.l2_opt_list = pq_opts.get_opt_from_opt_chain(desc)
        if self.l2_opt_list is None: return 

        names = pq_opts.get_single_names_from_opt(self.l2_opt_list)
        names = [x for x in names if x[0].lower() == 'p']
        self.leg2.clear() 
        self.leg2.addItems([self.place] + names) 

    def _callback(self):

        outer = QHBoxLayout()

        names = []
        for opt in self.gui.opts:
            names.append(pq_opts.get_opt_tenor_name(opt))

        opt_chain1 = QComboBox()
        opt_chain1.addItems(['Short DTE'] + names)
        opt_chain1.activated[str].connect(self._date_chosen1)

        opt_chain2 = QComboBox()
        opt_chain2.addItems(['Long DTE'] + names)
        opt_chain2.activated[str].connect(self._date_chosen2)

        self.short_long, self.long_widg, self.short_widg = pq_layout.make_long_short_widget()

        self.leg1 = QComboBox()
        self.leg1.addItems([self.place])
        self.leg2 = QComboBox()
        self.leg2.addItems([self.place])

        self.leg1_box = QHBoxLayout() 
        self.leg1_box.addWidget(opt_chain1)
        self.leg1_box.addWidget(self.leg1)

        self.leg2_box = QHBoxLayout() 
        self.leg2_box.addWidget(opt_chain2)
        self.leg2_box.addWidget(self.leg2)

        go = QPushButton('Add!')
        go.clicked.connect(self._add)

        notebox = QTextEdit(self.note)
        notebox.setReadOnly(True)
        notebox.setMaximumHeight(30)

        strat = QFormLayout()
        strat.addRow('',notebox)
        strat.addRow('opt', self.leg1_box)
        strat.addRow('opt', self.leg2_box)
        strat.addRow('', go)

        outer.addLayout(self.short_long)
        outer.addLayout(strat)

        self.gui.live_opt_box.addLayout(outer)

