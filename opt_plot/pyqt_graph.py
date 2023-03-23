import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from opt_plot.bs_opt_plot import *

import opt_plot.pyqt_options as pq_options
import opt_plot.pyqt_layout as pq_layout
import opt_plot.pyqt_strategy as pq_strategy

class Canvas(FigureCanvas):
    def __init__(self, parent):

        self.fig, self.axs = plt.subplots(3, 3)
        self.fig.tight_layout(pad=1.0)

        super().__init__(self.fig)

        self.setParent(parent)
        self.oplot = OptPlot(self.fig,self.axs)
        

class GUILayout(QWidget):
    def __init__(self, parent = None):
        super(GUILayout, self).__init__(parent)

        pq_layout.setup_layout(self)
        pq_strategy.setup_strategy(self)
        pq_options.setup_options(self)

        pq_layout.create_sliders()
        pq_layout.create_opt_layout()
        pq_layout.create_opt_list()

        self.canvas = Canvas(self)
        pq_layout.master_layout_create()

        self.resize(1600,800)

        self.opt_dic = {}

        self.td = None
        self.td_cache = [] # TODO 
        self.opt_tenor_dict = {}
        self.opt_tenor_descs = []

        self.opt_choice_tenor = None
    
        self.last = None
        self.single_strat_showing = None
        self.single_option_tenor_choices = None

        self.cur_dte_slider = 0 
        self.cur_vol_slider = 0 
        self.cur_atm_vol    = 0 

    def refresh_plot(self):
        self.canvas.oplot.fig.canvas.draw_idle()


class Window(QMainWindow):
    def __init__(self):
        super().__init__()

        self.resize(1600, 800)

        self.gui_lay = GUILayout(self)

        self.central_widg = QWidget()
        self.setCentralWidget(self.central_widg)
        self.centralWidget().setLayout(self.gui_lay.outer)

        self.show()             

