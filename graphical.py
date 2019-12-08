import sys
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

import time
import numpy as np
import math
from pathlib import Path

import neuron as nn
import plots

class DoubleSlider(QSlider):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.decimals = 5
        self._max_int = 10 ** self.decimals

        super().setMinimum(0)
        super().setMaximum(self._max_int)

        self._min_value = 0.0
        self._max_value = 1.0

    @property
    def _value_range(self):
        return self._max_value - self._min_value

    def value(self):
        return float(super().value()) / self._max_int * self._value_range + self._min_value

    def setValue(self, value):
        super().setValue(int((value - self._min_value) / self._value_range * self._max_int))

    def setMinimum(self, value):
        if value > self._max_value:
            raise ValueError("Minimum limit cannot be higher than maximum")

        self._min_value = value
        self.setValue(self.value())

    def setMaximum(self, value):
        if value < self._min_value:
            raise ValueError("Minimum limit cannot be higher than maximum")

        self._max_value = value
        self.setValue(self.value())

    def minimum(self):
        return self._min_value

    def maximum(self):
        return self._max_value

    def setRange(self,min,max):
        self.setMaximum(max)
        self.setMinimum(min)

name_dict = {
    "size":         "Size of each",        
    "K":            "Connection Value K",
    "timer":        "Number of runs",
    "jE":           "Weight for Inhibitory to Excitatory Neurons J_IE",
    "jI":           "Weight for Inhibitory to Inhibitory Neurons J_II",
    "tau":          "Average time to ",
    "threshE":      "Excitatory threshold",
    "threshI":      "Inhibitory threshold",
    "meanStartActi":"meanActivation at time 0",
    "extE":         "Relative External Input to Excitatory",
    "extI":         "Relative External Input to Inhibitory",
    "meanExt":      "External neuron mean activation", 
}

def make_drw():
    pIndiExt    = 1
    nDistri     = 1
    newMeanOT   = 1
    nInter      = 1
    nInter_log  = 1
    dots2       = 1
    return locals()

class ValueUnit(QHBoxLayout):
    @pyqtSlot(int)
    def sliChange(self,val):
        # val = 10**val
        self.val = val
        self.numbox.setValue(val)

    @pyqtSlot(int)
    def numChange(self,val):
        # val = math.log10(val)
        self.val = val
        self.slider.setValue(val)
        
    def __init__(self,name = "Eat Dicks", min_range = 10, max_range = 10000, default = 100, *args, **kwargs):
        super(QHBoxLayout, self).__init__(*args, **kwargs)
        

        self.label = QLabel(name_dict[name]+ ":\t")
        self.label.setAlignment(Qt.AlignCenter)
        self.val = default
        self.name = name
        # self.slider.setRange(math.log10(min_range), math.log10(max_range))
        if max_range < 10:
            self.numbox = QDoubleSpinBox()
            self.slider = DoubleSlider(Qt.Horizontal)
        else:
            self.numbox = QSpinBox()
            self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(min_range,max_range)
        self.numbox.setRange(self.slider.minimum(), self.slider.maximum())
        # self.slider.valueChanged.connect(self.numbox.setValue)
        # self.numbox.valueChanged.connect(self.slider.setValue)
        self.slider.valueChanged.connect(lambda x: self.sliChange(x))
        self.numbox.valueChanged.connect(lambda x: self.numChange(x))
        self.numbox.setValue(default)

        self.addWidget(self.label)
        self.addWidget(self.numbox)
        self.addWidget(self.slider)
    
class MiniUnit(QHBoxLayout):
    @pyqtSlot(int)
    def numChange(self,val):
        self.val = val
    def __init__(self,name = "Eat Dicks", min_range = 10, max_range = 10000, default = 100, *args, **kwargs):
        super(QHBoxLayout, self).__init__(*args, **kwargs)
        

        self.name= name
        lbl_txt = name_dict[name]
        self.val = default
        self.label = QLabel(lbl_txt+ ":\t")
        self.label.setAlignment(Qt.AlignCenter)
        self.numbox = QDoubleSpinBox()
        self.numbox.setRange(min_range, max_range)
        self.numbox.valueChanged.connect(lambda x: self.numChange(x))
        self.numbox.setValue(default)

        self.addWidget(self.label)
        self.addWidget(self.numbox)

class InputLayout(QVBoxLayout):

    range_dict = {
        #min, max, default
        "size":             [10,100000, 1000],
        "K":                [10, 10000, 100],
        "timer":            [1, 1000, 100],
        "jE":               [0, 5, 2],
        "jI":               [0, 5, 1.8],
        "tau":              [0, 5, 0.9],
        "threshE":          [0, 5, 1],
        "threshI":          [0, 5, .7],
        "meanStartActi":    [0, 1, .1],
        "extE":             [0, 1, 1],
        "extI":             [0, 1, .8],
        "meanExt":          [0, 1, .1],
    }
    def __init__(self, *args, **kwargs):
        super(QVBoxLayout, self).__init__(*args, **kwargs)

        for key,val in self.range_dict.items():
            if val[1]>10:   layout  = ValueUnit(key,*val)
            else:           layout  = MiniUnit (key,*val)
            layout.setObjectName("inp_%s" %key)
            self.addLayout(layout)
        exec_button = QPushButton("Run")
        exec_button.clicked.connect(self.run)
        self.addWidget(exec_button, alignment = Qt.AlignRight)
    
    def run(self):
        items = (self.itemAt(i) for i in range(self.count()) 
                if isinstance(self.itemAt(i),ValueUnit)
                or isinstance(self.itemAt(i),MiniUnit)) 
        timestr = time.strftime("%y%m%d_%H%M")
        figfolder = "../figs/testreihe_" + timestr
        valuefoldername = "../ValueVault/testreihe_"
        valueFolder     =  Path(valuefoldername + timestr)
        recNum = 1
        info = {}
        for wrd in ("timestr", "figfolder","valuefoldername","valueFolder","recNum"):
            info[wrd] = locals()[wrd]
        for w in items:
            info[w.name] = w.val
        info['extM'] = np.array([info.pop('extE'),info.pop('extI')])
        info['threshM'] = np.array([info.pop('threshE'),info.pop('threshI')])
        info['sizeM'] = np.array([info['size'],info.pop('size')])

        drw = make_drw()
    
        doThresh    = "constant" #"constant", "gauss", "bound"
        doRand      = 0     #Only one Sequence per Routine
        doDet       = 0

        toDo = {}
        for wrd in ("doThresh", "doRand","doDet"):
            toDo[wrd] = locals()[wrd]
        plots.savefig_GLOBAL    = 1
        plots.showPlots_GLOBAL  = 0
        ### Create constant inputs to function
        jCon = nn.createjCon(info["sizeM"], info["jE"], info["jI"], info["K"])
        external = nn.createExt(info["sizeM"],info["extM"], info["K"], info["meanExt"])     
        thresh = nn.createThresh(info["sizeM"], info["threshM"], toDo["doThresh"])
        
        #valueFolder = describe(toDo, info,0) 
        (indiNeuronsDetailed,   
                activeOT, fireOT, nval0
        ) = nn.run_box( jCon, thresh, external, info, toDo,)

        nn.plot_machine(
            activeOT, fireOT, indiNeuronsDetailed,
            info, drw, toDo)




class MainWindow(QMainWindow):

    def __init__(self, *args, **kwargs):
        super(MainWindow, self).__init__(*args, **kwargs)
        self.setWindowTitle("Balanced Network Simulation")
        self.label = QLabel("THIS")
        self.inputLayout = InputLayout()
        self.label2 = QLabel("THIS2")
        
        widget = QWidget()
        widget.setLayout(self.inputLayout )
        self.setCentralWidget(widget)


class Color(QWidget):
    def __init__(self,color, *args, **kwargs):
        super(Color, self).__init__(*args, **kwargs) 
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.Window, QColor(color))
        self.setPalette(palette)
app = QApplication(sys.argv)
window = MainWindow()
window.show()
app.exec_()