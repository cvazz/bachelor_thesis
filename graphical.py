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
    "size":         "Size",        
    "K":            "Connection Value K",
    "timer":        "Time",
    "jE":           "Weight J_IE",
    "jI":           "Weight J_II",
    "tau":          "Relative Update time Tau ",
    "threshE":      "Excitatory threshold",
    "threshI":      "Inhibitory threshold",
    "meanStartActi":"meanActivation at time 0",
    "extE":         "External Input to Excitatory ExtE",
    "extI":         "External Input to Inhibitory ExtI",
    "meanExt":      "External mean activation", 
    "doThresh":     "Shape of threshold distribution",
    "doUpdating":   "Update Mechanism",
    "pIndiExt":     "Plot Input to a Single Neuron",
    "nDistri":      "Plot Distribution of Firing Rate",
    "newMeanOT":    "Plot Mean Activation Over Time",
    "nInter":       "Plot Interspike Intervals",
    "nInter_log":   "Plot Interspike Intervals (logarithmic scale)",
    "dots2":        "Plot Input to a Single Neuron",
}

desc_dict = {
    "size":         "Size of each cluster of neurons, excitatory and inhibitory",        
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
    "doThresh":     "Define shape of threshold distribution",
    "doUpdating":   "Define Update Mechanism",
    "pIndiExt":     "Plot Input to a Single Neuron",
    "nDistri":      "Plot Distribution of Firing Rate",
    "newMeanOT":    "Plot Mean Activation Over Time",
    "nInter":       "Plot Interspike Intervals",
    "nInter_log":   "Plot Interspike Intervals (logarithmic scale)",
    "dots2":        "Plot Input to a Single Neuron",
}

def make_drw():
    pass

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
        
        self.name, self.label = initQt(name)
        # self.label = QLabel(name_dict[name]+ ":\t")
        # self.label.setAlignment(Qt.AlignLeft)
        # self.label.setFixedWidth(250)
        # self.label.setToolTip("Wasdf;asldk;asdf;klsdf")

        self.val = default
        # self.name = name
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
        

        self.name, self.label= initQt(name)
        # lbl_txt = name_dict[name]
        self.val = default
        # self.label = QLabel(lbl_txt+ ":\t")
        # self.label.setAlignment(Qt.AlignLeft)
        # self.label.setFixedWidth(250)

        self.numbox = QDoubleSpinBox()
        self.numbox.setRange(min_range, max_range)
        self.numbox.valueChanged.connect(lambda x: self.numChange(x))
        self.numbox.setValue(default)

        self.addWidget(self.label)
        self.addWidget(self.numbox)

def initQt(name):
    lbl_txt     = name_dict[name]
    label = QLabel(lbl_txt+ ":\t")
    label.setAlignment(Qt.AlignLeft)
    label.setFixedWidth(250)
    label.setToolTip(desc_dict[name])
    newfont = QFont("Times", 16)
    label.setFont(newfont)
    return name, label

class Haken(QHBoxLayout):
    def __init__(self,name = "Error", default = True, *args, **kwargs):
        super(QHBoxLayout, self).__init__(*args, **kwargs)

        self.name, self.label = initQt(name)
        check  = QCheckBox()
        check.setChecked(default)
        self.val = default
        check.stateChanged.connect(lambda:self.choice(check))

        self.addWidget(self.label)
        self.addWidget(check)

    def choice(self, box):
        self.val = box.isChecked()
        print(self.val)

class Dropdown(QHBoxLayout):
    def __init__(self,name = "Eat Healthy", strings = ["Missing",], *args, **kwargs):
        super(QHBoxLayout, self).__init__(*args, **kwargs)

        self.name, self.label = initQt(name)
        dropThresh  = QComboBox()
        for string in strings:
            dropThresh.addItem(string)
        default = strings[0]
        self.val = default
        dropThresh.activated[str].connect(self.choice)

        self.addWidget(self.label)
        self.addWidget(dropThresh)

    def choice(self, text):
        self.val = text
        print(self.val)
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
    drop_dict = {
        "doThresh":     ["bounded", "constant", "gaussian"],
        "doUpdating":   ["stochastic","strict","deterministic"]
    }
    check_dict = {
    "pIndiExt":         1,
    "nDistri":          1,
    "newMeanOT":        1,
    "nInter":           1,
    "nInter_log":       1,
    "dots2":            1,
    }
    def __init__(self, *args, **kwargs):
        super(QVBoxLayout, self).__init__(*args, **kwargs)

        for key,val in self.range_dict.items():
            if val[1]>10:   layout  = ValueUnit(key,*val)
            else:           layout  = MiniUnit (key,*val)
            layout.setObjectName("inp_%s" %key)
            self.addLayout(layout)
        for key,txt in self.drop_dict.items():
            drop = Dropdown(key,txt)
            drop.setObjectName("inp_%s" %key)
            self.addLayout(drop)
        for key,default in self.check_dict.items():
            check = Haken(key,default)
            check.setObjectName("inp_%s" %key)
            self.addLayout(check)

        exec_button = QPushButton("Run")
        exec_button.clicked.connect(self.run)
        self.addWidget(exec_button, alignment = Qt.AlignRight)

    def make_toDo(self,items):
        drop = {}
        for w in items:
            drop[w.name] = w.val

        doThresh    = "constant" #"constant", "gauss", "bound"
        doThresh    = drop["doThresh"]
        doRand      = 0     #Only one Sequence per Routine
        doDet       = 0
        doUpdating  = drop["doUpdating"]
        if drop["doUpdating"] == "stochastic":
            doDet           = 0
            doRand          = 1
        elif drop["doUpdating"] == "deterministic":
            doDet           = 0
            doRand          = 0
        elif drop["doUpdating"] == "strict":
            doDet           = 1
            doRand          = 0

        toDo = {}
        for wrd in ("doThresh", "doRand","doDet","doUpdating"):
            toDo[wrd] = locals()[wrd]
        return toDo

    def make_info(self, items):
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
        return info

    def run(self):
        
        items = (self.itemAt(i) for i in range(self.count()) 
                if isinstance(self.itemAt(i),ValueUnit)
                or isinstance(self.itemAt(i),MiniUnit))

        info = self.make_info(items)

        drw_items = (self.itemAt(i) for i in range(self.count()) 
                if isinstance(self.itemAt(i),Haken)) 
        drw = {}
        for w in drw_items:
            drw[w.name] = w.val

        drop_items = (self.itemAt(i) for i in range(self.count()) 
                if isinstance(self.itemAt(i),Dropdown)) 
        toDo = self.make_toDo(drop_items)

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
        
        self.principal = QHBoxLayout()
        self.principal.addLayout(self.inputLayout )
        loc ="../figs/testreihe_200110_1644_S3_K2_m1_t10_rY/IndiExt.png"
        self.label = QLabel("Hallo")
        self.label.setText("Heasf;lkasdf")
        img = QPixmap('image.jpg')
        scaled_img = img.scaled(self.label.size(), Qt.KeepAspectRatio)
        self.label.setPixmap(scaled_img)
        self.principal.addWidget(self.label)
        widget = QWidget()
        widget.setLayout(self.principal)
        self.setCentralWidget(widget)

class Principal(QHBoxLayout):

    def __init__(self, *args, **kwargs):
        super(QVBoxLayout, self).__init__(*args, **kwargs)
        self.principal = QHBoxLayout
        self.principal.addLayout(self.inputLayout )
        loc ="../figs/testreihe_200110_1644_S3_K2_m1_t10_rY/IndiExt.png"


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
"""
Bild auf rechte Seite
Freeze while processing
Namen kurz mit Erklärung
"""