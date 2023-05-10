from ast import If
from matplotlib import image
from Image import Image
from Gui import Ui_MainWindow

from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

# importing numpy and pandas
import numpy as np
import pandas as pd
import scipy

# importing pyqtgraph as pg
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from pyqtgraph.dockarea import *

# importing sys package
import sys
import os

# Logging configuration
import logging
logging.basicConfig(filename="errlog.log",
                    filemode="a",
                    format="(%(asctime)s)  | %(name)s | %(levelname)s:%(message)s",
                    datefmt="%d  %B  %Y , %H:%M:%S",
                    level=logging.INFO)

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Initilizer Variables
        self.image1 = None
        self.image2 = None

        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.componentList = [
            self.ui.component_1_select,
            self.ui.component_2_select,
            self.ui.component_1_type,
            self.ui.component_2_type,
        ]

        self.show()
        self.connect()

    def browse(self, imgNum: int):
        if imgNum == 1:
            self.image1 = Image()
            self.image1.browse()
            self.ui.image_1_original.setPixmap(self.image1.get_pixmap())
        else:
            self.image2 = Image()
            self.image2.browse()
            if self.image2.compare(self.image1):
                self.ui.image_2_original.setPixmap(self.image2.get_pixmap())
            else:
                QMessageBox.critical(self,"Error", "Please choose the same size as the first image.")

    # Show output
    def showOutput(self):
        if self.ui.component_1_type.currentText() in ["Real", "Imaginary"] and self.ui.component_2_type.currentText() in ["Real", "Imaginary"]:
            mode = 'real-imag'
        elif self.ui.component_1_type.currentText() in ["Magnitude", "Phase", "Uniform Magnitude", "Uniform Phase"] and self.ui.component_2_type.currentText() in ["Magnitude", "Phase", "Uniform Magnitude", "Uniform Phase"]:
            mode = 'mag-phase'
        else:
            return
        
        pixmapMix = self.image1.mix(self.image2, 
                                    self.ui.component_1_type.currentText(), 
                                    self.ui.component_2_type.currentText(), 
                                    self.ui.component_1_slider.value() / 100, 
                                    self.ui.component_2_slider.value() / 100,
                                    mode)

        if self.ui.output_select.currentText() == "Output 1":
            self.ui.output_1.setPixmap(pixmapMix)
        elif self.ui.output_select.currentText() == "Output 2":
            self.ui.output_2.setPixmap(pixmapMix)
        else:
            print("error")

    
    def connect(self):
        self.ui.action_open_image_1.triggered.connect(lambda: self.browse(1))
        self.ui.action_open_image_2.triggered.connect(lambda: self.browse(2))

        self.ui.image_1_pick.currentTextChanged.connect(lambda: self.chooseComponent(1,self.ui.image_1_pick.currentText()))
        self.ui.image_2_pick.currentTextChanged.connect(lambda: self.chooseComponent(2,self.ui.image_2_pick.currentText()))

        for comp in self.componentList:
            comp.currentTextChanged.connect(self.showOutput)

        self.ui.component_1_slider.sliderReleased.connect(self.showOutput)
        self.ui.component_2_slider.sliderReleased.connect(self.showOutput)
    
        # Exit
        self.ui.action_exit.triggered.connect(self.exit)

    def chooseComponent(self, num, value):
        if num == 1:
            self.ui.image_1_after_filter.setPixmap(self.image1.get_component_pixmap(value))
        else:
            self.ui.image_2_after_filter.setPixmap(self.image2.get_component_pixmap(value))

    # Exit the application
    def exit(self):
        exitDlg = QMessageBox.critical(self,
        "Exit the application",
        "Are you sure you want to exit the application?",
        buttons=QMessageBox.Yes | QMessageBox.No,
        defaultButton=QMessageBox.No)

        if exitDlg == QMessageBox.Yes:
            # Exit the application
            sys.exit()


def main_window():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = MainWindow()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main_window()

