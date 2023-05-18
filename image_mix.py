from PyQt5 import QtCore, QtGui, QtWidgets
from PIL import Image
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QHBoxLayout, QVBoxLayout , QShortcut, QLabel, QSlider, QStyle, QSizePolicy, QFileDialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as Canvas
import numpy as np 
import logging
from colorsys import hls_to_rgb
from numpy import pi


logging.basicConfig(filename='log,txt',level = logging.INFO,
                          format = '%(asctime)s:%(levelname)s:%(message)s'  )

class plot_image:
    def __init__(self,qvlayout):
        self.fig1  = plt.figure(figsize=(1.9,2.1),dpi=90)
        self.ax = self.fig1.add_subplot()
        self.plot_layout = Canvas(self.fig1)
        qvlayout.addWidget(self.plot_layout)
        self.ax.get_xaxis().set_visible(False)
        self.ax.get_yaxis().set_visible(False)
    def draw(self):
        self.plot_layout.draw()
        self.fig1.canvas.draw()
    def clear(self):
        self.fig1.clear()

class output_page:
    def __init__(self,Mixer):
        self.verticalLayoutWidget = QtWidgets.QWidget(Mixer)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 69, 441, 201))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.verticalLayoutWidget.setStyleSheet("background-color: powderblue;")  # Set background color

        self.out_put_layout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.out_put_layout.setContentsMargins(0, 0, 0, 0)
        self.out_put_layout.setObjectName("out_put_layout")
        self.widget = QtWidgets.QWidget(self.verticalLayoutWidget)
        self.widget.setObjectName("widget")
        self.comboBox_comp2_comp = QtWidgets.QComboBox(self.widget)
        self.comboBox_comp2_comp.setGeometry(QtCore.QRect(300, 110, 131, 31))
        self.comboBox_comp2_comp.setObjectName("comboBox_comp2_comp")
        self.comboBox_comp2_comp.setStyleSheet("background-color: #FEE4C5;")
        self.comboBox_comp2_imag = QtWidgets.QComboBox(self.widget)
        self.comboBox_comp2_imag.setGeometry(QtCore.QRect(160, 109, 131, 31))
        self.comboBox_comp2_imag.setObjectName("comboBox_comp2_imag")
        self.comboBox_comp2_imag.setStyleSheet("background-color: #FEE4C5;")
        self.Slider_comp1 = QtWidgets.QSlider(self.widget)
        self.Slider_comp1.setGeometry(QtCore.QRect(160, 60, 271, 22))
        self.Slider_comp1.setOrientation(QtCore.Qt.Horizontal)
        self.Slider_comp1.setObjectName("Slider_comp1")
        self.label_4 = QtWidgets.QLabel(self.widget)
        self.label_4.setGeometry(QtCore.QRect(140, 80, 47, 13))
        self.label_4.setObjectName("label_4")
        self.label_3 = QtWidgets.QLabel(self.widget)
        self.label_3.setGeometry(QtCore.QRect(10, 109, 151, 31))
        self.label_3.setObjectName("label_3")
        self.comboBox_comp1_imag = QtWidgets.QComboBox(self.widget)
        self.comboBox_comp1_imag.setGeometry(QtCore.QRect(160, 20, 131, 31))
        self.comboBox_comp1_imag.setObjectName("comboBox_comp1_imag")
        self.comboBox_comp1_imag.setStyleSheet("background-color: #FEE4C5;")
        self.label_2 = QtWidgets.QLabel(self.widget)
        self.label_2.setGeometry(QtCore.QRect(10, 20, 151, 31))
        self.label_2.setObjectName("label_2")
        self.label_5 = QtWidgets.QLabel(self.widget)
        self.label_5.setGeometry(QtCore.QRect(400, 80, 47, 13))
        self.label_5.setObjectName("label_5")
        self.comboBox_comp1_comp = QtWidgets.QComboBox(self.widget)
        self.comboBox_comp1_comp.setGeometry(QtCore.QRect(300, 21, 131, 31))
        self.comboBox_comp1_comp.setObjectName("comboBox_comp1_comp")
        self.comboBox_comp1_comp.setStyleSheet("background-color: #FEE4C5;")
        self.Slider_comp2 = QtWidgets.QSlider(self.widget)
        self.Slider_comp2.setGeometry(QtCore.QRect(160, 150, 271, 22))
        self.Slider_comp2.setOrientation(QtCore.Qt.Horizontal)
        self.Slider_comp2.setObjectName("Slider_comp2")
        self.label_7 = QtWidgets.QLabel(self.widget)
        self.label_7.setGeometry(QtCore.QRect(400, 170, 47, 13))
        self.label_7.setObjectName("label_7")
        self.label_6 = QtWidgets.QLabel(self.widget)
        self.label_6.setGeometry(QtCore.QRect(140, 170, 47, 13))
        self.label_6.setObjectName("label_6")
        self.out_put_layout.addWidget(self.widget)
        self.comboBox_comp2_comp.setCurrentText("Phase")
        self.validate_mixing(self.comboBox_comp1_comp,self.comboBox_comp2_comp)
        self.comboBox_comp1_comp.activated.connect(lambda:self.validate_mixing(self.comboBox_comp1_comp,self.comboBox_comp2_comp))
        #self.comboBox_comp2_comp.activated.connect(lambda:self.validate_mixing(self.comboBox_comp2_comp,self.comboBox_comp1_comp))
        
    
    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        self.label_2.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; font-weight:400;\">Component 1:</span></p></body></html>"))
        self.label_3.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:12pt; font-weight:400;\">Component 2:</span></p></body></html>"))
        self.label_4.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:8pt; color:#ff0000;\">0%</span></p></body></html>"))
        self.label_5.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:8pt; color:#00aa00;\">100%</span></p></body></html>"))
        self.label_6.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:8pt; color:#ff0000;\">0%</span></p></body></html>"))
        self.label_7.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:8pt; color:#00aa00;\">100%</span></p></body></html>"))

    def validate_mixing(self,image1_menu,image2_menu):
        if(image1_menu.currentText() == 'Magnitude') or (image1_menu.currentText() == 'uniMag'):
            array =["Phase","uniPhase"]
            image2_menu.clear()
            image2_menu.addItems(array)
            logging.info('image2_menu after choosing magnitude {}'.format(array) )
        elif(image1_menu.currentText() == 'Phase') or (image1_menu.currentText() == 'uniPhase') :
            array =["Magnitude","uniMag"]
            image2_menu.clear()
            image2_menu.addItems(array)
        elif(image1_menu.currentText() == 'Real'):
            array =["Imaginary"]
            image2_menu.clear()
            image2_menu.addItems(array)
        elif(image1_menu.currentText() == 'Imaginary' ):
            array =["Real"]
            image2_menu.clear()
            image2_menu.addItems(array)

    
    
    

# class drop_menu:
#     def __init__(self,combo,type):
#         self.type = type
#         self.combo=combo
#         self.reset_comboBox()
    
#     def reset_comboBox(self):
#         if(self.type==0):
#             self.combo.addItem('Magnitude')
#             self.combo.addItem('Phase')
#             self.combo.addItem('Real')
#             self.combo.addItem('Image')
#         elif(self.type==1):
#             self.combo.addItem('Magnitude')
#             self.combo.addItem('Phase')
#             self.combo.addItem('Real')
#             self.combo.addItem('Imaginary')
#             self.combo.addItem('uniMag')
#             self.combo.addItem('uniPhase')
#         elif(self.type == 2):
#             self.combo.addItem('output1')
#             self.combo.addItem('output2')
#         elif(self.type == 3):
#             self.combo.addItem('image1')
#             self.combo.addItem('image2')
class drop_menu:
    def __init__(self, combo, type):
        self.type = type
        self.combo = combo
        self.reset_comboBox()
    
    def reset_comboBox(self):
        options = {
            0: ['Magnitude', 'Phase', 'Real', 'Image'],
            1: ['Magnitude', 'Phase', 'Real', 'Imaginary', 'uniMag', 'uniPhase'],
            2: ['output1', 'output2'],
            3: ['image1', 'image2']
        }
        
        if self.type in options:
            items = options[self.type]
            self.combo.addItems(items)

class image_processing:
    def __init__(self,imag):
        self.or_image = imag
        image = np.array(imag)
        fft_image = np.fft.fft2(image)
        self.fft_image_phase_d = np.angle(fft_image)
        self.fft_image_mag_d = 20* np.log(np.abs(fft_image))
        self.fft_image_real_d = np.real(fft_image)
        self.fft_image_imag_d = np.imag(fft_image)
        self.fft_image_unimag = np.ones_like(self.fft_image_mag_d)
        self.fft_image_uniphase = np.zeros_like(self.fft_image_phase_d)
        self.image_comp_dic = { 'Magnitude':self.fft_image_mag_d,'Phase':self.fft_image_phase_d,'Imaginary':self.fft_image_imag_d, 'Real':self.fft_image_real_d , 'uniMag':self.fft_image_unimag , 'uniPhase':self.fft_image_uniphase }
        logging.info('image components have been set'.format())
        logging.info('array of ones {} '.format(self.fft_image_unimag))


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(990, 630)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.centralwidget.setStyleSheet("background-color: #FFEDD8;")
        

        self.IMAGE1 = QtWidgets.QGroupBox(self.centralwidget)
        self.IMAGE1.setGeometry(QtCore.QRect(10, 10, 461, 281))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setWeight(50)
        self.IMAGE1.setFont(font)
        self.IMAGE1.setObjectName("IMAGE1")
        self.IMAGE1.setStyleSheet("background-color: powderblue;") 
        

        self.image1_pathbox = QtWidgets.QTextBrowser(self.IMAGE1)
        self.image1_pathbox.setGeometry(QtCore.QRect(10, 30, 211, 31))
        self.image1_pathbox.setObjectName("image1_pathbox")
        self.image1_pathbox.setStyleSheet("background-color: #FEE4C5;")

        self.image1_pushbutton = QtWidgets.QPushButton(self.IMAGE1)
        self.image1_pushbutton.setGeometry(QtCore.QRect(230, 30, 75, 23))
        self.image1_pushbutton.setObjectName("image1_pushbutton")
        self.image1_pushbutton.setStyleSheet("background-color: #FEE4C5;")

        self.image1_comboBox = QtWidgets.QComboBox(self.IMAGE1)
        self.image1_comboBox.setGeometry(QtCore.QRect(320, 30, 111, 22))
        self.image1_comboBox.setObjectName("image1_comboBox")
        self.image1_comboBox.setStyleSheet("background-color: #FEE4C5;")

        

        self.horizontalLayoutWidget = QtWidgets.QWidget(self.IMAGE1)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(20, 79, 191, 191))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        

        self.image1_plot = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.image1_plot.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.image1_plot.setContentsMargins(0, 0, 0, 0)
        self.image1_plot.setObjectName("image1_plot")
        

        self.horizontalLayoutWidget_2 = QtWidgets.QWidget(self.IMAGE1)
        self.horizontalLayoutWidget_2.setGeometry(QtCore.QRect(240, 80, 191, 191))
        self.horizontalLayoutWidget_2.setObjectName("horizontalLayoutWidget_2")
        

        self.image1_plot_comp = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_2)
        self.image1_plot_comp.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.image1_plot_comp.setContentsMargins(0, 0, 0, 0)
        self.image1_plot_comp.setObjectName("image1_plot_comp")
        

        self.Mixer = QtWidgets.QGroupBox(self.centralwidget)
        self.Mixer.setGeometry(QtCore.QRect(510, 10, 461, 281))
        font = QtGui.QFont()
        font.setPointSize(9)
        font.setWeight(50)
        self.Mixer.setFont(font)
        self.Mixer.setObjectName("Mixer")
        self.Mixer.setStyleSheet("background-color: powderblue;") 
        

        self.label = QtWidgets.QLabel(self.Mixer)
        self.label.setGeometry(QtCore.QRect(60, 20, 151, 21))
        self.label.setObjectName("label")

        self.output_detect_combo = QtWidgets.QComboBox(self.Mixer)
        self.output_detect_combo.setGeometry(QtCore.QRect(220, 20, 111, 22))
        self.output_detect_combo.setObjectName("output_detect_combo")
        self.output_detect_combo.setStyleSheet("background-color: #FEE4C5;")
        
    
        self.label_8 = QtWidgets.QLabel(self.centralwidget)
        self.label_8.setGeometry(QtCore.QRect(580, 340, 111, 41))
        self.label_8.setObjectName("label_8")

        self.IMAGE2 = QtWidgets.QGroupBox(self.centralwidget)
        self.IMAGE2.setGeometry(QtCore.QRect(10, 320, 461, 281))
        font = QtGui.QFont()
        font.setPointSize(12)
        font.setWeight(50)
        self.IMAGE2.setFont(font)
        self.IMAGE2.setObjectName("IMAGE2")
        self.IMAGE2.setStyleSheet("background-color: powderblue;") 

        self.image2_pathbox = QtWidgets.QTextBrowser(self.IMAGE2)
        self.image2_pathbox.setGeometry(QtCore.QRect(10, 30, 211, 31))
        self.image2_pathbox.setObjectName("image2_pathbox")
        self.image2_pathbox.setStyleSheet("background-color: #FEE4C5;")

        self.image2_pushbutton = QtWidgets.QPushButton(self.IMAGE2)
        self.image2_pushbutton.setGeometry(QtCore.QRect(230, 30, 75, 23))
        self.image2_pushbutton.setObjectName("image2_pushbutton")
        self.image2_pushbutton.setStyleSheet("background-color: #FEE4C5;")

        self.image2_comboBox = QtWidgets.QComboBox(self.IMAGE2)
        self.image2_comboBox.setGeometry(QtCore.QRect(320, 30, 111, 22))
        self.image2_comboBox.setObjectName("image2_comboBox")
        self.image2_comboBox.setStyleSheet("background-color: #FEE4C5;")

        self.horizontalLayoutWidget_3 = QtWidgets.QWidget(self.IMAGE2)
        self.horizontalLayoutWidget_3.setGeometry(QtCore.QRect(20, 79, 191, 191))
        self.horizontalLayoutWidget_3.setObjectName("horizontalLayoutWidget_3")

        self.image2_plot = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_3)
        self.image2_plot.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.image2_plot.setContentsMargins(0, 0, 0, 0)
        self.image2_plot.setObjectName("image2_plot")

        self.horizontalLayoutWidget_4 = QtWidgets.QWidget(self.IMAGE2)
        self.horizontalLayoutWidget_4.setGeometry(QtCore.QRect(240, 80, 191, 191))
        self.horizontalLayoutWidget_4.setObjectName("horizontalLayoutWidget_4")

        self.image2_plot_comp = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_4)
        self.image2_plot_comp.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.image2_plot_comp.setContentsMargins(0, 0, 0, 0)
        self.image2_plot_comp.setObjectName("image2_plot_comp")

        self.label_9 = QtWidgets.QLabel(self.centralwidget)
        self.label_9.setGeometry(QtCore.QRect(800, 340, 111, 41))
        self.label_9.setObjectName("label_9")

        self.horizontalLayoutWidget_5 = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget_5.setGeometry(QtCore.QRect(550, 389, 191, 191))
        self.horizontalLayoutWidget_5.setObjectName("horizontalLayoutWidget_5")
        
        
        self.output_1 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_5)
        self.output_1.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.output_1.setContentsMargins(0, 0, 0, 0)
        self.output_1.setObjectName("output_1")
        

        self.horizontalLayoutWidget_6 = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget_6.setGeometry(QtCore.QRect(770, 390, 191, 191))
        self.horizontalLayoutWidget_6.setObjectName("horizontalLayoutWidget_6")
        self.output_2 = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget_6)
        self.output_2.setSizeConstraint(QtWidgets.QLayout.SetFixedSize)
        self.output_2.setContentsMargins(0, 0, 0, 0)
        self.output_2.setObjectName("output_2")

        MainWindow.setCentralWidget(self.centralwidget)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.output_page_1 = output_page(self.Mixer)
        self.output_page_2 = output_page(self.Mixer)
        self.output_detect_combo.setCurrentText("output1")
        
        self.output_detect_combo.activated.connect(self.output_channel)

        self.image_file_1 =0
        self.image_file_2 =0
        self.image1_pushbutton.clicked.connect(lambda:self.getfile(self.image1_pathbox,self.image_file_1))
        self.image2_pushbutton.clicked.connect(lambda:self.getfile(self.image2_pathbox,self.image_file_2))
        self.array_obj =[]
        

        array_plot_widgets=[self.image1_plot,self.image1_plot_comp,self.image2_plot,self.image2_plot_comp,self.output_1,self.output_2]
        for widget_ in array_plot_widgets:
            self.array_obj.append(plot_image(widget_))
       
       
        self.menu_array =[(self.image1_comboBox,0),(self.image2_comboBox,0),(self.output_detect_combo,2),
        (self.output_page_1.comboBox_comp1_imag,3),(self.output_page_1.comboBox_comp2_imag,3),(self.output_page_1.comboBox_comp1_comp,1),(self.output_page_1.comboBox_comp2_comp,1),
        (self.output_page_2.comboBox_comp1_imag,3),(self.output_page_2.comboBox_comp2_imag,3),(self.output_page_2.comboBox_comp1_comp,1),(self.output_page_2.comboBox_comp2_comp,1)] 
        for box,type in self.menu_array:
            drop_menu(box,type)


        self.image_prss_1 = 0
        self.image_prss_2 = 0  
        self.image1_comboBox.activated.connect(lambda:self.ploting_image(self.array_obj[0],self.array_obj[1],self.image_prss_1,self.image1_comboBox))
        self.image2_comboBox.activated.connect(lambda:self.ploting_image(self.array_obj[2],self.array_obj[3],self.image_prss_2,self.image2_comboBox))
        self.arr_page =[self.output_page_1.Slider_comp1,self.output_page_1.Slider_comp2,self.output_page_2.Slider_comp1,self.output_page_2.Slider_comp2 ]
        for page in self.arr_page:
            self.setting_slider(page)
        
        self.output_page_1.comboBox_comp1_comp.activated.connect(lambda:self.MixingFunc(self.output_page_1))
        self.output_page_1.comboBox_comp1_imag.activated.connect(lambda:self.MixingFunc(self.output_page_1))
        self.output_page_1.comboBox_comp2_comp.activated.connect(lambda:self.MixingFunc(self.output_page_1))
        self.output_page_1.comboBox_comp2_imag.activated.connect(lambda:self.MixingFunc(self.output_page_1))
        self.output_page_2.comboBox_comp1_comp.activated.connect(lambda:self.MixingFunc(self.output_page_2))
        self.output_page_2.comboBox_comp1_imag.activated.connect(lambda:self.MixingFunc(self.output_page_2))
        self.output_page_2.comboBox_comp2_comp.activated.connect(lambda:self.MixingFunc(self.output_page_2))
        self.output_page_2.comboBox_comp2_imag.activated.connect(lambda:self.MixingFunc(self.output_page_2))

        self.output_page_1.Slider_comp1.valueChanged.connect(lambda:self.MixingFunc(self.output_page_1))
        self.output_page_1.Slider_comp2.valueChanged.connect(lambda:self.MixingFunc(self.output_page_1))
        self.output_page_2.Slider_comp1.valueChanged.connect(lambda:self.MixingFunc(self.output_page_2))
        self.output_page_2.Slider_comp2.valueChanged.connect(lambda:self.MixingFunc(self.output_page_2))

        self.output_channel()
        

        self.retranslateUi(MainWindow)
        self.output_page_1.retranslateUi(MainWindow)
        self.output_page_2.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)


    def output_channel(self):
        if((self.image_file_1 == 0 ) or (self.image_file_2==0)):
            self.output_page_2.verticalLayoutWidget.hide()
            self.output_page_1.verticalLayoutWidget.hide()
        elif(self.output_detect_combo.currentText()=='output1'):
            self.output_page_2.verticalLayoutWidget.hide()
            self.output_page_1.verticalLayoutWidget.show()
            self.MixingFunc(self.output_page_1)
        elif(self.output_detect_combo.currentText()=='output2'):
            self.output_page_1.verticalLayoutWidget.hide()
            self.output_page_2.verticalLayoutWidget.show()
            self.MixingFunc(self.output_page_2)

    def getfile(self,textBrowser,image_data):
        for index in range(3):
            self.array_obj[index].ax.clear()
        file_path = QFileDialog.getOpenFileName(filter="JPEG (*.jpeg);;PNG (*.png);;JPG (*.jpg)")[0]
        textBrowser.setText(file_path)
        im = Image.open( file_path , mode = 'r'  )
        if (image_data == self.image_file_1) and (self.image_file_2==0):
            image_data = im 
            textBrowser.setText(file_path)
            self.image_file_1 = image_data
            self.image_prss_1 = image_processing(image_data)
            self.ploting_image(self.array_obj[0],self.array_obj[1],self.image_prss_1,self.image1_comboBox)
        elif (image_data == self.image_file_1) and (self.image_file_2 != 0):
            self.image_file_2 = 0
            self.image2_pathbox.clear()
            image_data =  im
            textBrowser.setText(file_path)
            self.image_prss_1 = image_processing(image_data)
            self.ploting_image(self.array_obj[0],self.array_obj[1],self.image_prss_1,self.image1_comboBox)
            self.array_obj[2].ax.clear()
            self.array_obj[3].ax.clear()
        elif(image_data == self.image_file_2) and (self.image_file_1 != 0 ):
            if(im.size == self.image_file_1.size):
                image_data = im
                textBrowser.setText(file_path)
                self.image_file_2 = image_data
                self.image_prss_2 = image_processing(self.image_file_2)  
                self.ploting_image(self.array_obj[2],self.array_obj[3],self.image_prss_2,self.image2_comboBox)
            else:
                Warning = "the image isn't the same size"
                textBrowser.setText(Warning)
        elif(image_data == self.image_file_2) and (self.image_file_1 == 0 ):
            self.image_file_1 = im
            self.image1_pathbox.setText(file_path)
            self.image_prss_1 = image_processing(self.image_file_1)  
            self.ploting_image(self.array_obj[0],self.array_obj[1],self.image_prss_1,self.image1_comboBox)
        
    def ploting_image(self,obj_img,obj_comp,image_data,comp):
        obj_img.ax.imshow(image_data.or_image)
        if (comp.currentText()=="Magnitude"):
            comp_image = image_data.fft_image_mag_d

            logging.info('comp_image {}'.format(comp_image))
            
            Image = comp_image/np.amax(comp_image)
            comp_image = np.clip(Image, 0, 1)

            logging.info('comp_image after clipping {}'.format(comp_image))

            obj_comp.ax.imshow((comp_image))
        elif (comp.currentText()=="Phase"):
            comp_image = image_data.fft_image_phase_d
            Image = comp_image/np.amax(comp_image)
            comp_image = np.clip(Image, 0, 1)
            obj_comp.ax.imshow((comp_image))
        elif (comp.currentText()=="Real"):
            comp_image = image_data.fft_image_real_d
            Image = comp_image/np.amax(comp_image)
            comp_image = np.clip(Image, 0, 1)
            obj_comp.ax.imshow(comp_image)
        elif (comp.currentText()=="Image"):
            comp_image = image_data.fft_image_imag_d
            obj_comp.ax.imshow((comp_image).astype('uint64'),cmap='gray')
        obj_img.draw()
        obj_comp.draw()

   

 
        
    def setting_slider(self,set_slider):

        set_slider.setStyleSheet("""
        QSlider {
            background-color: #F0F0F0;
            height: 10px;
            border-radius: 5px;
        }
        
        QSlider::groove:horizontal {
            background-color: #E0E0E0;
            height: 5px;
            border-radius: 2px;
        }
        
        QSlider::handle:horizontal {
            background-color: powderblue;
            width: 20px;
            margin: -5px 0;
            border-radius: 10px;
        }
        
        QSlider::sub-page:horizontal {
            background-color: #000000;
            border-radius: 2px;
        }
    """)
        
        set_slider.setTickPosition(QSlider.TicksRight)
        set_slider.setTickInterval(10)
        set_slider.setSingleStep(10)
        set_slider.setValue(100)
        set_slider.setMinimum(0)
        set_slider.setMaximum(100)
    

   
    def MixingFunc(self,output_channel):
        ccombined = [0]
        slider1_value = (output_channel.Slider_comp1.value())/100
        slider2_value = (output_channel.Slider_comp2.value())/100
        output_pannel = 0
        if(output_channel == self.output_page_1):
            output_pannel = self.array_obj[4]
        elif(output_channel == self.output_page_2):
            output_pannel = self.array_obj[5]
        
        images = {'image1':self.image_prss_1 , 'image2':self.image_prss_2} 
        image_key_1 = output_channel.comboBox_comp1_imag.currentText()
        image_key_2 = output_channel.comboBox_comp2_imag.currentText()
        comp_key_1 = output_channel.comboBox_comp1_comp.currentText()
        comp_key_2 = output_channel.comboBox_comp2_comp.currentText()
        choice_1 = images.get(image_key_1)
        choice_1 = choice_1.image_comp_dic.get(comp_key_1)
        logging.info('Choice 1 is {}'.format(choice_1))
        if(image_key_1=='image1'):
            if(comp_key_1 == 'uniMag' ):
                choice_1_compl =images.get('image2').image_comp_dic.get('Magnitude')
            elif(comp_key_1 == 'uniPhase' ):
                choice_1_compl =images.get('image2').image_comp_dic.get('Phase')
            else:
                choice_1_compl =images.get('image2').image_comp_dic.get(comp_key_1)
        elif (image_key_1=='image2'):
            if(comp_key_1 == 'uniMag' ):
                choice_1_compl =images.get('image1').image_comp_dic.get('Magnitude')
            elif(comp_key_1 == 'uniPhase' ):
                choice_1_compl =images.get('image1').image_comp_dic.get('Phase')
            else:
                choice_1_compl =images.get('image1').image_comp_dic.get(comp_key_1)
            
        
        choice_2 = images.get(image_key_2)
        choice_2 = choice_2.image_comp_dic.get(comp_key_2)
        logging.info('Choice 2 is {}'.format(choice_2))

        if(image_key_2=='image1'):
            if(comp_key_2 == 'uniMag' ):
                choice_2_compl =images.get('image2').image_comp_dic.get('Magnitude')
            elif(comp_key_2 == 'uniPhase' ):
                choice_2_compl =images.get('image2').image_comp_dic.get('Phase')
            else:
                choice_2_compl =images.get('image2').image_comp_dic.get(comp_key_2)
        elif (image_key_2=='image2'):
            if(comp_key_2 == 'uniMag' ):
                choice_2_compl =images.get('image1').image_comp_dic.get('Magnitude')
            elif(comp_key_2 == 'uniPhase' ):
                choice_2_compl =images.get('image1').image_comp_dic.get('Phase')
            else:
                choice_2_compl =images.get('image1').image_comp_dic.get(comp_key_2)
    
        if ((comp_key_1 == 'Magnitude')or(comp_key_1 == 'uniMag')) and ((comp_key_2 == 'Phase' )or(comp_key_2 == 'uniPhase' ) ):
            fft_mag = np.array(choice_1)
            fft_mag_compl=np.array(choice_1_compl)
            fft_phase = np.array(choice_2)
            fft_phase_compl=np.array(choice_2_compl)
            ccombined = np.multiply(((fft_mag*slider1_value)+((1-slider1_value)*fft_mag_compl)), np.exp(1j*((fft_phase*slider2_value)+((1-slider2_value)*fft_phase_compl))))
           
            logging.info('component 1 is {} and component 2 is {} and their combined array is {} '.format(comp_key_1,comp_key_2, ccombined) )
        
        elif ((comp_key_2 == 'Magnitude')or(comp_key_2 == 'uniMag')) and ((comp_key_1 == 'Phase' )or(comp_key_1 == 'uniPhase' ) ):
            fft_mag = np.array(choice_2)
            fft_mag_compl=np.array(choice_2_compl)
            fft_phase = np.array(choice_1)
            fft_phase_compl=np.array(choice_1_compl)
            ccombined = np.multiply(((fft_mag*slider2_value)+((1-slider2_value)*fft_mag_compl)), np.exp(1j*((fft_phase*slider1_value)+((1-slider1_value)*fft_phase_compl))))
        elif (comp_key_1 == 'Real' ) and (comp_key_2 == 'Imaginary'):
            fft_real = np.array(choice_1)
            fft_real_compl = np.array(choice_1_compl)
            fft_imag = np.array(choice_2)
            fft_imag_compl = np.array(choice_2_compl)
            ccombined = ((fft_real*slider1_value)+((1-slider1_value)*fft_real_compl)) + ((1j * ((fft_imag*slider2_value)+((1-slider2_value)*fft_imag_compl)) ))
        elif (comp_key_2 == 'Real' ) and (comp_key_1 == 'Imaginary'):
            fft_real = np.array(choice_2)
            fft_real_compl = np.array(choice_2_compl)
            fft_imag = np.array(choice_1)
            fft_imag_compl = np.array(choice_1_compl)
            ccombined = ((fft_real*slider2_value)+((1-slider2_value)*fft_real_compl)) + ((1j * ((fft_imag*slider1_value)+((1-slider1_value)*fft_imag_compl))))
        else:
            fft_mag = np.array(choice_1)
            fft_phase = np.array(choice_2)
            ccombined = np.multiply(fft_mag*slider1_value, np.exp(1j*fft_phase*slider2_value))
        ccombined = np.fft.ifft2(ccombined)
        ccombined = ccombined.astype('longlong')
        logging.info('Combined array after ifft {}'.format(ccombined))
        output_pannel.ax.imshow((ccombined))
        output_pannel.draw()


        
 




    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MIXER"))
        self.IMAGE1.setTitle(_translate("MainWindow", "IMAGE 1"))
        self.image1_pushbutton.setText(_translate("MainWindow", "Browse"))
        self.Mixer.setTitle(_translate("MainWindow", "MIXER"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:9pt; font-weight:600; \">MIXER OUTPUT of:</span></p></body></html>"))
        self.label_8.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:10pt; font-weight:600; \">Output 1</span></p></body></html>"))
        self.IMAGE2.setTitle(_translate("MainWindow", "IMAGE 2"))
        self.image2_pushbutton.setText(_translate("MainWindow", "Browse"))
        self.label_9.setText(_translate("MainWindow", "<html><head/><body><p align=\"center\"><span style=\" font-size:10pt; font-weight:600; \">Output 2</span></p></body></html>"))

if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    image_mixer = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(image_mixer)
    image_mixer.show()
    sys.exit(app.exec_())