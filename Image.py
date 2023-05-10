from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

import matplotlib.pyplot as plt
import numpy as np

class Image:
    def __init__(self):
        self.path = ""
        self.raw_data = None
                    
        # Get size
        self.width = 0
        self.height = 0
        
        # Get magnitude
        self.magnitude = list()
        # Get phase
        self.phase = list()
        # Get real
        self.real = list()
        # Get imag
        self.imaginary = list()

        self.pixmap = QPixmap("./img/default.png")
        self.components = dict()

    def browse(self):
        self.path, fileExtension = QFileDialog.getOpenFileName(None, 'Load Image', './', "Image File(*.png *.jpg *.jpeg)")
        fileExtension = self.path.split('.')[-1]

        if self.path == "":
            return

        if fileExtension in ['png','jpg', 'jpeg']:

            self.raw_data = plt.imread(self.path)
            if fileExtension != 'png':
                self.raw_data = self.raw_data.astype('float32')
                self.raw_data /= 255
                            
            # Get size
            self.shape = self.raw_data.shape
            self.width = self.shape[0]
            self.height = self.shape[1]
            
            # Fourier FFT
            self.fft = np.fft.fft2(self.raw_data)
            # Get magnitude
            self.magnitude = np.abs(self.fft)
            # Get phase
            self.phase = np.angle(self.fft)
            # Get real
            self.real = np.real(self.fft)
            # Get imag
            self.imaginary = np.imag(self.fft)

            self.pixmap = QPixmap(self.path)
            self.components = {
                'Magnitude':self.magnitude,
                'Phase':self.phase,
                'Real':self.real,
                'Imaginary': self.imaginary,
            }
 
    # Get pixmap
    def get_pixmap(self):
        return self.pixmap

    def mix(self, image_2: 'Image', type_1: str, type_2: str, component_1_ratio: float, component_2_ratio: float, mode: str) -> QPixmap:
        first = self.chooseComponent(type_1, component_1_ratio)
        second = image_2.chooseComponent(type_2, component_2_ratio)

        if mode == 'mag-phase':
            construct = np.real(np.fft.ifft2(np.multiply(first, second)))
        if mode == 'real-imag':
            construct = np.real(np.fft.ifft2(first + second))

        if np.max(construct) > 1.0:
            construct /= np.max(construct)
        plt.imsave('test.png', np.abs(construct))
        return QPixmap('test.png')

    # Get component
    def chooseComponent(self, type: str, ratio: float) -> np.ndarray:

        if type == "Magnitude":
            return self.components[type] * ratio
        elif type == "Phase":
            return np.exp(1j * self.components[type] * ratio)
        elif type == "Real":
            return self.components[type] * ratio
        elif type == "Imaginary":
            return 1j* self.components[type] * ratio
        elif type =="Uniform Magnitude" :
            return np.ones(shape=self.shape) * ratio
        elif type == "Uniform Phase":
            return np.exp(1j * np.zeros(shape=self.shape) * ratio)

    def get_component_pixmap(self, component: str) -> QPixmap:
        # component_val = np.dot(self.components[component][...,:3], [0.2989, 0.5870, 0.1140])
        if component != '':
            component_val = np.dot(self.components[component][...,:3], [0.2125, 0.7154, 0.0721])

        if component in ['Magnitude', 'Real']:
            plt.imsave('test.png',np.log(np.abs((component_val))), cmap='gray')
        elif component == 'Phase' :
            plt.imsave('test.png',(np.abs((component_val))), cmap='gray')
        elif component=='Imaginary':
            component_tmp  = np.where(component_val > 1.0e-10, component_val, 1.0e-10)
            result = np.where(component_val > 1.0e-10, np.log10(component_tmp), -10)
            plt.imsave('test.png',(np.abs(result)), cmap='gray')
        else :
            return QPixmap("./assets/placeholder.png")
        return QPixmap('test.png')
    # Check size equallity
    def compare(self, image: 'Image') -> bool:
        return self.width == image.width and self.height == image.height
