from PyQt5 import QtWidgets, QtCore, uic, QtGui, QtPrintSupport
from pyqtgraph import PlotWidget, plot
from PyQt5.uic import loadUiType
from PyQt5.QtWidgets import *   
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from os import path
import pyqtgraph as pg
import queue as Q
import pandas as pd
import numpy as np
import sys
import os
from PIL import Image
from PyQt5.QtGui import QPixmap
import cv2
import numpy as np
from matplotlib import pyplot as plt


class ImageModel():

    """
    A class that represents the ImageModel
    """

    def __init__(self, imgPath: str,id):
        """
        :param imgPath: absolute path of the image
        """
        self.imgPath = imgPath
        self.img = cv2.imread(self.imgPath, flags=cv2.IMREAD_GRAYSCALE).T
        self.imgShape = self.img.shape
        self.fourier = np.fft.fft2(self.img)
        self.real = np.real(self.fourier)
        self.imaginary = np.imag(self.fourier)
        self.magnitude = np.abs(self.fourier)
        self.mag_spectrum = np.log10(self.magnitude)
        self.phase = np.angle(self.fourier)
        self.uniformMagnitude = np.ones(self.img.shape)
        self.uniformPhase = np.zeros(self.img.shape)
        self.component_list=[self.mag_spectrum,self.phase,self.real,self.imaginary]

    def mix(self, imageToBeMixed, magnitudeOrRealRatio, phaesOrImaginaryRatio, mode):
        """
        a function that takes ImageModel object mag ratio, phase ration and
        return the magnitude of ifft of the mix
        return type ---> 2D numpy array
        """

        w1 = magnitudeOrRealRatio
        w2 = phaesOrImaginaryRatio
        mixInverse = None

        # Create a dictionary of the possible modes and their corresponding parameters.
        modes = {
            "MagnitudeAndPhase": {
                "M1": self.magnitude,
                "M2": imageToBeMixed.magnitude,
                "P1": self.phase,
                "P2": imageToBeMixed.phase,
            },
            "UniMagnitudeAndPhase": {
                "M1": self.uniformMagnitude,
                "M2": self.uniformMagnitude,
                "P1": self.phase,
                "P2": imageToBeMixed.phase,
            },
            "PhaseAndMagnitude": {
                "M1": self.magnitude,
                "M2": imageToBeMixed.magnitude,
                "P1": self.phase,
                "P2": imageToBeMixed.phase,
            },
            "PhaseAndUniMagnitude": {
                "M1": self.uniformMagnitude,
                "M2": self.uniformMagnitude,
                "P1": self.phase,
                "P2": imageToBeMixed.phase,
            },
            "UniPhaseAndMagnitude": {
                "M1": self.magnitude,
                "M2": imageToBeMixed.magnitude,
                "P1": self.uniformPhase,
                "P2": self.uniformPhase,
            },
            "MagnitudeAndUniPhase": {
                "M1": self.magnitude,
                "M2": imageToBeMixed.magnitude,
                "P1": self.uniformPhase,
                "P2": self.uniformPhase,
            },
            "RealAndImaginary": {
                "R1": self.real,
                "R2": imageToBeMixed.real,
                "I1": self.imaginary,
                "I2": imageToBeMixed.imaginary,
            },
            "ImaginaryAndReal": {
                "R1": self.real,
                "R2": imageToBeMixed.real,
                "I1": self.imaginary,
                "I2": imageToBeMixed.imaginary,
            },
        }

        # Get the parameters for the current mode.
        parameters = modes[mode]

        # Calculate the magnitude mix.
        if mode in ["MagnitudeAndPhase", "UniMagnitudeAndPhase"]:
            magnitudeMix = w1 * parameters["M1"] + (1-w1) * parameters["M2"]
            phaseMix = (1-w2) * parameters["P1"] + w2 * parameters["P2"]
            combined = magnitudeMix * np.exp(1j * phaseMix)
        elif mode in ["UniPhaseAndMagnitude", "PhaseAndMagnitude"]:
            magnitudeMix = (1-w2) * parameters["M1"] + w2 * parameters["M2"]
            phaseMix = w1 * parameters["P1"] + (1 - w1) * parameters["P2"]
            combined = magnitudeMix * np.exp(1j * phaseMix)
        elif mode == "RealAndImaginary":
            realMix = w1 * parameters["R1"] + (1 - w1) * parameters["R2"]
            imaginaryMix = (1 - w2) * parameters["I1"] + w2 * parameters["I2"]
            combined = realMix + imaginaryMix * 1j
        else:
            realMix = (1 - w2) * parameters["R1"] + w2 * parameters["R2"]
            imaginaryMix = w1 * parameters["I1"] + (1 - w1) * parameters["I2"]
            combined = realMix + imaginaryMix * 1j   

            

        # Combine the magnitude and phase mixes.
        

        # Calculate the magnitude of the inverse Fourier transform.
        mixInverse = np.real(np.fft.ifft2(combined))

        return abs(mixInverse)
