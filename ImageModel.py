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

    def __init__(self, imgPath: str, id):
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
        self.component_list = [self.mag_spectrum,
                               self.phase, self.real, self.imaginary]

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
                "magnitude": self.magnitude,
                "magnitude2": imageToBeMixed.magnitude,
                "phase": self.phase,
                "phase2": imageToBeMixed.phase,
            },
            "UniMagnitudeAndPhase": {
                "magnitude": self.uniformMagnitude,
                "magnitude2": imageToBeMixed.uniformMagnitude,
                "phase": self.phase,
                "phase2": imageToBeMixed.phase,
            },
            "PhaseAndMagnitude": {
                "magnitude": self.magnitude,
                "magnitude2": imageToBeMixed.magnitude,
                "phase": self.phase,
                "phase2": imageToBeMixed.phase,
            },
            "PhaseAndUniMagnitude": {
                "magnitude": self.uniformMagnitude,
                "magnitude2": imageToBeMixed.uniformMagnitude,
                "phase": self.phase,
                "phase2": imageToBeMixed.phase,
            },
            "UniPhaseAndMagnitude": {
                "magnitude": self.magnitude,
                "magnitude2": imageToBeMixed.magnitude,
                "phase": self.uniformPhase,
                "phase2": imageToBeMixed.uniformPhase,
            },
            "MagnitudeAndUniPhase": {
                "magnitude": self.magnitude,
                "magnitude2": imageToBeMixed.magnitude,
                "phase": self.uniformPhase,
                "phase2": imageToBeMixed.uniformPhase,
            },
            "RealAndImaginary": {
                "real": self.real,
                "real2": imageToBeMixed.real,
                "imaginary": self.imaginary,
                "imaginary2": imageToBeMixed.imaginary,
            },
            "ImaginaryAndReal": {
                "real": self.real,
                "real2": imageToBeMixed.real,
                "imaginary": self.imaginary,
                "imaginary2": imageToBeMixed.imaginary,
            },
        }

        # Get the parameters for the current mode.
        parameters = modes[mode]

        # Calculate the magnitude mix.
        if mode in ["MagnitudeAndPhase", "UniMagnitudeAndPhase", "PhaseAndMagnitude", "UniPhaseAndMagnitude"]:
            magnitudeMix = w1 * \
                parameters["magnitude"] + (1-w1) * parameters["magnitude2"]
            phaseMix = (1-w2) * parameters["phase"] + w2 * parameters["phase2"]
            combined = magnitudeMix * np.exp(1j * phaseMix)
        elif mode in ["RealAndImaginary", "ImaginaryAndReal"]:
            realMix = w1 * parameters["real"] + (1 - w1) * parameters["real2"]
            imaginaryMix = (
                1 - w2) * parameters["imaginary"] + w2 * parameters["imaginary2"]
            combined = realMix + imaginaryMix * 1j
        else:
            raise ValueError("Invalid mode")

        # Calculate the magnitude of the inverse Fourier transform.
        mixInverse = np.real(np.fft.ifft2(combined))

        return abs(mixInverse)

        # Get the parameters for the current mode.
