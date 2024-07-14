import librosa
from scipy.signal import butter, filtfilt, cheby1, sosfilt
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
import costants as c
# Constants

def chebyshev_filter(signal, fs, lowcut=c.LOWCUT, highcut=c.HIGHCUT, order=c.ORDER, rp=c.RP):
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    sos = cheby1(order, rp, [low, high], btype='band', output='sos')
    return sosfilt(sos, signal)

def load_audio_file(file_path, ismono=True, target_sr=c.TARGET_SR):
    y, sr = librosa.load(file_path, sr=target_sr, mono=ismono)
    return y, sr
