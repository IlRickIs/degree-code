import librosa
from scipy.signal import butter, filtfilt, cheby1, sosfilt
import numpy as np
import os
import pandas as pd
import costants as c
import matplotlib.pyplot as plt
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

dataset_path = 'datasets_raw/EMOVO/'
actors = os.listdir(dataset_path)
files = []

for actor in actors:
    actor_path = dataset_path + actor + '/'
    files += [actor_path + f for f in os.listdir(actor_path)]

for file in files:
    y, sr = load_audio_file(file)
    y = chebyshev_filter(y, sr)
    print(y)
    print(sr)
    print(y.shape)
    t = np.linspace(0, len(y)/sr, len(y))
    plt.plot(t, y)
    plt.show()
    break
   
