import librosa
from scipy.signal import butter, filtfilt, cheby1, sosfilt
import numpy as np
import os
import pandas as pd
import costants as c
import matplotlib.pyplot as plt
import os
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

def preprocess_audio_files(file_list: list, target_sr: int) -> list:
    """ Preprocess a list of audio files using a Chebyshev filter"""
    processed_signals = []
    for file_path in file_list:
        y, sr = load_audio_file(file_path, target_sr=target_sr)
        y_filtered = chebyshev_filter(y, sr)
        processed_signals.append(y_filtered)
    return processed_signals

def get_dataset_files(dataset_path: str) -> list:
    actors = os.listdir(dataset_path)
    #exclude non directories
    actors = [a for a in actors if os.path.isdir(dataset_path + a)]
    files = []
    print(actors)
    for actor in actors:
        actor_path = dataset_path + actor + '/'
        #exclude non wav files
        files += [actor_path + f for f in os.listdir(actor_path) if f.endswith('.wav')]
    
    return files



   
