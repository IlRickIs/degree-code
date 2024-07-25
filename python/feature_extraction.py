import matlab.engine
import os
import pandas as pd
from scipy.signal.windows import hamming
import librosa

def extract_mfccs(signals: list, fs: int) -> list:
    """Extract MFCCs from a list of audio signals"""
    n_mfcc = 13
    windowDuration = 0.025
    overlapPercentage = 0.5

    windowLenght = round(windowDuration * fs)
    overlap = round(windowLenght * overlapPercentage)
    window = hamming(windowLenght)

    def get_column_names(n_mfcc):
        column_names = []
        for i in range(1, 14):
            column_names.append(f'mean{i}')
            column_names.append(f'median{i}')
            column_names.append(f'std{i}')
            column_names.append(f'min{i}')
            column_names.append(f'max{i}')
        return column_names
    
    def get_mfcc_features(signal):
        mfccs = librosa.feature.mfcc(y = signal, sr = fs, n_mfcc=n_mfcc, n_fft=windowLenght, hop_length=overlap, window=window)
        mfccs = mfccs.T
        mfccs_df = pd.DataFrame(mfccs)
        mfcc_features = []
    
        for _, coeff in mfccs_df.items():
            mean = coeff.mean()
            std = coeff.std()
            minimum = coeff.min()
            maximum = coeff.max()
            median = coeff.median()
            #find if there are any NaN values
            mfcc_features += [mean,median, std, minimum, maximum]
        return mfcc_features
    
    features_df = pd.DataFrame(index=range(0,len(signals)),columns=get_column_names(n_mfcc))
    for i,signal in enumerate(signals):
        signal_features = get_mfcc_features(signal)
        features_df.loc[i] = signal_features
    
    print(features_df)
        
    return features_df


def extract_features(signals: list, fs: int) -> list:
    """Extract features from a list of audio signals, this function writes the features to a file"""
    eng = matlab.engine.start_matlab()
    eng.cd(r'matlab', nargout=0)
    features = eng.extract_features(signals, fs)
    #mfccs = extract_mfccs(signals, fs)
    #write features to file
    features_df = pd.DataFrame(columns=['pitch', 'energy', 'zcr', 'spectralKurtosis', 'spectralSkewness'], 
                                data=features)
    eng.quit()
    return features_df