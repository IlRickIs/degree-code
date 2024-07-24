import matlab.engine
import os
import pandas as pd

def extract_features(signals: list, fs: int) -> list:
    """Extract features from a list of audio signals, this function writes the features to a file"""
    eng = matlab.engine.start_matlab()
    eng.cd(r'matlab', nargout=0)
    features = eng.extract_features(signals, fs)
    #write features to file
    features_df = pd.DataFrame(columns=['pitch', 'energy', 'zcr', 'spectralKurtosis', 'spectralSkewness'], 
                                data=features)
    eng.quit()
    return features_df