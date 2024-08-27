import matlab.engine
import os
import pandas as pd
from scipy.signal.windows import hamming
import librosa

def extract_mfccs(signals: list, fs: int) -> pd.DataFrame:
    """Extract MFCCs from a list of audio signals"""
    n_mfcc = 13
    windowDuration = 0.025
    overlapPercentage = 0.5

    windowLenght = round(windowDuration * fs)
    overlap = round(windowLenght * overlapPercentage)
    window = hamming(windowLenght)

    def get_column_names(n_mfcc):
        column_names = []
        for i in range(1, n_mfcc+1):
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
    return features_df


def extract_features(signals: list, fs: int) -> list:
    """Extract features from a list of audio signals, this function writes the features to a file"""
    eng = matlab.engine.start_matlab()
    eng.cd(r'matlab', nargout=0)
    features = eng.extract_features(signals, fs)
    mfccs = extract_mfccs(signals, fs)
    #write features to file
    features_df = pd.DataFrame(columns=['pitch', 'energy', 'zcr', 'spectralKurtosis', 'spectralSkewness'], 
                                data=features)
    features_df = pd.concat([features_df, mfccs], axis=1)
    eng.quit()
    return features_df

def parse_filenames(files: list, dataset: int) -> pd.DataFrame: 
    """Parse the filenames of the audio files
    dataset: 0 for EMOVO, 1 for RAVDESS"""
    df = pd.DataFrame(columns=['actor', 'emotion'])

    if dataset == 0:
        for file in files:
            filename = os.path.basename(file)
            actor = filename.split('-')[1]
            #create a dictionary to map the emotions to the corresponding number
            emotions = {'neu': 1, 'gio': 3, 'tri': 4, 'rab': 5, 'pau': 6, 'dis': 7, 'sor': 8}
            emotion = emotions[filename.split('-')[0]]
            df = df._append({'actor': actor, 'emotion': emotion}, ignore_index=True)
        return df
    
    elif dataset == 1:
        for file in files:
            filename = os.path.basename(file)
            filename = filename[:-4]
            actor_mapping = {}
            for i in range(1, 25):
                if i % 2 == 0:
                    actor_mapping[i] = f'f{i//2}'
                else:
                    actor_mapping[i] = f'm{(i+1)//2}'
            actor = actor_mapping[int(filename.split('-')[6])]

            emotions = {'01': 1, '02': 2, '03': 3, '04': 4, '05': 5, '06': 6, '07': 7, '08': 8}
            emotion = emotions[filename.split('-')[2]]
            df = df._append({'actor': actor, 'emotion': emotion}, ignore_index=True)
        return df